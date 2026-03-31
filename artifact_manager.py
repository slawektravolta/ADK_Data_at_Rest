"""
Artifact Manager — artifact_manager.py

Centralny moduł zarządzania plikami dla wieloagentowego systemu ADK.
Implementuje architekturę "Data-at-Rest":
  - Pliki od użytkownika trafiają natychmiast do Artifact Service
  - session.state przechowuje tylko lekki rejestr metadanych (_ARTIFACT_REGISTRY)
  - Modele LLM widzą wyłącznie tekstowe znaczniki plików
  - Subagenci ładują pliki binarne ephemerally (JIT Injection) tuż przed LLM

Requires: GOOGLE_API_KEY (or Vertex AI credentials) for enrichment calls.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import uuid
from typing import Any, Optional

import google.genai.types as types
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.tool_context import ToolContext

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REGISTRY_KEY = "_ARTIFACT_REGISTRY"          # dict: filename → metadata
_HASH_INDEX_KEY = "_ARTIFACT_HASH_INDEX"      # dict: content_hash → filename (dedup)
_TASK_ARTIFACTS_KEY = "_CURRENT_TASK_ARTIFACTS"  # list: [{filename, version}] for JIT
_MAX_TEXT_CHARS = 50_000  # truncation limit for text artifacts

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _data_hash(data: bytes) -> str:
    """MD5 hex prefix — detects duplicate inline_data across history turns."""
    return hashlib.md5(data).hexdigest()[:16]


def _strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences (```json ... ```, ```csv ... ```, etc.) if present."""
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1:]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3].rstrip()
    return stripped


def _is_json_mime(mime: str) -> bool:
    return mime in ("application/json", "text/json") or mime.endswith("+json")


def _looks_like_json(raw: bytes) -> bool:
    try:
        return raw.lstrip()[:1] in (b"{", b"[")
    except Exception:
        return False


async def save_artifact_for_ctx(ctx: Any, filename: str, artifact: types.Part) -> int:
    """
    Save artifact using the appropriate API for the given context type.
    CallbackContext / ToolContext: ctx.save_artifact(filename, artifact)
    InvocationContext: ctx.artifact_service.save_artifact(...)
    """
    if hasattr(ctx, "save_artifact"):
        return await ctx.save_artifact(filename, artifact)
    # InvocationContext — use the underlying service directly
    return await ctx.artifact_service.save_artifact(
        app_name=ctx.app_name,
        user_id=ctx.user_id,
        session_id=ctx.session.id,
        filename=filename,
        artifact=artifact,
    )


async def load_artifact_for_ctx(
    ctx: Any, filename: str, version: Optional[int] = None
) -> Optional[types.Part]:
    """
    Load artifact using the appropriate API for the given context type.
    """
    if hasattr(ctx, "load_artifact"):
        return await ctx.load_artifact(filename, version=version)
    # InvocationContext — use the underlying service directly
    return await ctx.artifact_service.load_artifact(
        app_name=ctx.app_name,
        user_id=ctx.user_id,
        session_id=ctx.session.id,
        filename=filename,
        version=version,
    )


# ---------------------------------------------------------------------------
# Callback: _capture_attachments  (root agent before_model_callback)
# ---------------------------------------------------------------------------


async def _capture_attachments(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """
    Root-agent before_model_callback — "Data-at-Rest" ingestion.

    For each inline_data part found in user messages:
      - Uses content hash to detect files already saved in previous turns (dedup)
      - Saves new files to Artifact Service and registers them in _ARTIFACT_REGISTRY
      - Replaces every inline_data with a text marker so the LLM never receives raw bytes
    """
    registry: dict = callback_context.state.get(_REGISTRY_KEY) or {}
    hash_index: dict = callback_context.state.get(_HASH_INDEX_KEY) or {}

    for i, content in enumerate(llm_request.contents or []):
        if content.role != "user":
            continue

        new_parts = []
        content_changed = False

        for part in content.parts or []:
            if not part.inline_data:
                new_parts.append(part)
                continue

            mime: str = part.inline_data.mime_type or "application/octet-stream"
            data: bytes = part.inline_data.data or b""
            h = _data_hash(data)

            if h in hash_index:
                # Already saved in a previous turn — replace with existing marker
                existing = hash_index[h]
                version = registry.get(existing, {}).get("version", 0)
                new_parts.append(
                    types.Part(text=f"[Załącznik systemowy: {existing} (v{version})]")
                )
                content_changed = True
                continue

            # New file — determine extension and save
            ext = mimetypes.guess_extension(mime) or ".bin"
            if mime == "text/csv":
                ext = ".csv"
            elif _is_json_mime(mime) or (mime == "text/plain" and _looks_like_json(data)):
                ext = ".json"
            elif mime.startswith("image/"):
                # mimetypes sometimes returns .jpeg; normalise to .jpg
                ext = "." + mime.split("/")[1].replace("jpeg", "jpg")

            filename = f"upload_{uuid.uuid4().hex[:8]}{ext}"
            artifact_part = types.Part(
                inline_data=types.Blob(mime_type=mime, data=data)
            )

            try:
                version = await save_artifact_for_ctx(
                    callback_context, filename, artifact_part
                )
                registry[filename] = {
                    "version": version,
                    "mime_type": mime,
                    "source": "user_upload",
                    "status": "ready",
                    "content_hash": h,
                    "tags": [mime.split("/")[0]],
                }
                hash_index[h] = filename
                new_parts.append(
                    types.Part(text=f"[Załącznik systemowy: {filename} (v{version})]")
                )
            except Exception as exc:
                new_parts.append(types.Part(text=f"[BŁĄD ZAPISU: {exc}]"))

            content_changed = True

        if content_changed:
            llm_request.contents[i] = types.Content(role=content.role, parts=new_parts)

    callback_context.state[_REGISTRY_KEY] = registry
    callback_context.state[_HASH_INDEX_KEY] = hash_index
    return None


# ---------------------------------------------------------------------------
# Tool: save_text_artifact  (universal text/JSON/CSV artifact saver)
# ---------------------------------------------------------------------------

_EXTENSION_MIME: dict[str, str] = {
    "json": "application/json",
    "csv": "text/csv",
    "txt": "text/plain",
    "md": "text/markdown",
    "xml": "application/xml",
    "html": "text/html",
}


async def save_text_artifact(
    content: str,
    file_prefix: str,
    extension: str,
    tags: list[str],
    tool_context: ToolContext,
) -> dict:
    """
    Uniwersalne narzędzie zapisu pliku tekstowego do Artifact Service.

    Czyści zawartość z markdown fences, generuje unikalne imię pliku,
    zapisuje artefakt i rejestruje go w _ARTIFACT_REGISTRY.

    Args:
        content:     Treść pliku (może zawierać markdown fences — zostaną usunięte).
        file_prefix: Prefix nazwy pliku, np. "spectral_analysis", "report".
        extension:   Rozszerzenie bez kropki, np. "json", "csv", "txt".
        tags:        Lista tagów metadanych, np. ["json", "spectral", "analysis"].

    Returns:
        Dict z kluczami: status, filename, version, tags — lub status i error.
    """
    try:
        import datetime as _dt
        clean = _strip_markdown_fences(content)
        ext = extension.lstrip(".")
        mime = _EXTENSION_MIME.get(ext, f"text/{ext}")
        ts = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{file_prefix}_{ts}.{ext}"

        artifact_part = types.Part(
            inline_data=types.Blob(
                mime_type=mime,
                data=clean.encode("utf-8"),
            )
        )
        version = await tool_context.save_artifact(filename, artifact_part)

        registry: dict = tool_context.state.get(_REGISTRY_KEY) or {}
        registry[filename] = {
            "version": version,
            "mime_type": mime,
            "source": "generated_by_agent",
            "status": "ready",
            "tags": tags,
        }
        tool_context.state[_REGISTRY_KEY] = registry

        return {"status": "saved", "filename": filename, "version": version, "tags": tags}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Callback: _inject_artifacts_before_llm  (subagent before_model_callback)
# ---------------------------------------------------------------------------


async def _inject_artifacts_before_llm(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """
    Subagent before_model_callback — JIT (Just-In-Time) file injection.

    Loads files listed in _CURRENT_TASK_ARTIFACTS from Artifact Service and
    prepends them to the last user message.  Immediately clears the pointer
    after injection to prevent state leakage on subsequent LLM calls.
    """
    task_refs: list = callback_context.session.state.get(_TASK_ARTIFACTS_KEY) or []
    if not task_refs:
        return None

    parts_to_inject = []
    for ref in task_refs:
        try:
            part = await load_artifact_for_ctx(
                callback_context, ref["filename"], version=ref.get("version")
            )
            if part and part.inline_data:
                parts_to_inject.append(part)
        except Exception as exc:
            parts_to_inject.append(
                types.Part(text=f"[Błąd ładowania {ref['filename']}: {exc}]")
            )

    if parts_to_inject and llm_request.contents:
        for i in range(len(llm_request.contents) - 1, -1, -1):
            if llm_request.contents[i].role == "user":
                new_parts = parts_to_inject + list(llm_request.contents[i].parts or [])
                llm_request.contents[i] = types.Content(role="user", parts=new_parts)
                break

    # CRITICAL: clear after injection — prevents re-injection on the next LLM call
    callback_context.session.state[_TASK_ARTIFACTS_KEY] = []
    return None


# ---------------------------------------------------------------------------
# Tool: set_task_artifacts  (root agent)
# ---------------------------------------------------------------------------


async def set_task_artifacts(filenames: list[str], tool_context: ToolContext) -> str:
    """
    Wskazuje pliki, które subagent ma załadować tuż przed wywołaniem LLM (JIT Injection).

    Wywołaj to narzędzie PRZED delegacją zadania do subagenta analitycznego.
    Subagent załaduje wskazane pliki ephemerally i natychmiast wyczyści wskaźnik.

    Args:
        filenames: Lista nazw plików z rejestru (np. ["upload_abc12345.jpg"]).

    Returns:
        Potwierdzenie lub ostrzeżenie o brakujących plikach.
    """
    registry: dict = tool_context.state.get(_REGISTRY_KEY) or {}
    task_list = []
    missing = []

    for fname in filenames:
        if fname in registry:
            task_list.append({
                "filename": fname,
                "version": registry[fname].get("version", 0),
            })
        else:
            missing.append(fname)

    tool_context.state[_TASK_ARTIFACTS_KEY] = task_list

    result = f"Ustawiono {len(task_list)} plik(ów) do JIT injection."
    if missing:
        result += (
            f" UWAGA: Pominięto nieistniejące pliki: {', '.join(missing)}."
            " Sprawdź nazwy w rejestrze."
        )
    return result


# ---------------------------------------------------------------------------
# Tools: lookup_registry + get_artifact_registry_summary + load_specific_text_artifact
# ---------------------------------------------------------------------------


async def get_artifact_registry_summary(tool_context: ToolContext) -> str:
    """
    Zwraca pełny rejestr artefaktów jako czytelny słownik JSON.

    Użyj tego narzędzia gdy chcesz samodzielnie przejrzeć wszystkie dostępne
    pliki i wybrać odpowiedni na podstawie metadanych (file_nature, user_intent,
    tags, source, mime_type, version).

    Returns:
        JSON ze wszystkimi wpisami rejestru lub "[Rejestr pusty]".
    """
    registry: dict = tool_context.state.get(_REGISTRY_KEY) or {}
    if not registry:
        return "[Rejestr pusty]"
    entries = [{"filename": k, **v} for k, v in registry.items()]
    return json.dumps(entries, ensure_ascii=False)


async def lookup_registry(query: str, tool_context: ToolContext) -> str:
    """
    Wyszukuje pliki w rejestrze artefaktów metodą full-text search.

    Sprawdza czy fraza 'query' występuje (case-insensitive) w nazwie pliku,
    tagach, file_nature, user_intent lub source.

    Args:
        query: Dowolna fraza, np. "spectral", "json", "obraz", "widmo".

    Returns:
        JSON z metadanymi pasujących plików lub "[Brak wyników]".
    """
    registry: dict = tool_context.state.get(_REGISTRY_KEY) or {}
    needle = query.lower()

    def _matches(filename: str, meta: dict) -> bool:
        haystack_parts = [
            filename,
            meta.get("file_nature", ""),
            meta.get("user_intent", ""),
            meta.get("source", ""),
            meta.get("mime_type", ""),
            " ".join(meta.get("tags", [])),
        ]
        return any(needle in part.lower() for part in haystack_parts)

    matches = [
        {"filename": k, **v}
        for k, v in registry.items()
        if _matches(k, v)
    ]
    return json.dumps(matches, ensure_ascii=False) if matches else "[Brak wyników]"


async def load_specific_text_artifact(filename: str, tool_context: ToolContext) -> str:
    """
    Wczytuje zawartość tekstowego artefaktu z Artifact Service (z truncation).

    Używaj tylko dla plików tekstowych (JSON, CSV, TXT).
    Dla binariów (obrazy) wymagana re-delegacja przez Root Agenta z set_task_artifacts.

    Args:
        filename: Nazwa pliku z rejestru artefaktów.

    Returns:
        Treść pliku (obcięta do 50 000 znaków) lub komunikat błędu.
    """
    registry: dict = tool_context.state.get(_REGISTRY_KEY) or {}
    meta = registry.get(filename)

    if not meta:
        return f"[Błąd: Plik {filename} nie istnieje w rejestrze]"

    mime = meta.get("mime_type", "")
    if not (mime.startswith("text/") or _is_json_mime(mime)):
        return (
            f"[Błąd Architektury: Plik {filename} jest binarny ({mime}). "
            "Wymagana re-delegacja przez Root Agenta z set_task_artifacts.]"
        )

    try:
        part = await load_artifact_for_ctx(
            tool_context, filename, version=meta.get("version")
        )
        if not part:
            return "[Błąd: Artefakt pusty]"

        if part.inline_data and part.inline_data.data:
            text = _strip_markdown_fences(
                part.inline_data.data.decode("utf-8", errors="replace")
            )
        elif part.text:
            text = _strip_markdown_fences(part.text)
        else:
            return "[Błąd: Artefakt bez treści tekstowej]"

        if len(text) > _MAX_TEXT_CHARS:
            return (
                text[:_MAX_TEXT_CHARS]
                + f"\n\n... [OSTRZEŻENIE SYSTEMOWE: Plik obcięty. "
                f"Pełna zawartość: {len(text)} znaków. "
                f"Widać pierwsze {_MAX_TEXT_CHARS}.]"
            )
        return text
    except Exception as exc:
        return f"[Błąd odczytu: {exc}]"

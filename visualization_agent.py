"""
Spectral Visualization Sub-agent — visualization_agent.py

BaseAgent implementation: no LLM calls, no loops.

Data-at-Rest flow:
  1. Reads JSON file reference from _CURRENT_TASK_ARTIFACTS in session state
  2. Loads the JSON from Artifact Service
  3. Validates structure and generates a matplotlib chart
  4. Saves the PNG chart to Artifact Service and registers it
  5. Yields a text event with [Załącznik systemowy: chart_spectral_XXX.png]
     so the ADK web UI renders a download button

Requires: pip install matplotlib
"""

from __future__ import annotations

import datetime
import io
import json
from typing import AsyncGenerator, Optional

import google.genai.types as types
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from my_agent.artifact_manager import (
    _REGISTRY_KEY,
    _TASK_ARTIFACTS_KEY,
    _strip_markdown_fences,
    load_artifact_for_ctx,
    save_artifact_for_ctx,
)

try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    _MATPLOTLIB_OK = True
except ImportError:
    _MATPLOTLIB_OK = False

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate(data: dict) -> Optional[str]:
    """Return error description or None if structure is valid."""
    for field in ("metadata", "data", "analysis_summary"):
        if field not in data:
            return f"Brak wymaganego pola: '{field}'"
    ds = data["data"]
    if not isinstance(ds, dict):
        return "Pole 'data' musi być obiektem JSON"
    for field in ("wavelengths", "energy_values"):
        if field not in ds:
            return f"Brak pola 'data.{field}'"
        if not isinstance(ds[field], list):
            return f"Pole 'data.{field}' musi być tablicą"
    wl, ev = ds["wavelengths"], ds["energy_values"]
    if min(len(wl), len(ev)) < 2:
        return "Za mało punktów (minimum 2)"
    return None


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------


def _build_chart(data: dict) -> bytes:
    """Generate the spectral chart and return as PNG bytes."""
    wl = [float(v) for v in data["data"]["wavelengths"]]
    ev = [float(v) for v in data["data"]["energy_values"]]
    # Truncate to the shorter array — LLM-generated data can have off-by-one mismatches
    n = min(len(wl), len(ev))
    wl, ev = wl[:n], ev[:n]
    summary = data.get("analysis_summary", {})
    peak_wl = summary.get("peak_wavelength")
    max_abs = summary.get("max_absorbance")
    units = data.get("metadata", {}).get("units", {})

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(wl, ev, color="#1565C0", linewidth=2, marker="o", markersize=4,
            label="Absorbancja")
    if peak_wl is not None:
        ax.axvline(x=float(peak_wl), color="#C62828", linestyle="--",
                   linewidth=1.5, label=f"Peak: {peak_wl} nm")
    if peak_wl is not None and max_abs is not None:
        x_range = max(wl) - min(wl)
        ax.annotate(
            f"{float(max_abs):.3f}",
            xy=(float(peak_wl), float(max_abs)),
            xytext=(float(peak_wl) + x_range * 0.06, float(max_abs)),
            fontsize=9, color="#C62828",
            arrowprops=dict(arrowstyle="->", color="#C62828"),
        )
    wl_unit = units.get("wavelength", "nm")
    en_unit = units.get("energy", "AU")
    ax.set_xlabel(f"Długość fali ({wl_unit})", fontsize=12)
    ax.set_ylabel(f"Absorbancja ({en_unit})", fontsize=12)
    ax.set_title("Widmo spektralne", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class SpectralVisualizationAgent(BaseAgent):
    """
    Deterministic BaseAgent — zero LLM calls, zero loops.

    Reads spectral JSON from Artifact Service (via _CURRENT_TASK_ARTIFACTS),
    generates a matplotlib chart, saves the PNG to Artifact Service, and
    yields a single Event with a text marker pointing to the artifact.
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:

        def _err(msg: str) -> Event:
            return Event(
                author=self.name,
                content=types.Content(
                    role="model", parts=[types.Part(text=msg)]
                ),
            )

        if not _MATPLOTLIB_OK:
            yield _err(
                "Błąd: biblioteka matplotlib nie jest zainstalowana. "
                "Uruchom: pip install matplotlib"
            )
            return

        # ── 1. Resolve JSON source from _CURRENT_TASK_ARTIFACTS ──────────────
        task_refs: list = ctx.session.state.get(_TASK_ARTIFACTS_KEY) or []
        json_ref = next(
            (r for r in task_refs if r.get("filename", "").endswith(".json")),
            None,
        )
        if not json_ref:
            yield _err(
                "Brak wskazanego pliku JSON w task artifacts. "
                "Upewnij się, że Root Agent wywołał set_task_artifacts lub "
                "lookup_registry + set_task_artifacts przed transferem."
            )
            return

        # ── 2. Load JSON from Artifact Service ────────────────────────────────
        try:
            part = await load_artifact_for_ctx(
                ctx, json_ref["filename"], version=json_ref.get("version")
            )
        except Exception as exc:
            yield _err(f"Błąd ładowania artefaktu '{json_ref['filename']}': {exc}")
            return

        if not part:
            yield _err(f"Artefakt '{json_ref['filename']}' nie istnieje.")
            return

        if part.inline_data and part.inline_data.data:
            raw_text = _strip_markdown_fences(
                part.inline_data.data.decode("utf-8", errors="replace")
            )
        elif part.text:
            raw_text = _strip_markdown_fences(part.text)
        else:
            yield _err("Artefakt JSON nie zawiera danych tekstowych.")
            return

        # Clear task artifacts after reading (prevent stale re-use)
        ctx.session.state[_TASK_ARTIFACTS_KEY] = []

        # ── 3. Parse and validate JSON ────────────────────────────────────────
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            yield _err(f"Nieprawidłowy JSON: {exc}")
            return

        validation_error = _validate(data)
        if validation_error:
            yield _err(f"Nieprawidłowa struktura JSON: {validation_error}")
            return

        # ── 4. Generate chart ─────────────────────────────────────────────────
        try:
            chart_bytes = _build_chart(data)
        except Exception as exc:
            yield _err(f"Błąd generowania wykresu: {exc}")
            return

        # ── 5. Save chart PNG to Artifact Service ─────────────────────────────
        ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        chart_filename = f"chart_spectral_{ts}.png"
        chart_part = types.Part(
            inline_data=types.Blob(mime_type="image/png", data=chart_bytes)
        )
        try:
            chart_version = await save_artifact_for_ctx(ctx, chart_filename, chart_part)
        except Exception as exc:
            yield _err(f"Błąd zapisu wykresu do Artifact Service: {exc}")
            return

        # ── 6. Yield result event with artifact_delta so ADK UI shows the chart ─
        summary = data.get("analysis_summary", {})
        peak_wl = summary.get("peak_wavelength", "N/A")
        max_abs = summary.get("max_absorbance", "N/A")

        # Build updated registry for state_delta
        registry: dict = ctx.session.state.get(_REGISTRY_KEY) or {}
        registry[chart_filename] = {
            "version": chart_version,
            "mime_type": "image/png",
            "source": "visualization_agent",
            "status": "ready",
            "tags": ["image", "chart", "spectral"],
        }

        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        text=(
                            f"Wykres widma spektralnego wygenerowany pomyślnie.\n"
                            f"Peak: {peak_wl} nm | Max absorbancja: {max_abs}\n\n"
                            f"[Załącznik systemowy: {chart_filename} (v{chart_version})]"
                        )
                    ),
                ],
            ),
            actions=EventActions(
                artifact_delta={chart_filename: chart_version},
                state_delta={_REGISTRY_KEY: registry},
            ),
        )


visualization_agent = SpectralVisualizationAgent(
    name="visualization_agent",
    description=(
        "Generuje wykres widma spektralnego z danych JSON i zapisuje go jako artefakt PNG. "
        "Używaj gdy użytkownik przesyła plik JSON z widmem i prosi o wizualizację lub wykres."
    ),
)

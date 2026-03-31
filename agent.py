"""
Spectral Analysis Demo — ADK multi-agent application.

Architecture
------------
root_agent  (LlmAgent)
  ├── before_agent_callback: _init_session_defaults
  │     Initialises N_values default in session state.
  ├── before_model_callback: _capture_attachments  (from artifact_manager)
  │     Saves every inline attachment to Artifact Service, replaces it in the
  │     LLM request with a text marker [Załącznik systemowy: filename (vN)].
  │     Maintains _ARTIFACT_REGISTRY (metadata) and _ARTIFACT_HASH_INDEX (dedup).
  ├── AgentTool → spectral_analysis_agent
  │     ├── before_model_callback: _inject_artifacts_before_llm  (JIT)
  │     │     Loads the image indicated by _CURRENT_TASK_ARTIFACTS ephemerally
  │     │     into the LlmRequest, then immediately clears the pointer.
  │     └── Returns: raw JSON string (spectral data or error JSON)
  ├── Tool: set_task_artifacts
  │     Root agent calls this to point a subagent at specific artifact files
  │     before delegating (sets _CURRENT_TASK_ARTIFACTS).
  ├── Tool: save_text_artifact
  │     Universal text/JSON/CSV saver — replaces all domain-specific savers.
  │     Strips markdown fences, generates timestamped filename, registers in
  │     _ARTIFACT_REGISTRY.
  ├── Tool: lookup_registry / load_specific_text_artifact
  │     Subagent-autonomy tools (text file search and loading with truncation).
  └── sub_agents: [visualization_agent]
        BaseAgent that reads JSON from _CURRENT_TASK_ARTIFACTS, generates a
        matplotlib chart, saves the PNG to Artifact Service, and returns a
        text marker.

Why AgentTool for spectral_analysis_agent
-----------------------------------------
AgentTool creates an isolated text-only conversation for the subagent; the
image is injected ephemerally via before_model_callback and never written to
session events, so SSE serialisation always succeeds.

Assumptions
-----------
- Model: gemini-2.5-pro.
- N_values controls how many data points the parser generates (default 100).
- GOOGLE_API_KEY or Vertex AI credentials must be present in the environment.
"""

from __future__ import annotations

from typing import Optional

import google.genai.types as types
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.agent_tool import AgentTool
from my_agent.artifact_manager import (
    _REGISTRY_KEY,
    _TASK_ARTIFACTS_KEY,
    _capture_attachments,
    _inject_artifacts_before_llm,
    get_artifact_registry_summary,
    load_specific_text_artifact,
    lookup_registry,
    save_text_artifact,
    set_task_artifacts,
)
from my_agent.prompts import ROOT_AGENT_INSTRUCTION, SPECTRAL_SUBAGENT_INSTRUCTION
from my_agent.visualization_agent import visualization_agent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL = "gemini-2.5-pro"

# Number of data points the spectral parser should generate ({N_values} placeholder)
_STATE_N_VALUES = "N_values"
_DEFAULT_N_VALUES = 100

# ---------------------------------------------------------------------------
# Session initialisation callback
# ---------------------------------------------------------------------------


def _init_session_defaults(callback_context: CallbackContext) -> Optional[types.Content]:
    """Set session defaults once at agent startup."""
    if callback_context.state.get(_STATE_N_VALUES) is None:
        callback_context.state[_STATE_N_VALUES] = _DEFAULT_N_VALUES
    return None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

spectral_analysis_agent = LlmAgent(
    name="spectral_analysis_agent",
    model=_MODEL,
    description=(
        "Analyzes a spectral chart image using a multimodal LLM. "
        "Returns structured JSON data if the image is a spectral chart, "
        "or an error JSON otherwise."
    ),
    instruction=SPECTRAL_SUBAGENT_INSTRUCTION,
    before_model_callback=_inject_artifacts_before_llm,
    include_contents="none",
)

root_agent = LlmAgent(
    name="spectral_demo_assistant",
    model=_MODEL,
    instruction=ROOT_AGENT_INSTRUCTION,
    tools=[
        AgentTool(agent=spectral_analysis_agent),
        set_task_artifacts,
        save_text_artifact,
        get_artifact_registry_summary,
        # lookup_registry,
        load_specific_text_artifact,
    ],
    sub_agents=[visualization_agent],
    before_agent_callback=_init_session_defaults,
    before_model_callback=_capture_attachments,
)

# ADK_Data_at_Rest
Smart File Registry for Multi-Agent Systems. Decouple files from LLM context in Google ADK using the Artifacts-at-Rest pattern.

## Overview
This repository demonstrates an advanced file management system for **Google ADK (Agent Development Kit)** multi-agent applications. It eliminates the overhead of sending heavy binary data (Base64) directly within the conversation history, ensuring system stability and cost-efficiency.

By separating the **reasoning layer** (LLM) from the **data layer** (Storage), this architecture allows agents to handle large files, complex charts, and datasets without bloating the prompt context.

---

## Key Components

* **`artifact_manager.py`**: The core module implementing the *Artifacts-at-Rest* pattern. It ensures files are saved immediately to the **Artifact Service**, while agents interact only with a lightweight metadata registry in `session.state`. It utilizes **JIT (Just-In-Time) Injection** to ephemerally load binary data into the model context only at the exact moment of analysis.
* **`agent.py`**: Defines the system hierarchy, featuring a **Root Agent** (orchestrator) and specialized sub-agents.
* **`prompts.py`**: System instructions that define a rigorous scientific data extraction process for spectral charts.
* **`visualization_agent.py`**: A deterministic **BaseAgent** that generates `matplotlib` charts from JSON data without requiring additional LLM calls.

---

## Example Workflow

1.  **Analysis (Multimodal LLM)**: A user uploads a spectral chart image (**PNG**). The `spectral_analysis_agent` extracts scientific data into a structured **JSON** format.
2.  **Persistence**: The analysis result is saved as a new text artifact via the `save_text_artifact` tool, making it available for the rest of the session.
3.  **Visualization**: Upon user request, the `visualization_agent` reads the **JSON** file, generates a professional chart, and saves it as a new image artifact (**PNG**), which is then rendered in the UI for download.

---

## Benefits

* **Token Optimization**: Large images and datasets are not re-sent in every turn of the conversation history, drastically reducing costs.
* **System Stability**: Prevents **SSE (Server-Sent Events)** serialization failures by keeping binary blobs out of the event stream.
* **Enhanced Reasoning**: The metadata registry allows the Root Agent to perform **"Smart Selection"** of resources based on intent and nature, rather than just rigid filenames.


```bash
# Ensure you have the ADK and plotting dependencies
pip install google-adk matplotlib

---
name: testing-hayo-agent
description: Test the HAYO AI Agent Chainlit app end-to-end. Use when verifying UI, chat pipeline, tool execution, model switching, or new tool integrations.
---

# Testing HAYO AI Agent

## Environment Setup

1. Navigate to the project directory:
   ```bash
   cd "HAYO AI AGENT"
   ```

2. Create and activate the virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the Chainlit server:
   ```bash
   chainlit run app.py --port 8000 --headless
   ```
   The `--headless` flag prevents auto-opening a browser.

5. Open browser to `http://localhost:8000`

## Devin Secrets Needed

- `GOOGLE_API_KEY` — for Google Gemini model
- `ANTHROPIC_API_KEY` — for Claude model
- `OPENAI_API_KEY` — for GPT-4o model
- `DEEPSEEK_API_KEY` — for DeepSeek model (default provider)
- `GROQ_API_KEY` — for Groq model

At minimum, one API key is needed. DeepSeek is the default provider.
All keys should be in `HAYO AI AGENT/.env` file.

## Arabic Text Input

Chainlit's textarea does not accept direct `type` actions for Arabic text. Use clipboard paste:
```bash
echo -n "مرحبا" | xclip -selection clipboard
```
Then click the textarea and press `ctrl+v`. This properly triggers React state updates and enables the send button.

## Key Test Flows

### 1. UI Verification
- Check page title contains "HAYO AI Agent" (not "Assistant")
- Verify dark theme is applied (dark background)
- Verify capabilities section lists all tool categories
- Check that the sidebar panel is visible (chat history area)

### 2. Agent Chat Pipeline — Deduplication Check
- Send a simple Arabic message like "مرحبا"
- Agent should respond within 30 seconds
- **Critical**: Response text must appear EXACTLY ONCE in the chat bubble
- Pipeline stages should appear: "يفكر... | Thinking..." and/or "يراجع... | Reviewing..."
- Response should be in Arabic
- No internal markers should be visible: `CONVERSATIONAL_ONLY`, `NEW TASK BOUNDARY`, `TASK_COMPLETE`, separator lines (`───`, `━━━`)
- Use DOM verification to confirm:
  ```javascript
  // In browser console
  const allText = document.body.innerText;
  const markers = ['CONVERSATIONAL_ONLY', 'TASK_COMPLETE', 'NEW TASK BOUNDARY'];
  markers.forEach(m => { if (allText.includes(m)) console.log('MARKER FOUND:', m); });
  ```

### 3. Model Switching
- Type `/model google` (or `anthropic`, `openai`, `deepseek`, `groq`)
- Should show confirmation "تم تغيير النموذج بنجاح" with model name
- A new session ID is generated

### 4. Tool Execution
- After switching to a capable model (Google Gemini recommended), ask:
  "أنشئ مجلد جديد اسمه test_hayo في المسار /tmp/"
- Agent should show a plan, invoke `make_dir`, and report success
- Verify with `ls -la /tmp/test_hayo`

### 5. Tool Registration
- Code-level check:
  ```python
  from tools.registry import ALL_TOOLS
  print(len(ALL_TOOLS))  # Should be 109
  ```

### 6. Google Drive Graceful Fallback
- Code-level check:
  ```python
  from tools.gdrive_tools import gdrive_list
  result = gdrive_list.invoke({})
  # Should return setup instructions, not crash
  ```

### 7. Image Upload (Multimodal)
- Code-level verification of the image processing path:
  ```python
  import base64
  from langchain_core.messages import HumanMessage
  # Verify base64 encoding + multimodal HumanMessage construction
  ```
- UI-level image upload via Chainlit's paperclip button may not work with automated browser tools — Chainlit's React file input does not respond to DOM manipulation. Manual testing on the user's Windows machine is recommended for this feature.

## Known Issues & Workarounds

### Streaming Deduplication (Fixed in PR #16)
`stream_mode="messages"` in LangGraph emits both AIMessageChunk (streaming tokens) AND AIMessage (state update) for every LLM call. The fix in `_run_graph()` tracks which nodes streamed chunks and skips their duplicate AIMessage. Additionally, internal markers like `CONVERSATIONAL_ONLY` can arrive as individual tokens ("CON", "V", "ERS"...) — the line-buffer approach accumulates text and filters at newline boundaries.

If the repetition bug reappears, check:
1. `_nodes_with_chunks` set in `_run_graph()` — is it tracking nodes correctly?
2. `_line_buf` — is text being accumulated and flushed at newlines?
3. `isinstance` checks — AIMessageChunk must be checked BEFORE AIMessage (since AIMessageChunk inherits from AIMessage)

### Chainlit File Upload Automation
Chainlit's hidden file input (`#upload-drop-input`) does not respond to:
- Direct `select_file` browser automation
- DOM manipulation (setting value + dispatching change event)
- Playwright `set_input_files` via CDP

The React framework intercepts the change event and the Chainlit upload handler is not triggered. For image upload testing, use code-level verification of the base64 encoding path, or test manually.

### Agent Response Times
The LangGraph pipeline (Planner → Worker → Reviewer) takes 15-30 seconds per message. Allow adequate wait time between sending a message and checking results.

### Model Recommendations for Testing
- **Google Gemini (gemini-2.5-flash)**: Best for tool execution and image analysis testing — fast, supports vision
- **DeepSeek**: Default provider, good for conversational testing but may not support multimodal vision
- **Anthropic Claude**: Supports vision but may be slower

## Architecture Notes

- **Framework**: Chainlit (web chat UI) + LangGraph (agent orchestration)
- **Config**: `.chainlit/config.toml` — theme, layout, sidebar, app name
- **Agent nodes**: `agent/nodes.py` — PlannerNode → WorkerNode → ReviewerNode pipeline
- **Tools**: `tools/registry.py` — ALL_TOOLS list with 109 tools
- **Tool modules**: `tools/browser_tools.py`, `tools/github_tools.py`, `tools/gdrive_tools.py`, etc.
- **Streaming**: `app.py` `_run_graph()` function handles streaming output to Chainlit UI
- **Deduplication**: `_nodes_with_chunks` set + `_line_buf` line buffer in `_run_graph()`
- **Internal markers**: `_INTERNAL_MARKERS` tuple + `_filter_internal_tokens()` in `app.py`

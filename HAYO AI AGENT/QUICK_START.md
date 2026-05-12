# HAYO AI Agent — Quick Start Guide

## Installation & Setup

### Prerequisites
- Python 3.8+
- Windows OS (for full functionality)
- API keys (configure in `.env`)

### Quick Setup
```bash
cd "C:\Users\PT\Desktop\HAYO\ahmad-hayo-bot\HAYO AI AGENT"

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from core.agent_brain import get_brain; print(get_brain().get_memory_stats())"

# Run the agent
python main.py
```

## Basic Usage

### Interactive Mode
```bash
python main.py
```
Type tasks in English or Arabic:
```
> Open Chrome
> Download this PDF: https://example.com/file.pdf
> Create Excel file with sales data
> /quit
```

### Single Command Mode
```bash
python main.py --once "Open Chrome and download file.pdf"
```

## Model Configuration

### Switch AI Models
Edit `.env`:
```bash
# Anthropic Claude
MODEL_PROVIDER=anthropic
ANTHROPIC_AGENT_MODEL=claude-opus-4-7

# OR Google Gemini
MODEL_PROVIDER=google
GOOGLE_AGENT_MODEL=gemini-2.5-flash

# OR Groq
MODEL_PROVIDER=groq
GROQ_AGENT_MODEL=llama-3.3-70b-versatile
```

### Change Limits
Edit `.env`:
```bash
MAX_ITERATIONS=5000      # Increase for longer tasks
MAX_HISTORY=500          # More context retention
PS_TIMEOUT=300           # PowerShell timeout in seconds
```

## Advanced Features

### 1. Using the Brain System
```python
from core.agent_brain import get_brain

brain = get_brain()

# Get intelligent context for a task
prompt = brain.get_contextual_system_prompt("convert MP3 to WAV")

# Check tool reliability
reliability = brain.get_tool_reliability("download_file")

# Get learned solutions
solution = brain.get_best_solution_for_problem("file conversion")
```

### 2. Memory Management
```python
from core.memory_system import MemorySystem

memory = MemorySystem()

# Current working context
short_term = memory.get_short_term()  # 20 messages

# Recent history
medium_term = memory.get_medium_term()  # 100 messages

# Past insights
insights = memory.get_relevant_long_term("download file")
```

### 3. Learning System
```python
from core.learning_system import LearningSystem

learning = LearningSystem()

# Record a successful execution
learning.record_tool_execution(
    tool_name="download_file",
    parameters={"url": "https://..."},
    success=True,
    output="File downloaded successfully",
    execution_time_ms=2500
)

# Get reliability score
score = learning.get_tool_reliability("download_file")  # 0.95

# Get learning report
report = learning.get_learning_report()
```

### 4. Replit Integration
```python
# Open your Replit project
python main.py --once "Open my Replit project: username/project-name"

# Clone and work with Replit projects
python main.py --once "
1. Clone https://replit.com/@user/my-project
2. Read main.py
3. Fix syntax errors
4. Commit and push changes
"
```

## Workflow Examples

### Example 1: Download & Convert
```
User: "Download song.mp3 from YouTube and convert to WAV"

Agent:
1. [Planning] Analyze best path (search → download → convert)
2. [Learning] Recall previous YouTube conversions
3. [Execution] Execute optimal steps
4. [Learning] Record outcome for future tasks
```

### Example 2: Data Processing
```
User: "Download CSV file and convert to Excel"

Agent:
1. [Memory] Check if similar task done before
2. [Planning] Choose best download method
3. [Execution] Download → Convert
4. [Learning] Remember this solution path
```

### Example 3: Replit Development
```
User: "Fix bugs in my Replit project"

Agent:
1. [Dialogue] Understand intent (code debugging)
2. [Planning] Analyze multiple approaches
3. [Replit] Open project → Read files → Identify issues
4. [Memory] Use past debugging patterns
5. [Execution] Update files → Commit → Test
6. [Learning] Record solution for similar bugs
```

## Available Tools

### File Operations (9 tools)
- `read_file` - Read text files
- `write_file` - Create/update files
- `download_file` - Download from URL
- `move_file`, `copy_file` - File management
- `list_dir`, `search_files` - Directory operations

### Applications (4 tools)
- `open_app` - Launch applications
- `close_app` - Close applications
- `list_running_apps` - List running processes
- `focus_window` - Bring window to front

### Desktop Control (7 tools)
- `screen_screenshot` - Take screenshots
- `keyboard_type` - Type text
- `keyboard_hotkey` - Press key combinations
- `mouse_click`, `mouse_move` - Mouse control
- `wait` - Wait for specified time

### Browser (10 tools)
- `browser_open` - Open URLs
- `browser_screenshot` - Capture web pages
- `browser_click`, `browser_fill` - Web interaction
- `browser_get_text` - Extract text

### Replit (8 tools)
- `replit_open_project` - Open in browser
- `replit_read_file` - Read project files
- `replit_update_file` - Edit files
- `replit_git_commit` - Commit changes
- `replit_git_sync` - Push/pull
- `replit_run_project` - Execute locally

### Office (13 tools)
- `excel_create`, `excel_read`, `excel_edit` - Excel operations
- `word_create`, `word_read`, `word_edit` - Word operations
- `pdf_create`, `pdf_read`, `pdf_merge` - PDF operations

### File Conversion (3 tools)
- `convert_file` - Convert between formats
- `get_supported_formats` - List supported formats
- `check_conversion_support` - Check if conversion possible

### Download & Media (9 tools)
- `download_with_progress` - Download with progress tracking
- `chrome_search_and_open` - Google search & open
- `chrome_search_media_file` - Search and download media

### System (8 tools)
- `run_powershell` - Execute PowerShell commands
- `run_cmd` - Execute CMD commands
- `get_system_info` - System information
- `list_processes` - Running processes
- `kill_process` - Terminate processes

## Troubleshooting

### "Model not responding"
- Check API keys in `.env`
- Verify internet connection
- Check MODEL_PROVIDER setting

### "Tool execution failed"
- Check if required application is installed
- Review error logs in console
- Try a different approach (agent learns from failures)

### "Memory issues"
- System automatically manages memory with three-level system
- Reduce MAX_HISTORY if needed
- Agent learns to work with constraints

### "Tool not found"
- Run `python -c "from tools.registry import ALL_TOOLS; print([t.name for t in ALL_TOOLS])"`
- Verify tool is registered in `tools/registry.py`
- Check imports in tool file

## Performance Tips

1. **Reuse Learned Solutions**: Agent gets faster over time
2. **Clear Intentions**: More specific = better planning
3. **Batch Operations**: Request multiple related tasks
4. **Monitor Learning**: Check `agent_memory/` for insights

## Development

### Add Custom Tool
1. Create `tools/my_tools.py`
2. Add `@tool` decorator functions
3. Import in `tools/registry.py`
4. Add to `ALL_TOOLS` list

```python
from langchain_core.tools import tool

@tool
def my_tool(param: str) -> str:
    """Tool description"""
    return "result"
```

### Customize Memory Limits
Edit `core/memory_system.py`:
```python
self.SHORT_TERM_LIMIT = 20      # Current context
self.MEDIUM_TERM_LIMIT = 100    # Recent history
self.LONG_TERM_INSIGHT_LIMIT = 50  # Persistent insights
```

### Extend Brain System
All systems accessible via `AgentBrain`:
```python
brain = get_brain()
brain.memory           # Three-level memory
brain.learning         # Learning system
brain.planning         # Planning system
brain.dialogue         # Dialogue system
```

## Support & Documentation

- **Architecture**: See `SYSTEM_UPGRADE.md`
- **Code**: Comprehensive docstrings in each module
- **Examples**: Check `agent/nodes.py` for workflow integration

## License & Attribution

HAYO AI Agent — Locally executed intelligent assistant.
Powered by Claude Opus 4.7 and advanced agent systems.

---

**Ready to use?** Start with: `python main.py`

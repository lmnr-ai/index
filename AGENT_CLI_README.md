# Index Browser Agent CLI

A modern, interactive CLI for interacting with the Index browser agent, featuring:

- State persistence between sessions
- Support for follow-up messages after "give_control" actions
- Real-time streaming updates
- Beautiful terminal UI using Textual

## Installation

The CLI is included with the lmnr-index package:

```bash
pip install lmnr-index
```

## Usage

Run the interactive CLI:

```bash
index run
```

You can also provide an initial prompt:

```bash
index run --prompt "Go to news.ycombinator.com and summarize the top 3 articles"
```

To reset the agent state:

```bash
index reset
```

### Features

- **State Persistence**: The agent state is automatically saved and loaded between sessions
- **Follow-up Messaging**: Send follow-up messages to the agent after it requests human control
- **Real-time Updates**: See agent actions as they happen with streaming updates
- **Reset Session**: Clear the agent state and start fresh

### Keyboard Shortcuts

- `Ctrl+S`: Send a message
- `R`: Reset the agent state
- `Q`: Quit the application

## Configuration

Edit the configuration parameters in `index/cli.py`:

- `AGENT_STATE_FILE`: Path for saving agent state (default: "agent_state.json")
- `BROWSER_STATE_FILE`: Path for saving browser state (default: "browser_state.json")
- Browser configuration in the `AgentSession` class (headless mode, viewport, etc.) 
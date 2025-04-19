#!/usr/bin/env python
import asyncio
import json
import os
from typing import Dict, List, Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from textual.app import App
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Button, Footer, Header, Input, Static

from index.agent.agent import Agent
from index.agent.models import ActionResult, AgentOutput, AgentState
from index.browser.browser import BrowserConfig
from index.llm.llm import BaseLLMProvider
from index.llm.providers.anthropic import AnthropicProvider

# Create Typer app
app = typer.Typer(help="Index - Browser AI agent CLI")

# Configuration constants
AGENT_STATE_FILE = "agent_state.json"
BROWSER_STATE_FILE = "browser_state.json"

console = Console()

class AgentSession:
    """Manages an agent session with state persistence"""
    
    def __init__(self, llm: Optional[BaseLLMProvider] = None):
        self.llm = llm or AnthropicProvider(model="claude-3-7-sonnet-20250219", enable_thinking=True, thinking_token_budget=2048)
        self.browser_config = BrowserConfig(
            viewport={"width": 1280, "height": 720},
            headless=False  # Set to True in production
        )
        self.agent = Agent(llm=self.llm, browser_config=self.browser_config)
        self.agent_state: Optional[str] = None
        self.step_count: int = 0
        self.action_results: List[Dict] = []
        self.is_running: bool = False
        self.storage_state: Optional[Dict] = None
        
    def load_state(self) -> bool:
        """Load agent state from file if it exists"""
        if os.path.exists(AGENT_STATE_FILE):
            try:
                with open(AGENT_STATE_FILE, "r") as f:
                    self.agent_state = f.read()
                
                if os.path.exists(BROWSER_STATE_FILE):
                    with open(BROWSER_STATE_FILE, "r") as f:
                        self.storage_state = json.load(f)
                        
                console.print("[green]Loaded existing agent state[/green]")
                return True
            except Exception as e:
                console.print(f"[red]Error loading state: {e}[/red]")
        return False
    
    def save_state(self, agent_output: AgentOutput):
        """Save agent state to file"""
        with open(AGENT_STATE_FILE, "w") as f:
            f.write(agent_output.agent_state.model_dump_json())
        
        if agent_output.storage_state:
            with open(BROWSER_STATE_FILE, "w") as f:
                json.dump(agent_output.storage_state, f)
                
        console.print("[green]Saved agent state[/green]")
    
    async def run_agent(self, prompt: str) -> AgentOutput:
        """Run the agent with the given prompt"""
        self.is_running = True
        
        try:
            # Initialize browser with storage state if available
            if self.storage_state:
                await self.agent.browser.context.add_cookies(self.storage_state["cookies"])
                await self.agent.browser.context.add_storage_state(self.storage_state)
            
            # Run the agent
            if self.agent_state:
                result = await self.agent.run(
                    prompt=prompt, 
                    agent_state=self.agent_state, 
                    close_context=False
                )
            else:
                result = await self.agent.run(
                    prompt=prompt,
                    close_context=False
                )
            
            self.step_count = result.step_count
            self.agent_state = result.agent_state.model_dump_json()
            self.save_state(result)
            
            return result
        finally:
            self.is_running = False

    async def send_follow_up(self, message: str) -> AgentOutput:
        """Send a follow-up message to the agent"""
        if not self.agent_state:
            console.print("[red]No agent state available. Run the agent first.[/red]")
            raise ValueError("No agent state available")
        
        # Create a previous action result with give_control=True
        prev_action_result = ActionResult(
            give_control=True,
            content=message,
            is_done=False
        )
        
        result = await self.agent.run(
            prompt=message,
            agent_state=self.agent_state,
            prev_action_result=prev_action_result,
            close_context=False
        )
        
        self.step_count = result.step_count
        self.agent_state = result.agent_state.model_dump_json()
        self.save_state(result)
        
        return result

    async def stream_run(self, prompt: str):
        """Run the agent with streaming output"""
        self.is_running = True
        
        try:
            # Initialize browser with storage state if available
            if self.storage_state:
                await self.agent.browser.context.add_cookies(self.storage_state["cookies"])
                await self.agent.browser.context.add_storage_state(self.storage_state)
            
            # Run the agent with streaming
            if self.agent_state:
                stream = self.agent.run_stream(
                    prompt=prompt, 
                    agent_state=self.agent_state, 
                    close_context=False
                )
            else:
                stream = self.agent.run_stream(
                    prompt=prompt,
                    close_context=False
                )
            
            final_output = None
            async for chunk in stream:
                if chunk.type == "step":
                    yield {"type": "step", "content": chunk.content}
                elif chunk.type == "final_output":
                    final_output = chunk.content
                    yield {"type": "final", "content": final_output}
                elif chunk.type == "step_error":
                    yield {"type": "error", "content": chunk.content}
                elif chunk.type == "step_timeout":
                    yield {"type": "timeout", "content": chunk.content}
            
            if final_output:
                self.step_count = final_output.step_count
                self.agent_state = final_output.agent_state.model_dump_json()
                self.save_state(final_output)
        finally:
            self.is_running = False

    def reset(self):
        """Reset agent state"""
        if os.path.exists(AGENT_STATE_FILE):
            os.remove(AGENT_STATE_FILE)
        if os.path.exists(BROWSER_STATE_FILE):
            os.remove(BROWSER_STATE_FILE)
        self.agent_state = None
        self.step_count = 0
        self.action_results = []
        console.print("[yellow]Agent state reset[/yellow]")


class AgentUI(App):
    """Textual-based UI for interacting with the agent"""
    
    CSS = """
    Header {
        background: #3b82f6;
        color: white;
        text-align: center;
        padding: 1;
    }
    
    Footer {
        background: #1e3a8a;
        color: white;
        text-align: center;
        padding: 1;
    }
    
    #prompt-input {
        padding: 1 2;
        border: tall $accent;
        margin: 1 1;
        height: 3;
    }
    
    #output-container {
        height: 1fr;
        border: solid #ccc;
        background: #f8fafc;
        padding: 1;
        margin: 0 1;
        overflow-y: auto;
    }
    
    #action-results {
        height: 15;
        border: solid #ccc;
        background: #f8fafc;
        margin: 0 1 1 1;
        overflow-y: auto;
    }
    
    .action-result {
        border: solid #e5e7eb;
        margin: 1 0;
        padding: 1;
    }
    
    .action-title {
        color: #3b82f6;
        text-style: bold;
    }
    
    .action-content {
        margin-top: 1;
    }
    
    Button {
        margin: 1 1;
    }
    
    #buttons-container {
        height: auto;
        align: center middle;
    }
    
    .running {
        color: #f97316;
        text-style: bold;
    }
    
    .completed {
        color: #22c55e;
        text-style: bold;
    }
    
    .error {
        color: #ef4444;
        text-style: bold;
    }
    """
    
    TITLE = "Index Browser Agent CLI"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "reset", "Reset Agent"),
        ("ctrl+s", "send", "Send Message"),
    ]
    
    agent_session = AgentSession()
    status = reactive("Ready")
    
    def compose(self):
        yield Header()
        
        with Vertical():
            with Container(id="output-container"):
                yield Static(id="output", expand=True)
                
            with Container(id="action-results"):
                yield Static(id="results", expand=True)
                
            with Horizontal(id="buttons-container"):
                yield Button("Send", id="send-btn", variant="primary")
                yield Button("Reset", id="reset-btn", variant="error")
                
            yield Input(placeholder="Enter your task or follow-up message...", id="prompt-input")
                
        yield Footer()
    
    def on_mount(self):
        """When the app is mounted, try to load previous state"""
        self.agent_session.load_state()
        self.update_output()
        
    def update_output(self):
        """Update the output display"""
        output = ""
        
        if self.agent_session.agent_state:
            state = AgentState.model_validate_json(self.agent_session.agent_state)
            
            # Get the latest user and assistant messages
            user_msgs = [m for m in state.messages if m.role == "user"]
            assistant_msgs = [m for m in state.messages if m.role == "assistant"]
            
            if user_msgs:
                latest_user = user_msgs[-1]
                output += f"[bold blue]User:[/] {latest_user.content}\n\n"
                
            if assistant_msgs:
                latest_assistant = assistant_msgs[-1]
                output += f"[bold green]Assistant:[/] {latest_assistant.content}\n\n"
                
            output += f"[dim]Steps completed: {self.agent_session.step_count}[/]\n"
            output += f"[dim]Status: {self.status}[/]\n"
        else:
            output = "[italic]No previous session. Start by sending a task.[/]"
            
        self.query_one("#output", Static).update(Markdown(output))
        
        # Update action results
        if self.agent_session.action_results:
            results_output = ""
            for i, result in enumerate(reversed(self.agent_session.action_results[-5:])):
                action_type = result.get("type", "unknown")
                content = result.get("content", {})
                
                if action_type == "step":
                    action_result = content.get("action_result", {})
                    summary = content.get("summary", "No summary available")
                    
                    results_output += f"[bold]Step {i+1}[/]\n"
                    results_output += f"Summary: {summary}\n"
                    
                    if action_result.get("is_done"):
                        results_output += "[green]Task completed[/]\n"
                    
                    if action_result.get("give_control"):
                        results_output += "[yellow]Agent requested human control[/]\n"
                        results_output += f"Message: {action_result.get('content', '')}\n"
                    
                    results_output += "\n"
                    
                elif action_type == "error":
                    results_output += "[bold red]Error[/]\n"
                    results_output += f"{content}\n\n"
                    
            self.query_one("#results", Static).update(Markdown(results_output))
    
    async def on_button_pressed(self, event: Button.Pressed):
        """Handle button presses"""
        if event.button.id == "send-btn":
            await self.action_send()
        elif event.button.id == "reset-btn":
            self.action_reset()
    
    def action_reset(self):
        """Reset the agent state"""
        self.agent_session.reset()
        self.agent_session.action_results = []
        self.update_output()
    
    async def action_send(self):
        """Send the current prompt to the agent"""
        prompt = self.query_one("#prompt-input", Input).value
        
        if not prompt.strip():
            return
            
        self.status = "Running..."
        self.query_one("#prompt-input", Input).value = ""
        self.update_output()
        
        try:
            # Stream the results to provide real-time feedback
            async for chunk in self.agent_session.stream_run(prompt):
                self.agent_session.action_results.append(chunk)
                self.update_output()
                await asyncio.sleep(0.1)  # Small delay to ensure UI updates
                
            self.status = "Ready"
        except Exception as e:
            self.status = f"Error: {str(e)}"
        finally:
            self.update_output()
    
    def action_quit(self):
        """Quit the application"""
        self.exit()


@app.command()
def run(prompt: str = typer.Option(None, "--prompt", "-p", help="Initial prompt to send to the agent")):
    """
    Launch the interactive CLI for the Index browser agent
    """
    agent_ui = AgentUI()
    
    if prompt:
        # If a prompt is provided, we'll send it once the UI is ready
        async def send_initial_prompt():
            await asyncio.sleep(0.5)  # Give UI time to initialize
            agent_ui.query_one("#prompt-input", Input).value = prompt
            await agent_ui.action_send()
        
        agent_ui.set_interval(0.1, lambda: asyncio.create_task(send_initial_prompt()))
    
    agent_ui.run()


@app.command()
def interactive():
    """
    Run in interactive loop mode with sequential message input and agent response
    """
    asyncio.run(_interactive_loop())


async def _interactive_loop():
    """Implementation of the interactive loop mode"""
    session = AgentSession()
    session.load_state()
    
    console.print(Panel.fit(
        "Index Browser Agent Interactive Mode\n"
        "Type your message and press Enter. The agent will respond.\n"
        "Press Ctrl+C to exit.",
        title="Interactive Mode",
        border_style="blue"
    ))
    
    try:
        while True:
            # Get user input
            console.print("\n[bold blue]Your message:[/] ", end="")
            user_message = input()
            
            if not user_message.strip():
                continue
            
            console.print("\n[bold cyan]Agent is working...[/]")
            
            # Create a live display for updating with chunks
            with Live(Text("Starting agent..."), refresh_per_second=4) as live:
                step_num = 1
                human_control_requested = False
                last_content = None
                
                # Run the agent with streaming output
                try:
                    async for chunk in session.stream_run(user_message):
                        chunk_type = chunk.get("type", "unknown")
                        content = chunk.get("content", {})
                        
                        if chunk_type == "step":
                            action_result = content.get("action_result", {})
                            summary = content.get("summary", "No summary available")
                            
                            output = Text()
                            output.append(f"Step {step_num}: ", style="bold")
                            output.append(f"{summary}\n")
                            
                            if action_result.get("is_done"):
                                output.append("Task completed successfully!\n", style="green bold")
                            
                            if action_result.get("give_control"):
                                human_control_requested = True
                                message = action_result.get("content", "No message provided")
                                output.append("Agent requested human control:\n", style="yellow bold")
                                output.append(f"{message}\n")
                            
                            live.update(output)
                            step_num += 1
                            last_content = output
                            
                        elif chunk_type == "error":
                            error_msg = Text()
                            error_msg.append("Error: ", style="red bold")
                            error_msg.append(f"{content}\n")
                            live.update(error_msg)
                            last_content = error_msg
                
                except Exception as e:
                    error_msg = Text()
                    error_msg.append("Error: ", style="red bold")
                    error_msg.append(f"{str(e)}\n")
                    live.update(error_msg)
            
            # After agent completes
            if human_control_requested:
                console.print("[yellow]Agent has requested human control.[/]")
            else:
                console.print("[green]Agent has completed the task.[/]")
            
            console.print("[dim]Waiting for your next message...[/]")
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting interactive mode...[/]")
        # Close the browser before exiting
        await session.agent.browser.close()


@app.command()
def reset():
    """
    Reset the agent state
    """
    session = AgentSession()
    session.reset()


def main():
    """Entry point for the CLI"""
    app()


if __name__ == "__main__":
    main()
#!/usr/bin/env python
import asyncio
import json
import logging
import os
import subprocess
import time
from typing import Dict, List, Optional

import requests
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from textual.app import App
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Button, Footer, Header, Input, Static

from index.agent.agent import Agent
from index.agent.models import AgentOutput, AgentState
from index.browser.browser import BrowserConfig
from index.llm.llm import BaseLLMProvider
from index.llm.providers.anthropic import AnthropicProvider
from index.llm.providers.gemini import GeminiProvider
from index.llm.providers.openai import OpenAIProvider

load_dotenv()

# Create Typer app
app = typer.Typer(help="Index - Browser AI agent CLI")

# Configuration constants
BROWSER_STATE_FILE = "browser_state.json"
DEFAULT_CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
DEFAULT_DEBUGGING_PORT = 9222

console = Console()

def setup_logging(debug: bool = False):
    """Configure logging based on debug flag"""
    log_level = logging.INFO if debug else logging.WARNING
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)]
    )
    
    # Set specific logger levels
    logging.getLogger("index").setLevel(log_level)
    logging.getLogger("playwright").setLevel(logging.WARNING)  # Always keep playwright at WARNING
    
    if debug:
        console.print("[yellow]Debug mode enabled - logging set to INFO level[/]")

class AgentSession:
    """Manages an agent session with state persistence"""
    
    def __init__(self, llm: Optional[BaseLLMProvider] = None, use_local_chrome: bool = False, chrome_path: str = DEFAULT_CHROME_PATH, debugging_port: int = DEFAULT_DEBUGGING_PORT, debug: bool = False):
        self.llm = llm
        self.chrome_process = None
        self.use_local_chrome = use_local_chrome
        self.chrome_path = chrome_path
        self.debugging_port = debugging_port
        self.logger = logging.getLogger("index.agent_session")
        
        browser_config = None

        if os.path.exists(BROWSER_STATE_FILE) and not use_local_chrome:
            with open(BROWSER_STATE_FILE, "r") as f:
                self.storage_state = json.load(f)
                console.print("[green]Loaded existing browser state[/green]")
                browser_config = BrowserConfig(
                    storage_state=self.storage_state,
                    viewport_size={
                        "width": 1200,
                        "height": 800
                    }
                )
        else:
            if use_local_chrome:
                # Launch Chrome and connect to it
                self._launch_local_chrome()
                browser_config = BrowserConfig(
                    cdp_url="http://localhost:" + str(self.debugging_port),
                )
            else:
                browser_config = BrowserConfig(
                    viewport_size={
                        "width": 1200,
                        "height": 800
                    }
                )

        self.agent = Agent(llm=self.llm, browser_config=browser_config)
        self.agent_state: Optional[str] = None
        self.step_count: int = 0
        self.action_results: List[Dict] = []
        self.is_running: bool = False
        self.storage_state: Optional[Dict] = None
    
    def _launch_local_chrome(self):
        """Launch a local Chrome instance with remote debugging enabled"""
        # Check if Chrome is already running with the specified debugging port
        try:
            response = requests.get(f"http://localhost:{self.debugging_port}/json/version", timeout=2)
            if response.status_code == 200:
                console.print(f"[green]Connected to already running Chrome instance on port {self.debugging_port}[/green]")
                self.logger.info(f"Connected to existing Chrome instance on port {self.debugging_port}")
                return
        except requests.RequestException:
            # No running Chrome instance found on the specified port, proceed with launching a new one
            pass
            
        console.print(f"[blue]Launching Chrome from {self.chrome_path} with debugging port {self.debugging_port}[/blue]")
        
        try:
            self.chrome_process = subprocess.Popen(
                [self.chrome_path, f"--remote-debugging-port={self.debugging_port}", "--no-first-run", "--no-default-browser-check"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            console.print("[green]Chrome launched successfully[/green]")
            self.logger.info(f"Chrome process started with PID {self.chrome_process.pid}")
            # Give Chrome time to start up
            time.sleep(2)
        except Exception as e:
            self.logger.error(f"Failed to launch Chrome: {str(e)}")
            console.print(f"[red]Failed to launch Chrome: {str(e)}[/red]")
            raise
        
    def save_state(self, agent_output: AgentOutput):
        """Save agent state to file"""
        
        if agent_output.storage_state:
            with open(BROWSER_STATE_FILE, "w") as f:
                json.dump(agent_output.storage_state, f)
                
            self.logger.info("Agent state saved to file")
            console.print("[green]Saved agent state[/green]")
    
    async def run_agent(self, prompt: str) -> AgentOutput:
        """Run the agent with the given prompt"""
        self.is_running = True
        self.logger.info(f"Running agent with prompt: {prompt}")
        
        try:
            # Run the agent
            if self.agent_state:
                result = await self.agent.run(
                    prompt=prompt, 
                    agent_state=self.agent_state, 
                    close_context=False,
                    return_storage_state=True,
                    return_agent_state=True
                )
            else:
                result = await self.agent.run(
                    prompt=prompt,
                    close_context=False,
                    return_storage_state=True,
                    return_agent_state=True
                )
            
            self.step_count = result.step_count
            self.agent_state = result.agent_state.model_dump_json()
            self.save_state(result)
            
            return result
        finally:
            self.is_running = False

    async def stream_run(self, prompt: str):
        """Run the agent with streaming output"""
        self.is_running = True
        self.logger.info(f"Running agent with streaming and prompt: {prompt}")
        
        try:
            # Run the agent with streaming
            if self.agent_state:
                stream = self.agent.run_stream(
                    prompt=prompt, 
                    agent_state=self.agent_state, 
                    close_context=False,
                    max_steps=500, # large number to allow the agent to run for a long time
                    return_agent_state=True,
                    return_storage_state=True
                )
            else:
                stream = self.agent.run_stream(
                    prompt=prompt,
                    close_context=False,
                    max_steps=500, # large number to allow the agent to run for a long time
                    return_agent_state=True,
                    return_storage_state=True
                )
            
            final_output = None
            async for chunk in stream:
                # Directly yield the raw chunk without any modifications
                yield chunk
                
                # Store final output for state saving
                if chunk.type == "final_output":
                    final_output = chunk.content
            
            if final_output:
                self.step_count = final_output.step_count
                self.agent_state = final_output.agent_state.model_dump_json()
                self.save_state(final_output)
                
        finally:
            self.is_running = False

    def reset(self):
        """Reset agent state"""
        if os.path.exists(BROWSER_STATE_FILE):
            os.remove(BROWSER_STATE_FILE)
        self.agent_state = None
        self.step_count = 0
        self.action_results = []
        self.logger.info("Agent state reset")
        console.print("[yellow]Agent state reset[/yellow]")
    
    async def close(self):
        """Close the agent and any associated resources"""
        # Close the browser instance
        if self.agent and self.agent.browser:
            self.logger.info("Closing browser instance")
            await self.agent.browser.close()
        
        # Terminate Chrome process if launched locally
        if self.chrome_process:
            self.logger.info(f"Terminating Chrome process with PID {self.chrome_process.pid}")
            console.print("[yellow]Terminating local Chrome instance...[/yellow]")
            self.chrome_process.terminate()
            self.chrome_process = None


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
    
    agent_session = None
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
    
    async def on_mount(self):
        """Called when the app is mounted"""
        # Register cleanup handler
        self.set_interval(0.1, self._check_exit)
    
    async def _check_exit(self):
        """Check if app is exiting and clean up resources"""
        if self.exiting:
            if self.agent_session:
                await self.agent_session.close()
    
    def action_quit(self):
        """Quit the application"""
        self.exit()


@app.command()
def run(
    prompt: str = typer.Option(None, "--prompt", "-p", help="Initial prompt to send to the agent"),
    use_local_chrome: bool = typer.Option(False, "--local-chrome", help="Use local Chrome instance instead of launching a new browser"),
    chrome_path: str = typer.Option(DEFAULT_CHROME_PATH, "--chrome-path", help="Path to Chrome executable"),
    debugging_port: int = typer.Option(DEFAULT_DEBUGGING_PORT, "--port", help="Remote debugging port for Chrome"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging")
):
    """
    Launch the interactive loop for the Index browser agent
    """
    # Set up logging if debug mode is enabled
    setup_logging(debug)
    
    asyncio.run(_interactive_loop(
        initial_prompt=prompt, 
        use_local_chrome=use_local_chrome, 
        chrome_path=chrome_path, 
        debugging_port=debugging_port,
        debug=debug
    ))


@app.command(name="ui")
def run_ui(
    prompt: str = typer.Option(None, "--prompt", "-p", help="Initial prompt to send to the agent"),
    use_local_chrome: bool = typer.Option(False, "--local-chrome", help="Use local Chrome instance instead of launching a new browser"),
    chrome_path: str = typer.Option(DEFAULT_CHROME_PATH, "--chrome-path", help="Path to Chrome executable"),
    debugging_port: int = typer.Option(DEFAULT_DEBUGGING_PORT, "--port", help="Remote debugging port for Chrome"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging")
):
    """
    Launch the graphical UI for the Index browser agent
    """
    # Set up logging if debug mode is enabled
    setup_logging(debug)
    
    # Select model and check API key
    llm_provider = select_model_and_check_key()
    
    # Initialize UI with the selected LLM provider
    agent_ui = AgentUI()
    agent_ui.agent_session = AgentSession(
        llm=llm_provider,
        use_local_chrome=use_local_chrome,
        chrome_path=chrome_path,
        debugging_port=debugging_port,
        debug=debug
    )
    
    if prompt:
        # If a prompt is provided, we'll send it once the UI is ready
        async def send_initial_prompt():
            await asyncio.sleep(0.5)  # Give UI time to initialize
            agent_ui.query_one("#prompt-input", Input).value = prompt
            await agent_ui.action_send()
        
        agent_ui.set_interval(0.1, lambda: asyncio.create_task(send_initial_prompt()))
    
    agent_ui.run()


def create_llm_provider(provider: str, model: str) -> BaseLLMProvider:
    """Create an LLM provider based on model choice"""
    if provider == "openai":
        # OpenAI model
        console.print(f"[cyan]Using OpenAI model: {model}[/]")
        return OpenAIProvider(model=model, reasoning_effort="low")
    elif provider == "gemini":
        # Gemini model
        if model == "gemini-2.5-pro-preview-03-25":
            console.print(f"[cyan]Using Gemini model: {model}[/]")
            return GeminiProvider(
                model=model,
                thinking_token_budget=8192
            )
        elif model == "gemini-2.5-flash-preview-04-17":
            console.print(f"[cyan]Using Gemini model: {model}[/]")
            return GeminiProvider(
                model=model,
                thinking_token_budget=8192
            )
        else:
            raise ValueError(f"Unsupported Gemini model: {model}")
    elif provider == "anthropic":
        # Anthropic model
        console.print(f"[cyan]Using Anthropic model: {model}[/]")
        return AnthropicProvider(
            model=model,
            enable_thinking=True,
            thinking_token_budget=2048
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def check_and_save_api_key(required_key: str):
    """Check if API key exists, prompt for it if missing, and save to .env file"""
    if not os.environ.get(required_key):
        console.print(f"\n[yellow]API key {required_key} not found in environment.[/]")
        api_key = Prompt.ask(f"Enter your {required_key}", password=True)
        
        # Save to .env file
        env_path = ".env"
        
        if os.path.exists(env_path):
            # Read existing content
            with open(env_path, "r") as f:
                env_content = f.read()
            env_content += f"\n{required_key}={api_key}"
            
            with open(env_path, "w") as f:
                f.write(env_content)
            console.print(f"[green]Saved {required_key} to .env file[/]")
        else:
            # Create new .env file
            with open(env_path, "w") as f:
                f.write(f"{required_key}={api_key}")
            console.print("[green]Created .env file[/]")

        # Update environment variable for current session
        os.environ[required_key] = api_key
        
        # Reload dotenv to ensure changes are applied
        load_dotenv(override=True)
        

def select_model_and_check_key():
    """Select a model and check for required API key"""
    console.print("\n[bold green]Choose an LLM model:[/]")
    console.print("1. [bold]Gemini 2.5 Pro[/]")
    console.print("2. [bold]Gemini 2.5 Flash[/]")
    console.print("3. [bold]Claude 3.7 Sonnet[/]")
    console.print("4. [bold]OpenAI o4-mini[/]")
    
    choice = Prompt.ask(
        "[bold]Select model[/]",
        choices=["1", "2", "3", "4"],
        default="1"
    )
    
    provider = ""
    model = ""
    required_key = ""
    
    # Create LLM provider based on selection
    if choice == "1":
        provider = "gemini"
        model = "gemini-2.5-pro-preview-03-25"
        required_key = "GEMINI_API_KEY"
    elif choice == "2":
        provider = "gemini"
        model = "gemini-2.5-flash-preview-04-17"
        required_key = "GEMINI_API_KEY"
    elif choice == "3":
        provider = "anthropic"
        model = "claude-3-7-sonnet-20250219"
        required_key = "ANTHROPIC_API_KEY"
    elif choice == "4":
        provider = "openai"
        model = "o4-mini"
        required_key = "OPENAI_API_KEY"
    else:
        raise ValueError(f"Invalid choice: {choice}")
    
    # Check and save API key if needed
    check_and_save_api_key(required_key)
    
    return create_llm_provider(provider, model)


async def _interactive_loop(initial_prompt: str = None, use_local_chrome: bool = False, chrome_path: str = DEFAULT_CHROME_PATH, debugging_port: int = DEFAULT_DEBUGGING_PORT, debug: bool = False):
    """Implementation of the interactive loop mode"""
    # Display welcome panel
    console.print(Panel.fit(
        "Index Browser Agent Interactive Mode\n"
        "Type your message and press Enter. The agent will respond.\n"
        "Press Ctrl+C to exit.",
        title="Interactive Mode",
        border_style="blue"
    ))
    
    # Select model and check API key
    llm_provider = select_model_and_check_key()
    
    # Create agent session with selected provider
    session = AgentSession(
        llm=llm_provider,
        use_local_chrome=use_local_chrome,
        chrome_path=chrome_path,
        debugging_port=debugging_port,
        debug=debug
    )
    
    try:
        first_message = True
        awaiting_human_input = False
        
        while True:
            # Check if we're waiting for the user to return control to the agent
            if awaiting_human_input:
                console.print("\n[yellow]Agent is waiting for control to be returned.[/]")
                console.print("[yellow]Press Enter to return control to the agent...[/]", end="")
                input()  # Wait for Enter key
                user_message = "Returning control back, continue your task"
                console.print(f"\n[bold blue]Your message:[/] {user_message}")
                awaiting_human_input = False
            # Normal message input flow
            elif first_message and initial_prompt:
                user_message = initial_prompt
                console.print(f"\n[bold blue]Your message:[/] {user_message}")
                first_message = False
            else:
                console.print("\n[bold blue]Your message:[/] ", end="")
                user_message = input()
                first_message = False
            
            if not user_message.strip():
                continue
            
            console.print("\n[bold cyan]Agent is working...[/]")
            
            step_num = 1
            human_control_requested = False
            
            # Run the agent with streaming output
            try:
                async for chunk in session.stream_run(user_message):
                    if chunk.type == "step":
                        action_result = chunk.content.action_result
                        summary = chunk.content.summary
                        
                        # Simple single-line output for steps
                        console.print(f"[bold blue]Step {step_num}:[/] {summary}")
                        # Display additional info for special actions as separate lines
                        if action_result and action_result.is_done and not action_result.give_control:
                            console.print("  [green bold]✓ Task completed successfully![/]")
                        
                        if action_result and action_result.give_control:
                            human_control_requested = True
                            message = action_result.content or "No message provided"
                            console.print("  [yellow bold]⚠ Human control requested:[/]")
                            console.print(f"  [yellow]{message}[/]")
                        
                        # Increment step counter for next step
                        step_num += 1
                        
                    elif chunk.type == "step_error":
                        console.print(f"[bold red]Error:[/] {chunk.content}")
                        
                    elif chunk.type == "final_output":
                        # Keep panel for final output
                        result_content = chunk.content.result.content if chunk.content.result else "No result content"
                        console.print(Panel(
                            f"{result_content}",
                            title="Final Output",
                            border_style="green",
                            expand=False
                        ))
                
            except Exception as e:
                console.print(f"[bold red]Error:[/] {str(e)}")
                console.print(f"[dim]Type: {type(e)}[/]")
                console.print_exception()
            
            # After agent completes
            if human_control_requested:
                console.print("\n[yellow]Agent has requested human control.[/]")
                awaiting_human_input = True
            else:
                console.print("\n[green]Agent has completed the task.[/]")
                console.print("[dim]Waiting for your next message...[/]")
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting interactive mode...[/]")
        # Close the browser before exiting
        await session.close()


def main():
    """Entry point for the CLI"""
    app()


if __name__ == "__main__":
    main()
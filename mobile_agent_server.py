#!/usr/bin/env python3
"""
Mobile Agent MCP Server - Provides MCP tools for phone automation.

This server exposes three tools:
1. mobile_agent_status - Check current task execution status
2. mobile_agent_call - Execute mobile automation tasks
"""

import sys
import logging
import yaml
import threading
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

# Fix UTF-8 encoding for Windows console
if sys.platform == "win32":
    sys.stderr.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")

from fastmcp import FastMCP
import math
import random

# CRITICAL: Save original stdout for MCP protocol, then redirect print() to stderr
# This must be done BEFORE importing phone_agent to intercept all print() calls
_original_stdout = sys.stdout
_original_stderr = sys.stderr

class PrintToStderr:
    """Intercept print() calls and redirect to stderr, but preserve original stdout for MCP"""
    def __init__(self, original_stdout):
        self._original_stdout = original_stdout
        self._in_mcp_write = False

    def write(self, text):
        # Check if this is a JSON-RPC message (starts with '{' or is part of MCP protocol)
        # These should go to original stdout
        stripped = text.strip()
        if stripped.startswith('{') or stripped.startswith('Content-Length:'):
            return self._original_stdout.write(text)
        else:
            # All other output (print statements) go to stderr
            return _original_stderr.write(text)

    def flush(self):
        self._original_stdout.flush()
        _original_stderr.flush()

    def isatty(self):
        return self._original_stdout.isatty()

    @property
    def buffer(self):
        """Expose the underlying buffer for fastmcp"""
        return self._original_stdout.buffer

    @property
    def encoding(self):
        """Expose encoding for compatibility"""
        return getattr(self._original_stdout, 'encoding', 'utf-8')

    @property
    def errors(self):
        """Expose error handling mode"""
        return getattr(self._original_stdout, 'errors', 'strict')

# Replace stdout with smart redirector
sys.stdout = PrintToStderr(_original_stdout)

from phone_agent import PhoneAgent
from phone_agent.agent import AgentConfig
from phone_agent.model import ModelConfig

# Configure logging to output to stderr (not stdout, which is piped to WebSocket)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=_original_stderr,  # Output to stderr so logs appear in console
    force=True,  # Override any existing config
)
logger = logging.getLogger("MobileAgentMCP")


# Global state for task management
class TaskState:
    def __init__(self):
        self.lock = threading.Lock()
        self.current_task: Optional[str] = None
        self.status: str = "idle"  # idle, running, completed, error
        self.result: Optional[str] = None
        self.error: Optional[str] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.agent: Optional[PhoneAgent] = None
        self.current_step: int = 0
        self.max_steps: int = 0
        self.last_action: Optional[str] = None
        self.task_thread: Optional[threading.Thread] = None

    def start_task(self, task: str, max_steps: int = 100):
        with self.lock:
            self.current_task = task
            self.status = "running"
            self.result = None
            self.error = None
            self.start_time = datetime.now()
            self.end_time = None
            self.current_step = 0
            self.max_steps = max_steps
            self.last_action = None

    def update_progress(self, step: int, action: str):
        with self.lock:
            self.current_step = step
            self.last_action = action

    def complete_task(self, result: str):
        with self.lock:
            self.status = "completed"
            self.result = result
            self.end_time = datetime.now()
            self.task_thread = None

    def error_task(self, error: str):
        with self.lock:
            self.status = "error"
            self.error = error
            self.end_time = datetime.now()
            self.task_thread = None

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            elapsed = (
                (datetime.now() - self.start_time).total_seconds()
                if self.start_time and self.status == "running"
                else (
                    (self.end_time - self.start_time).total_seconds()
                    if self.start_time and self.end_time
                    else None
                )
            )
            return {
                "status": self.status,
                "current_task": self.current_task,
                "result": self.result,
                "error": self.error,
                "progress": (
                    {
                        "current_step": self.current_step,
                        "max_steps": self.max_steps,
                        "last_action": self.last_action,
                        "percentage": (
                            int(self.current_step / self.max_steps * 100)
                            if self.max_steps > 0
                            else 0
                        ),
                    }
                    if self.status == "running"
                    else None
                ),
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "elapsed_seconds": elapsed,
            }

    def is_idle(self) -> bool:
        with self.lock:
            return self.status in ("idle", "completed", "error")


# Global task state
task_state = TaskState()

# Create an MCP server
mcp = FastMCP("MobileAgent")


def load_config() -> Dict[str, Any]:
    """Load configuration from .config.yaml or config.yaml"""
    config_path = Path(".config.yaml")
    template_path = Path("config.yaml")

    # Check if .config.yaml exists
    if not config_path.exists():
        if template_path.exists():
            logger.error("Configuration file .config.yaml not found!")
            logger.error(
                f"Please copy {template_path} to {config_path} and fill in your settings."
            )
            logger.error("")
            logger.error("Required settings:")
            logger.error("  1. mcp_endpoint - Your MCP WebSocket endpoint")
            logger.error("  2. api_key - Your vision model API key (in VLLM section)")
            sys.exit(1)
        else:
            logger.error("Neither .config.yaml nor config.yaml found!")
            sys.exit(1)

    # Load .config.yaml
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load {config_path}: {e}")
        sys.exit(1)

    # Validate required fields
    mcp_endpoint = config.get("mcp_endpoint", "")
    if not mcp_endpoint or mcp_endpoint == "你的MCP_ENDPOINT":
        logger.error("mcp_endpoint is not configured in .config.yaml!")
        logger.error("Please set your MCP WebSocket endpoint in .config.yaml")
        sys.exit(1)

    # Get vision model config
    selected_vllm = config.get("selected_module", {}).get("VLLM")
    if not selected_vllm:
        logger.error("No VLLM model selected in .config.yaml!")
        sys.exit(1)

    vllm_config = config.get("VLLM", {}).get(selected_vllm, {})
    if not vllm_config:
        logger.error(f"VLLM config for {selected_vllm} not found in .config.yaml!")
        sys.exit(1)

    api_key = vllm_config.get("api_key", "")
    if not api_key or api_key == "你的api_key":
        logger.error(f"api_key for {selected_vllm} is not configured in .config.yaml!")
        logger.error("Please set your vision model API key in .config.yaml")
        sys.exit(1)

    return {
        "mcp_endpoint": mcp_endpoint,
        "model_config": {
            "type": vllm_config.get("type", "openai"),
            "model_name": vllm_config.get("model_name", "autoglm-phone-9b"),
            "base_url": vllm_config.get("url", "http://localhost:8000/v1"),
            "api_key": api_key,
        },
    }


def initialize_agent(model_config_dict: Dict[str, Any]) -> PhoneAgent:
    """Initialize the PhoneAgent with model configuration"""
    model_config = ModelConfig(
        base_url=model_config_dict["base_url"],
        model_name=model_config_dict["model_name"],
        api_key=model_config_dict["api_key"],
    )

    agent_config = AgentConfig(
        max_steps=100, device_id=None, lang="cn", verbose=False  # Disable verbose to prevent stdout pollution
    )

    return PhoneAgent(model_config=model_config, agent_config=agent_config)


def run_task_with_progress(task: str):
    """Run task in background thread with progress updates"""
    try:
        agent = task_state.agent
        agent.reset()

        # Execute first step
        result = agent.step(task)
        task_state.update_progress(
            agent.step_count,
            result.action.get("action", "unknown") if result.action else "unknown",
        )

        if result.finished:
            task_state.complete_task(result.message or "Task completed")
            return

        # Continue until finished or max steps
        while agent.step_count < agent.agent_config.max_steps:
            result = agent.step()
            task_state.update_progress(
                agent.step_count,
                result.action.get("action", "unknown") if result.action else "unknown",
            )

            if result.finished:
                task_state.complete_task(result.message or "Task completed")
                return

        task_state.complete_task("Max steps reached")

    except Exception as e:
        import traceback

        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(f"Task execution error: {error_msg}")
        task_state.error_task(str(e))


# Load config on startup
try:
    config = load_config()
    logger.info("Configuration loaded successfully")
    logger.info(f"MCP Endpoint: {config['mcp_endpoint']}")
    logger.info(f"Model: {config['model_config']['model_name']}")

    # Initialize agent
    task_state.agent = initialize_agent(config["model_config"])
    logger.info("PhoneAgent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize: {e}")
    sys.exit(1)


# Tool 1: Mobile Agent Status
@mcp.tool()
def mobile_agent_status() -> dict:
    """Check the current status of the mobile agent. Use this before calling mobile_agent_call to see if another task is running."""
    status = task_state.get_status()
    logger.info(f"Status check: {status['status']}")
    return {"success": True, "status": status}


# Tool 2: Mobile Agent Call
@mcp.tool()
def mobile_agent_call(task: str) -> dict:
    """Execute a mobile automation task asynchronously.

    Args:
        task: Natural language description of the task to execute (e.g., "打开淘宝搜索小智虾哥，然后点击第一个商品加入购物车")

    Returns:
        Dictionary containing:
        - success: Boolean indicating if the task was accepted
        - message: Status message
        - task_id: Task identifier (the task description)
        - warning: Warning if another task is running

    Note:
        This function starts the task in the background and returns immediately.
        Use mobile_agent_status to check progress and results.
    """
    # Check if another task is running
    if not task_state.is_idle():
        logger.warning(f"Task rejected: Another task is already running")
        return {
            "success": False,
            "warning": "Another task is currently running. Please check mobile_agent_status and wait for it to complete.",
            "current_status": task_state.get_status(),
        }

    logger.info(f"Accepting task: {task}")

    # Start task in background thread
    task_state.start_task(task, max_steps=100)
    task_thread = threading.Thread(
        target=run_task_with_progress, args=(task,), daemon=True
    )
    task_state.task_thread = task_thread
    task_thread.start()

    logger.info(
        f"Task started in background. Use mobile_agent_status to check progress."
    )
    return {
        "success": True,
        "message": "Task started in background. Use mobile_agent_status to check progress.",
        "task_id": task,
        "status": task_state.get_status(),
    }


# Start the server
if __name__ == "__main__":
    logger.info("Starting Mobile Agent MCP Server...")
    mcp.run(transport="stdio")

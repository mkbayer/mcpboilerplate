"""
Centralized logging configuration for the trend radar application.
"""

import logging
import sys
from typing import Optional
from rich.logging import RichHandler
from rich.console import Console


def get_logger(
    name: str,
    level: str = "INFO",
    use_rich: bool = True
) -> logging.Logger:
    """
    Get a configured logger instance
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        use_rich: Whether to use rich formatting
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    
    if use_rich:
        # Use Rich handler for beautiful console output
        handler = RichHandler(
            console=Console(stderr=True),
            show_time=True,
            show_path=True,
            markup=True,
            rich_tracebacks=True
        )
        formatter = logging.Formatter(
            fmt="%(message)s",
            datefmt="[%X]"
        )
    else:
        # Standard handler
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


def configure_root_logger(level: str = "INFO", use_rich: bool = True) -> None:
    """
    Configure the root logger for the entire application
    
    Args:
        level: Logging level
        use_rich: Whether to use rich formatting
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    if use_rich:
        handler = RichHandler(
            console=Console(stderr=True),
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True
        )
        formatter = logging.Formatter(
            fmt="%(message)s",
            datefmt="[%X]"
        )
    else:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


class AgentLogger:
    """Specialized logger for MCP agents with context tracking"""
    
    def __init__(self, agent_id: str, base_logger: Optional[logging.Logger] = None):
        self.agent_id = agent_id
        self.logger = base_logger or get_logger(f"agent.{agent_id}")
        self.context = {}
    
    def set_context(self, **kwargs):
        """Set context variables for this agent"""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear all context variables"""
        self.context.clear()
    
    def _format_message(self, message: str) -> str:
        """Format message with agent context"""
        context_str = ""
        if self.context:
            context_parts = [f"{k}={v}" for k, v in self.context.items()]
            context_str = f" [{', '.join(context_parts)}]"
        
        return f"[bold blue]{self.agent_id}[/bold blue]{context_str}: {message}"
    
    def debug(self, message: str, **kwargs):
        """Log debug message with agent context"""
        self.set_context(**kwargs)
        self.logger.debug(self._format_message(message))
    
    def info(self, message: str, **kwargs):
        """Log info message with agent context"""
        self.set_context(**kwargs)
        self.logger.info(self._format_message(message))
    
    def warning(self, message: str, **kwargs):
        """Log warning message with agent context"""
        self.set_context(**kwargs)
        self.logger.warning(self._format_message(message))
    
    def error(self, message: str, **kwargs):
        """Log error message with agent context"""
        self.set_context(**kwargs)
        self.logger.error(self._format_message(message))
    
    def critical(self, message: str, **kwargs):
        """Log critical message with agent context"""
        self.set_context(**kwargs)
        self.logger.critical(self._format_message(message))
        
"""
Model Context Protocol message structures and types.
"""

from datetime import datetime
from typing import Any, Dict
from dataclasses import dataclass
from enum import Enum


class MCPMessageType(Enum):
    """Types of MCP messages"""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    ERROR_REPORT = "error_report"
    CONTEXT_SHARE = "context_share"
    COORDINATION = "coordination"


@dataclass
class MCPMessage:
    """Model Context Protocol Message Structure"""
    agent_id: str
    message_type: MCPMessageType
    content: Dict[str, Any]
    timestamp: datetime
    correlation_id: str = None
    priority: int = 1  # 1-10, higher is more urgent
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization"""
        return {
            'agent_id': self.agent_id,
            'message_type': self.message_type.value,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'correlation_id': self.correlation_id,
            'priority': self.priority
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPMessage':
        """Create message from dictionary"""
        return cls(
            agent_id=data['agent_id'],
            message_type=MCPMessageType(data['message_type']),
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            correlation_id=data.get('correlation_id'),
            priority=data.get('priority', 1)
        )


@dataclass
class AgentCapability:
    """Agent capability description"""
    name: str
    description: str
    input_types: list[str]
    output_types: list[str]
    confidence_level: float  # 0.0 to 1.0


@dataclass
class AgentStatus:
    """Agent status information"""
    agent_id: str
    status: str  # "idle", "busy", "error", "offline"
    current_task: str = None
    progress: float = 0.0  # 0.0 to 1.0
    last_update: datetime = None
    capabilities: list[AgentCapability] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'agent_id': self.agent_id,
            'status': self.status,
            'current_task': self.current_task,
            'progress': self.progress,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'capabilities': [cap.__dict__ for cap in (self.capabilities or [])]
        }
    
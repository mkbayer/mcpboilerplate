"""
Base agent class implementing MCP protocol for trend radar agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..models.mcp_message import MCPMessage, MCPMessageType, AgentCapability, AgentStatus
from ..utils.llm_client import LLMClient
from ..utils.logger import AgentLogger


class MCPAgent(ABC):
    """Base class for Model Context Protocol Agents"""
    
    def __init__(
        self,
        agent_id: str,
        llm_base_url: str = "http://localhost:11434",
        model_name: str = "gpt-oss:20b"
    ):
        self.agent_id = agent_id
        self.llm_base_url = llm_base_url
        self.model_name = model_name
        self.logger = AgentLogger(agent_id)
        
        # Agent state
        self.status = AgentStatus(
            agent_id=agent_id,
            status="idle",
            last_update=datetime.now(),
            capabilities=self._define_capabilities()
        )
        
        # MCP context and history
        self.context_window: List[MCPMessage] = []
        self.max_context_size = 10
        
        # Task tracking
        self.current_task: Optional[Dict[str, Any]] = None
        self.task_history: List[Dict[str, Any]] = []
    
    @abstractmethod
    def _define_capabilities(self) -> List[AgentCapability]:
        """Define agent capabilities - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process assigned task - must be implemented by subclasses"""
        pass
    
    async def send_message(
        self,
        message: MCPMessage,
        orchestrator
    ) -> None:
        """Send message through MCP to orchestrator"""
        self.logger.info(
            f"Sending {message.message_type.value} message",
            correlation_id=message.correlation_id
        )
        await orchestrator.receive_message(message)
    
    async def receive_message(self, message: MCPMessage) -> None:
        """Receive and process MCP message"""
        self.logger.info(
            f"Received {message.message_type.value} from {message.agent_id}",
            correlation_id=message.correlation_id
        )
        
        # Add to context window
        self._add_to_context(message)
        
        # Handle different message types
        if message.message_type == MCPMessageType.TASK_REQUEST:
            await self._handle_task_request(message)
        elif message.message_type == MCPMessageType.CONTEXT_SHARE:
            await self._handle_context_share(message)
        elif message.message_type == MCPMessageType.STATUS_UPDATE:
            await self._handle_status_update(message)
    
    def _add_to_context(self, message: MCPMessage) -> None:
        """Add message to context window"""
        self.context_window.append(message)
        
        # Maintain context window size
        if len(self.context_window) > self.max_context_size:
            self.context_window = self.context_window[-self.max_context_size:]
    
    async def _handle_task_request(self, message: MCPMessage) -> None:
        """Handle incoming task request"""
        self.status.status = "busy"
        self.status.current_task = message.content.get('task_type', 'unknown')
        self.status.progress = 0.0
        self.status.last_update = datetime.now()
        
        try:
            result = await self.process_task(message.content)
            self.status.progress = 1.0
            self.status.status = "idle"
            
            # Send response back
            response = MCPMessage(
                agent_id=self.agent_id,
                message_type=MCPMessageType.TASK_RESPONSE,
                content={
                    'result': result,
                    'status': 'completed',
                    'task_id': message.content.get('task_id')
                },
                timestamp=datetime.now(),
                correlation_id=message.correlation_id
            )
            
            # Note: In a real implementation, we'd send this back to orchestrator
            self.logger.info("Task completed successfully")
            
        except Exception as e:
            self.status.status = "error"
            self.logger.error(f"Task failed: {str(e)}")
            
            # Send error response
            error_response = MCPMessage(
                agent_id=self.agent_id,
                message_type=MCPMessageType.ERROR_REPORT,
                content={
                    'error': str(e),
                    'task_id': message.content.get('task_id')
                },
                timestamp=datetime.now(),
                correlation_id=message.correlation_id
            )
    
    async def _handle_context_share(self, message: MCPMessage) -> None:
        """Handle shared context from other agents"""
        self.logger.info(f"Received context share from {message.agent_id}")
        # Context is automatically added to context window
    
    async def _handle_status_update(self, message: MCPMessage) -> None:
        """Handle status updates from other agents"""
        self.logger.debug(f"Status update from {message.agent_id}: {message.content}")
    
    async def query_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """Query the LLM with context awareness"""
        self.logger.debug("Querying LLM", prompt_length=len(prompt))
        
        async with LLMClient(
            base_url=self.llm_base_url,
            model=self.model_name
        ) as llm:
            try:
                response = await llm.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                self.logger.debug("LLM response received", response_length=len(response))
                return response
                
            except Exception as e:
                self.logger.error(f"LLM query failed: {str(e)}")
                return f"Error querying LLM: {str(e)}"
    
    def get_context_summary(self) -> str:
        """Generate summary of current context window"""
        if not self.context_window:
            return "No context available"
        
        summary_parts = []
        for msg in self.context_window[-5:]:  # Last 5 messages
            summary_parts.append(
                f"{msg.agent_id}:{msg.message_type.value} - "
                f"{str(msg.content)[:100]}..."
            )
        
        return "Recent context:\n" + "\n".join(summary_parts)
    
    def update_progress(self, progress: float, status_message: str = None) -> None:
        """Update task progress"""
        self.status.progress = max(0.0, min(1.0, progress))
        self.status.last_update = datetime.now()
        
        if status_message:
            self.logger.info(f"Progress: {progress*100:.1f}% - {status_message}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return self.status.to_dict()
    
    async def health_check(self) -> bool:
        """Check agent health and LLM connectivity"""
        try:
            async with LLMClient(
                base_url=self.llm_base_url,
                model=self.model_name
            ) as llm:
                return await llm.health_check()
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return False
        
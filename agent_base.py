from abc import ABC, abstractmethod
from queue import Queue

class Agent(ABC):
    """
    Base class for agents in the multi-agent system.
    Each agent has an inbox (Queue) for receiving messages.
    """

    def __init__(self, name):
        self.name = name
        self.inbox = Queue()

    def send(self, recipient, message):
        """
        Send a message to another agent.
        """
        recipient.inbox.put((self.name, message))

    @abstractmethod
    def step(self):
        """
        Defines the agent's behavior per time step or tick.
        Should be implemented by subclasses.
        """
        pass

    def receive(self):
        """
        Receive a message from the inbox (if any).
        Returns (sender, message) or None if inbox is empty.
        """
        if not self.inbox.empty():
            return self.inbox.get()
        return None

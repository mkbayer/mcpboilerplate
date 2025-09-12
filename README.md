# MCP Multi-Agent Boilerplate

This is a Python boilerplate for building multi-agent systems in the style of an MCP (Multi-Agent Collaboration Platform).

## Structure

- `agent_base.py` – Defines the abstract `Agent` class with message passing.
- `agents/example_agents.py` – Example agents: `EchoAgent` and `GreeterAgent`.
- `orchestrator.py` – Runs the agents and coordinates their steps.
- `main.py` – Entrypoint that initializes and runs the system.

## Quick Start

```bash
# Clone the repository and navigate to it
git clone https://github.com/mkbayer/mcpboilerplate.git
cd mcpboilerplate

# Run the example
python main.py
```

## Extending

- Create new agent classes by subclassing `Agent` in `agent_base.py`.
- Implement the `step()` method with desired agent logic.
- Add agents in `main.py` and plug them into the `Orchestrator`.

## Example Output

```
--- Step 1 ---
[EchoAgent] received from GreeterAgent: Hello from GreeterAgent!
[GreeterAgent] received from EchoAgent: Echo: Hello from GreeterAgent!
--- Step 2 ---
--- Step 3 ---
--- Step 4 ---
--- Step 5 ---
```

## License

MIT
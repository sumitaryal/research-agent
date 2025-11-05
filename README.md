# Research Agent

Async-first research agent that plans, searches, reflects, and synthesizes grounded answers by orchestrating Google Gemini models with adaptive budgets and citation handling.

## Getting Started

- Python 3.12+ and a Google Gemini API key (`GEMINI_API_KEY` environment variable or `.env` file).
- Install dependencies: `pip install -e .` (uses `pyproject.toml`).
- Run the sample script: `python main.py` (updates `main.py` to change the user prompt).

## Usage

```python
from research_agent import ResearchAgent
import asyncio

agent = ResearchAgent()

# Synchronous convenience wrapper
sync_result = agent.chat("Summarize the AI research highlights from last week")
print(sync_result.final_answer)

# Asynchronous usage
async def main():
    agent = ResearchAgent()
    async_result = await agent.achat("Summarize the AI research highlights from last week")
    print(async_result.final_answer)

asyncio.run(main())
```

## Architecture

The agent hydrates configuration, generates an initial batch of queries, and hands control to a deterministic state machine that cycles through planning, searching, reflecting, and synthesizing until it meets confidence targets or exhausts budgets.

The execution flow works as follows:

1. A caller (CLI or library) instantiates `ResearchAgent`, which loads configuration from environment variables, `.env`, and runtime overrides.
2. `ResearchAgent` seeds an initial set of search queries using Gemini models before delegating control to `ResearchGraphRunner`.
3. `ResearchGraphRunner` executes a deterministic loop of plan → search → reflect → decide → synthesize steps, maintaining workflow state and progress metrics.
4. During each loop iteration the runner coordinates Google Gemini calls, adaptive budgeting heuristics (`budgeting.py`), shared utilities (`research_utils.py`, `prompts.py`), and structured Pydantic models (`research_models.py`), while de-duplicating and tracking unique sources.
5. Once synthesis completes, the runner produces a `ResearchRunResult` containing the final answer, citations, and telemetry that `ResearchAgent` returns to the caller.

## Key Components

- `research_agent/agent.py`: entry point; validates config, exposes `chat`/`achat`, generates search queries, and drives the research run.
- `research_agent/agent_runner.py`: small deterministic graph that launches searches, schedules reflections, and synthesizes answers.
- `research_agent/budgeting.py`: adaptive heuristics for thinking budgets, initial query sizing, and telemetry.
- `research_agent/research_utils.py`: helper utilities for topic extraction, URL normalization, and citation formatting.
- `research_agent/research_models.py`: Pydantic models describing structured prompts, reflections, and final results.
- `main.py`: minimal CLI example for running a single research request.

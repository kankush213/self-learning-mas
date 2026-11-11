# Your First Multi-Agent System

A small Python + LangGraph example with real OpenAI LLM calls.
Start by reading `self-learning-mas.py`; all application logic is in that file.

## Run on Windows (PowerShell)

The local `.venv` is already set up. Add your key to the existing `.env` file:

```dotenv
OPENAI_API_KEY=your_actual_key
OPENAI_MODEL=gpt-4o-mini
```

Do not share your key or commit `.env`. An existing `OPENAI_API_KEY` environment
variable takes precedence over `.env`. Model availability depends on your account;
you can change `OPENAI_MODEL` to another supported OpenAI chat model.

```powershell
.\.venv\Scripts\python.exe self-learning-mas.py
.\.venv\Scripts\python.exe self-learning-mas.py "Why is the sky blue?"
```

Each run makes **three model calls** and uses paid OpenAI API credits.
ChatGPT subscriptions do not include API usage. Your topic and intermediate
outputs are sent to OpenAI, so avoid sensitive information.

To recreate the environment on another machine (Python 3.10 or newer):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Architecture

```text
START -> Researcher -> Writer -> Reviewer -> END
           |            |           |
        research      draft     final_answer
           +------------+-----------+
                    TeamState
```

1. **Shared state (`TeamState`)** is a dictionary holding the topic and each
   agent's work. Think of it as the team's shared notebook.
2. **Researcher** reads the topic and returns research notes.
3. **Writer** reads the topic and notes, then returns a draft.
4. **Reviewer** reads the notes and draft, then returns the final explanation.
5. **Graph edges** decide who runs next. LangGraph merges each agent's returned
   dictionary into the shared state before starting the next agent.

Each agent is an ordinary Python function with a different system prompt (role).
They reuse the same model client, but make separate calls. The model does not
remember earlier calls automatically: the state explicitly supplies the context.
The terminal shows each agent's output so you can follow the handoffs.

This is a minimal, fixed multi-agent workflow, not an autonomous team. There is
no supervisor, web search, tool use, parallel execution, or revision loop. The
researcher uses model knowledge, and the reviewer can still miss factual errors.
State lives only for the current run; nothing is saved to a database.

## Execution Logs

Run the same command as before; timestamped INFO logs appear automatically in the
terminal alongside the agent outputs. They show configuration checks, graph
compilation, initial state, each agent's start and model call, state updates, and
completion. Third-party INFO logs are suppressed to keep the flow readable.

For example (timestamps and response lengths will vary):

```text
14:30:00 | INFO | [1/3] Researcher started. Reading state: topic.
14:30:00 | INFO | Researcher: calling OpenAI; waiting for research notes.
14:30:02 | INFO | Researcher finished. Returning state update: research (450 characters).
14:30:02 | INFO | State updated by researcher. Updated fields: research.
14:30:02 | INFO | [2/3] Writer started. Reading state: topic, research.
```

The logs describe execution, not the model's internal reasoning. They do not
include API keys, full prompts, or raw API error details. Agent responses are
still printed as before. Logs are terminal-only and are not saved to a file.
If an OpenAI call fails, the last agent-start message identifies where execution
stopped, followed by an ERROR message and the existing troubleshooting guidance.

## What to Read First

1. `TeamState`: what information is passed around?
2. The three functions inside `build_team`: what does each role receive and return?
3. `graph.add_edge(...)`: who gets control next?
4. `team.stream(...)`: how do we run the graph and display completed steps?

Try changing the writer's prompt to explain things to a ten-year-old, then rerun
the same topic and compare the draft and final answer.

## Offline Test

```powershell
.\.venv\Scripts\python.exe -m unittest -v
```

The test uses fake model responses with the real LangGraph graph. It verifies
execution order and that notes and drafts reach the next agent. It needs no key
and makes no network requests; it does not verify OpenAI credentials or billing.
# Multi-Agent System Flow Diagrams

These diagrams describe the application in `self-learning-mas.py`.
View this file in a Markdown viewer that supports Mermaid to render the diagrams.

## Program Architecture

Three role-specific agents share one OpenAI model client. LangGraph passes their
outputs through `TeamState`.

```mermaid
flowchart TD
    Topic["User's topic"] --> R["Researcher agent"]
    R -->|"Research notes"| W["Writer agent"]
    W -->|"Draft + research notes"| V["Reviewer agent"]
    V --> Final["Final explanation"]

    R -.-> LLM["OpenAI model"]
    W -.-> LLM
    V -.-> LLM

    State["Shared TeamState<br/>topic / research / draft / final_answer"]
    State -.- R
    State -.- W
    State -.- V
```

## Program Execution

This follows the application's `main()` entry point.

```mermaid
flowchart TD
    Start(["Run script"]) --> Args["Read topic from command line<br/>or use the default topic"]
    Args --> TopicCheck{"Topic non-empty?"}
    TopicCheck -->|No| Invalid["Show argument error and exit"]
    TopicCheck -->|Yes| Env["Load .env"]
    Env --> Key{"API key configured?"}
    Key -->|No| Missing["Show setup message and exit"]
    Key -->|Yes| Setup["Create OpenAI client<br/>Build and compile LangGraph"]
    Setup --> State["Initialize shared state"]
    State --> Research["Researcher calls OpenAI<br/>Store research notes"]
    Research --> PrintR["Print researcher output"]
    PrintR --> Write["Writer calls OpenAI with notes<br/>Store draft"]
    Write --> PrintW["Print writer output"]
    PrintW --> Review["Reviewer calls OpenAI with notes and draft<br/>Store final answer"]
    Review --> PrintV["Print final answer"]
    PrintV --> End(["Finish"])

    Research -->|API error| Error["Show troubleshooting message and exit"]
    Write -->|API error| Error
    Review -->|API error| Error
```

The agents run sequentially, not in parallel. Each makes a separate model call;
shared state supplies the context between calls. The researcher uses the model's
existing knowledge, not web search. The key check detects missing configuration,
not whether OpenAI will accept the key.
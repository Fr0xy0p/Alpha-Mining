# AlphaMiningV2

An **Autonomous Alpha Factor Mining System** designed for the WorldQuant Brain platform. At its core is an LLM-driven Direct Agent that operates like a professional quantitative researcher: using tool calls to explore datasets, inspect operators, formulate FASTEXPR alpha expressions, submit backtests, analyze performance metrics, and iteratively refine factors until convergence.

```
Research → Submit → Poll → Analyze → Improve → ... → Converge
         ↕ tools (15 tools)
  Datasets · Fields · Operators · Settings · Knowledge Base
```

---

## Quick Start

```bash
# Start from investment intuition / idea
python -m agent.direct.cli \
  --idea "Momentum combined with low volatility"

# Specify dataset and jump directly into research
python -m agent.direct.cli \
  --dataset-id pv13 \
  --idea "Momentum combined with low volatility"

# Improve an existing expression with a targeted critique
python -m agent.direct.cli \
  --dataset-id pv13 \
  --expression "group_rank(ts_mean(returns,21),industry)" \
  --critique "High turnover, try longer lookback"
```

Credentials must be configured prior to execution. See [Environment Configuration](#environment-configuration).

---

## Directory Structure

```
alphaminingv2/
├── agent/                           # Core Agent Framework
│   ├── direct/                     #   DirectAgent architecture
│   │   ├── agent.py                #     DirectAgent orchestrator
│   │   ├── cli.py                  #     CLI entry point
│   │   ├── job_store.py            #     Job state, results, and convergence tracking
│   │   ├── llm_conversation.py     #     LLM chat, tool execution & conversation history
│   │   ├── phase_manager.py        #     Creative/Improvement phase state machine & directions
│   │   ├── prompt_builder.py       #     Stateless prompt construction
│   │   ├── response_processor.py   #     LLM response parsing & phase dispatch
│   │   └── submission_engine.py    #     Factor submission, FIFO queue & polling
│   ├── wq_tools.py                 #   WorldQuant Brain tool layer (16 callable tools)
│   ├── llm_client.py               #   OpenAI-compatible / DeepSeek API client
│   ├── llm_logger.py               #   Structured LLM exchange logger
│   ├── expression_validator.py     #   FASTEXPR syntax & field validation
│   ├── prompts.py                  #   System & discovery prompts
│   │
│   ├── multi/                      #   Multi-Agent framework (experimental)
│   │   ├── cli.py                  #     Multi-Agent CLI
│   │   ├── orchestrator.py         #     Multi-Agent orchestrator (OrchestratorV2)
│   │   ├── explorer.py             #     LLM expression explorer
│   │   ├── param_optimizer.py      #     Parameter grid optimizer
│   │   ├── idea_scout.py           #     Autonomous factor idea scout
│   │   ├── expression_fingerprint.py # Expression structural deduplication
│   │   └── results_store.py        #     Shared SQLite results database
│   └── config.py                   #   Agent thresholds & simulation configurations
│
├── pipeline/                        # Backtest Pipeline (Infrastructure Layer)
│   ├── backtest_runner.py          #   Concurrent backtest execution engine
│   ├── datafields_store.py         #   Dataset fields fetcher & local cache
│   ├── result_filter.py            #   Result filtering & Core-level aggregation
│   ├── adaptive_scheduler.py       #   Probe-Expand closed-loop scheduler
│   └── run_pipeline.py             #   All-in-one automated pipeline runner
│
├── wq_operators_cleaned.json       # Cleaned WorldQuant Brain operator definitions
├── template_catalog.json           # Expression template catalog
├── template_catalog_mixed.json     # Multi-dataset template catalog
├── common_operator_slot_mappings.json # Operator slot definitions
├── .env.example                    # Environment variable template
│
├── agent_output/                   # Session outputs & logs (gitignored)
├── datafields_cache/               # Cached data fields (gitignored)
└── backtest_results/               # Backtest results & reports (gitignored)
```

---

## Environment Configuration

### Credentials

Configure the following variables in a `.env` file at the project root or in your process environment:

| Variable | Description | Purpose |
| --- | --- | --- |
| `BRAIN_USERNAME` | WorldQuant Brain account username | Backtest authentication & submission |
| `BRAIN_PASSWORD` | WorldQuant Brain account password | Backtest authentication & submission |
| `GEMINI_API_KEY` | Google Gemini API key (`AIza...`) | Agent LLM reasoning (default: `gemini-2.5-flash`) |
| `GROQ_API_KEY` | Groq API key (`gsk_...`) | Agent LLM reasoning (default: `openai/gpt-oss-120b`) |
| `DEEPSEEK_API_KEY` | DeepSeek API key (alternative) | Agent LLM reasoning (default: `deepseek-chat`) |
| `LLM_MODEL` *(optional)* | Custom model name | Override default model (e.g. `gemini-2.5-flash`, `gemini-2.0-flash`) |
| `LLM_BASE_URL` *(optional)* | Custom API base URL | Custom OpenAI-compatible endpoint |

*(Optional)* Network proxies can be set via `HTTP_PROXY` and `HTTPS_PROXY` if required by your network environment.

### Data Fields Caching

Before running offline or targeted simulations, fetch and cache target dataset field definitions using `datafields_store.py`:

```bash
python pipeline/datafields_store.py --dataset-id pv13
```

---

## Core Concepts

### DirectAgent — Event-Driven Factor Mining Loop

The LLM interacts with WorldQuant Brain through 15 tools in an event-driven loop, traversing the full research cycle from hypothesis formulation to convergence:

```
                    ┌──────────────┐
                    │   Research   │
                    │ Exploration  │
                    └──────┬───────┘
                           ↓ Investment Hypothesis
              ┌────────────────────────┐
              │    Iteration Loop      │
              │                        │
              │  ┌──────────────────┐  │
              │  │  Creative Phase  │  │  ← Propose new directions (multi-directional exploration)
              │  └────────┬─────────┘  │
              │           ↓            │
              │  ┌──────────────────┐  │
              │  │Improvement Phase │  │  ← Deepen existing directions (parameter tuning & mutations)
              │  └────────┬─────────┘  │     Stall detected → Switch back to Creative
              │           ↓            │
              │      Converged?        │──→ Below target → Continue loop
              └───────────┬────────────┘
                          ↓ Target achieved
                    ┌───────────┐
                    │   Stop    │
                    │Converged  │
                    └───────────┘
```

#### Phase State Machine

The iteration cycle alternates between two **mutually exclusive phases** managed by the `PhaseManager`:

| Phase | Behavior | Entry Condition | Transition Condition |
| --- | --- | --- | --- |
| **Creative** | The LLM proposes new directions based on current findings, each accompanied by initial expressions | Session start / Improvement limit reached or progress stalled | Automatically switches to **Improvement** after submission is complete |
| **Improvement** | The LLM refines expressions within existing active directions; creating new directions is disallowed | Entered after Creative submissions are confirmed | Maximum 8 rounds reached / 5 consecutive rounds without Sharpe improvement / all directions abandoned |

These alternating phases balance exploration and exploitation:
- **Creative Phase** prevents local optima by forcing periodic perspective shifts and new hypotheses.
- **Improvement Phase** prevents superficial testing by forcing deep refinement and parameter optimization on promising directions.

Each state transition resets phase counters to give the agent a full turn allocation in the new phase.

### End-to-End Workflow

1. **Research (Phase 1)** — The LLM calls tools to explore datasets, field definitions, operator capabilities, and simulation settings to form a testable financial hypothesis.
2. **Creative (Phase 2a)** — The LLM submits initial directions with corresponding candidate expressions, which automatically transitions to Improvement.
3. **Improvement (Phase 2b)** — As backtest jobs finish, the LLM analyzes metrics and submits improvements linked to active directions (maintaining up to 3 concurrent simulations).
4. **Phase Switch** — When progress stalls or round limits are met, the state machine switches back to Creative to formulate fresh ideas.
5. **Convergence** — When a factor satisfies quality targets (Sharpe ≥ 1.25, Fitness ≥ 1.0, valid Turnover range), the session converges and outputs a final report.

### Reliability & Fault Tolerance

- **Transient Network Handling**: Network interruptions (connection resets, gateway timeouts, 5xx status codes) are retried with exponential backoff and transparently transferred to a background polling queue.
- **Factor Error Isolation**: Malformed expressions (syntax errors, invalid operator arguments) fail immediately without consuming retry quotas or interrupting the agent session.
- **Stall Guards**: Automated heartbeat monitoring and consecutive stall detectors prevent deadlocks during research or iteration.

### Knowledge Base

Reusable insights, discovered anomalies, and successful factor patterns are saved to `agent_output/knowledge_base.json`. In future sessions, relevant historical entries are automatically retrieved and injected into the LLM context (capped at 2 entries per iteration and 10 per session).

---

## Experimental: Multi-Agent Architecture

In addition to DirectAgent, AlphaMiningV2 includes a decoupled multi-agent architecture:

- **OrchestratorV2**: Coordinates agent workflows, manages state, and aggregates performance.
- **Explorer**: LLM agent focused on broad hypothesis discovery and novel factor expressions.
- **ParamOptimizer**: Specialized agent that sweeps lookbacks, operators, and neutralization parameters.
- **IdeaScout**: Web-connected agent that searches academic literature (arXiv, SSRN) and quantitative blogs for actionable ideas.
- **ResultsStore**: SQLite-backed storage for tracking factor lineages, expressions, and simulation metrics.

Launch the multi-agent system:
```bash
python -m agent.multi.cli --dataset-id pv13 --idea "Earnings quality and cashflow accruals"
```

---

## Pipeline Execution (Batch Infrastructure)

For systematic template-based factor evaluation:

```bash
# Run full automated Probe-Expand pipeline
python -m pipeline.run_pipeline --dataset-id pv13

# Probe-only mode (screen initial representative parameter grid)
python -m pipeline.run_pipeline --dataset-id pv13 --probe-only

# Filter and aggregate backtest results by Core
python -m pipeline.result_filter --results-dir backtest_results/expand/pv13 --group-by-core
```

---

## Testing

AlphaMiningV2 includes comprehensive test suites with mock validation that do not require active Brain credentials:

```bash
# Run pipeline logic and flow-control verification
python test_pipeline.py

# Run end-to-end template, slot domain, and scheduler tests
python test_improvements.py
```

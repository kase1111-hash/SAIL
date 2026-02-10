# SAIL Refocus Plan

This plan restructures SAIL from 22,500 lines of disconnected components into a working product. It is organized into 4 phases, each ending with a demonstrable milestone.

---

## Guiding Principle

**Ship one vertical slice before building any more horizontal infrastructure.**

The current codebase has real, tested modules for voice input, LLM inference, knowledge retrieval, intervention logic, context management, and TTS output -- but zero code that connects them. Every phase below ends with something a user can interact with.

---

## Phase 0: Clean Cut (Day 1)

**Goal:** Remove dead weight. Reduce surface area from ~22,500 lines to ~15,000 lines of code that matters.

### 0.1 Delete deployment module (1,539 lines)

Every function body in this module is `await asyncio.sleep(N)`. No production code imports it.

| File | Lines | Action |
|------|-------|--------|
| `src/deployment/provisioning.py` | 414 | Delete |
| `src/deployment/ota.py` | 501 | Delete |
| `src/deployment/health.py` | 579 | Delete |
| `src/deployment/__init__.py` | 45 | Delete |
| `tests/unit/test_deployment.py` | 627 | Delete |

**Dependency check:** Only test files import from `src/deployment/`. CLI does not. Safe to delete.

### 0.2 Delete mobile and vehicle nodes (1,315 lines)

No mobile app exists. OBD-II connection is simulated (`connect()` returns `True` unconditionally).

| File | Lines | Action |
|------|-------|--------|
| `src/nodes/mobile.py` | 605 | Delete |
| `src/nodes/vehicle.py` | 710 | Delete |

**Required fix:** Edit `src/nodes/__init__.py` to remove the imports and `__all__` entries for `MobileNode`, `MobileNodeConfig`, `VehicleNode`, `VehicleNodeConfig`.

### 0.3 Shelve hub, remaining nodes, and integration tests (3,900 lines)

These modules work but serve a distributed architecture that is premature. Leave in the tree but mark as deferred -- do not invest further until Phase 3.

| File | Lines | Status |
|------|-------|--------|
| `src/hub/coordinator.py` | 766 | Defer |
| `src/hub/context_sync.py` | 575 | Defer |
| `src/hub/router.py` | 469 | Defer |
| `src/hub/base.py` | 224 | Defer (keep -- nodes depend on its types) |
| `src/hub/__init__.py` | 40 | Defer |
| `src/nodes/base.py` | 666 | Defer |
| `src/nodes/room.py` | 519 | Defer |
| `src/nodes/protocol.py` | 504 | Defer |
| `src/nodes/__init__.py` | 46 | Update (remove mobile/vehicle) |
| `tests/unit/test_hub.py` | 818 | Defer |
| `tests/unit/test_nodes.py` | 757 | Defer |
| `tests/integration/test_node_integration.py` | 626 | Delete (imports deleted deployment module) |

### 0.4 Shelve multi-user system (4,295 lines)

No production code imports from `src/users/`. Family features require a working single-user product first.

| File | Lines | Status |
|------|-------|--------|
| `src/users/manager.py` | 726 | Defer |
| `src/users/guardian.py` | 741 | Defer |
| `src/users/briefings.py` | 607 | Defer |
| `src/users/voice_id.py` | 589 | Defer |
| `src/users/base.py` | 556 | Defer |
| `src/users/permissions.py` | 536 | Defer |
| `src/users/isolation.py` | 415 | Defer |
| `src/users/__init__.py` | 125 | Defer |
| `tests/unit/test_users.py` | 876 | Defer |

### Phase 0 Outcome

~4,100 lines deleted. ~9,000 lines deferred. Remaining ~13,000 lines are the core that needs to be wired together:

- `src/config/` -- Configuration (working)
- `src/core/llm/` -- LLM integration (working)
- `src/voice/` -- Voice I/O (working components)
- `src/context/` -- Context buffer (working)
- `src/intervention/` -- Intervention engine (working)
- `src/knowledge/` -- Knowledge retrieval (working)
- `src/sensors/` -- Sensor fusion (working, partially premature)
- `src/cli.py` -- Entry point (needs rewrite)

---

## Phase 1: Minimal End-to-End Pipeline (Week 1-2)

**Goal:** A user says "hey sail, what are my rights during a traffic stop in California?" and gets a spoken answer. Text-mode fallback for environments without audio hardware.

### 1.1 Wire the CLI (`src/cli.py`)

Replace the placeholder `while True: pass` at line 82 with an async main loop that instantiates and connects components.

**New `run()` command structure:**

```python
@cli.command()
@click.option("--text-mode", is_flag=True, help="Use text I/O instead of voice")
@click.pass_context
def run(ctx: click.Context, text_mode: bool) -> None:
    """Start the SAIL assistant."""
    config = load_config(ctx.obj.get("config_path"))
    asyncio.run(_run_async(config, text_mode))


async def _run_async(config: Config, text_mode: bool) -> None:
    """Async main loop."""
    # 1. Create LLM provider
    llm = LLMProviderFactory.from_config(config.llm)

    # 2. Create context manager
    context_mgr = await create_context_manager(config.context)
    await context_mgr.start()

    # 3. Create knowledge pipeline
    knowledge = await create_initialized_rag_pipeline(config.knowledge)

    # 4. Create prompt library
    prompts = get_prompt_library()

    # 5. Choose I/O mode
    if text_mode:
        await _text_loop(llm, context_mgr, knowledge, prompts, config)
    else:
        voice_in = await create_voice_input_manager(config.voice)
        voice_out = await create_voice_output_manager(config.voice)
        await _voice_loop(llm, context_mgr, knowledge, prompts, voice_in, voice_out)
```

**Key wiring -- the query handler (shared by both text and voice modes):**

```python
async def _handle_query(
    query: str,
    llm: LLMProvider,
    context_mgr: ContextBufferManager,
    knowledge: RAGPipeline,
    prompts: PromptLibrary,
    config: Config,
) -> str:
    """Process a single user query through the full pipeline."""
    # 1. Add user input to context
    context_mgr.add_user_input(query)

    # 2. Retrieve relevant knowledge
    rag_result = await knowledge.retrieve(query, max_results=3)

    # 3. Build system prompt with context
    system_prompt = prompts.render("system_base",
        user_name="user",
        location_context="unknown",
        time_context=datetime.now().strftime("%H:%M"),
        intervention_mode="ambient",
    )

    # 4. Build message list
    messages = [
        Message(role="system", content=system_prompt),
    ]
    if rag_result.augmented_context:
        messages.append(Message(role="system", content=f"Relevant knowledge:\n{rag_result.augmented_context}"))

    # Add conversation history from context buffer
    for entry in context_mgr.get_conversation(max_turns=6):
        role = "user" if entry.entry_type == EntryType.USER_INPUT else "assistant"
        messages.append(Message(role=role, content=entry.content))

    # Current query
    messages.append(Message(role="user", content=query))

    # 5. Generate response
    result = await llm.generate(messages)

    # 6. Store response in context
    context_mgr.add_assistant_response(result.content)

    return result.content
```

**Text loop (simplest path -- get this working first):**

```python
async def _text_loop(llm, context_mgr, knowledge, prompts, config):
    """Simple text I/O loop for testing without audio hardware."""
    console.print("[green]SAIL ready. Type your question (Ctrl+C to exit).[/green]\n")
    while True:
        query = input("You: ").strip()
        if not query:
            continue
        response = await _handle_query(query, llm, context_mgr, knowledge, prompts, config)
        console.print(f"\n[cyan]SAIL:[/cyan] {response}\n")
```

**Voice loop (requires audio hardware):**

```python
async def _voice_loop(llm, context_mgr, knowledge, prompts, voice_in, voice_out):
    """Full voice I/O loop."""
    await voice_in.start()
    await voice_out.start()
    console.print("[green]SAIL listening. Say 'hey sail' to activate.[/green]")

    async for utterance in voice_in.run():
        if utterance.was_stopped or not utterance.text:
            continue
        response = await _handle_query(utterance.text, llm, context_mgr, knowledge, prompts, config)
        await voice_out.speak(response)
```

### 1.2 Add `--text-mode` integration test

Create `tests/integration/test_pipeline.py` that:
1. Instantiates the LLM provider (mocked), context manager, and knowledge pipeline
2. Sends a query through `_handle_query`
3. Asserts the response is non-empty and context was updated
4. Asserts knowledge retrieval was attempted

This is the most important test in the project. It validates the one thing that currently doesn't work: components talking to each other.

### 1.3 Verify with real Ollama

Manual test with a running Ollama instance:
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Run SAIL in text mode
sail run --text-mode
You: What are my rights during a traffic stop in California?
# Expected: Coherent response referencing 4th/5th amendment, jurisdiction-specific details
```

### Phase 1 Milestone

`sail run --text-mode` produces answers by routing queries through context management and knowledge retrieval to a local LLM. Voice mode works where audio hardware is available. The core product hypothesis is validated.

---

## Phase 2: Intervention & Knowledge Quality (Week 3-4)

**Goal:** SAIL proactively guides the user based on context, not just reacts to questions. Knowledge responses cite sources and are jurisdiction-aware.

### 2.1 Wire the intervention engine into the main loop

The intervention engine (`src/intervention/engine.py`) already has real mode transition logic with risk thresholds. Connect it:

```python
async def _handle_query(query, llm, context_mgr, knowledge, prompts, intervention, config):
    # ... existing pipeline ...

    # After generating response, evaluate for intervention
    context_dict = {
        "query": query,
        "response": result.content,
        "domain": rag_result.domain_searched,
    }
    intervention_result = await intervention.evaluate(context_dict)
    if intervention_result:
        # Prepend or append intervention guidance to response
        response = f"{result.content}\n\n⚠ {intervention_result.content}"
    else:
        response = result.content

    return response
```

### 2.2 Add citation formatting to knowledge responses

The RAG pipeline already returns `citations: list[str]` in `RAGResult`. Surface them:

```python
if rag_result.citations:
    response += "\n\nSources: " + ", ".join(rag_result.citations)
```

### 2.3 Index the knowledge domain data

The knowledge domains (`src/knowledge/domains/`) contain 3,041 lines of curated content (legal rights, safety procedures, financial scam patterns, emergency protocols) stored as Python dictionaries. Write a startup task that indexes them into the RAG pipeline's vector store:

```python
async def _index_knowledge_domains(knowledge: RAGPipeline):
    """Index all knowledge domain items at startup."""
    from src.knowledge.domains import legal, safety, financial, emergency
    # Collect all KnowledgeItem instances from domain dictionaries
    # Call knowledge.index_items(items)
```

### 2.4 Sensor context (minimal)

Wire the temporal sensor only (no hardware required). It provides time-of-day context that the LLM can use ("it's 2 AM -- are you safe?").

```python
from src.sensors.temporal import TemporalSensor
temporal = TemporalSensor()
# Add temporal context to system prompt
```

### Phase 2 Milestone

`sail run --text-mode` returns jurisdiction-aware answers with citations. Intervention engine activates for high-risk queries (financial scam patterns, emergency situations). Temporal context influences response tone.

---

## Phase 3: Code Quality & Patterns (Week 5-6)

**Goal:** Eliminate the copy-paste debt that will slow future development.

### 3.1 Extract `EventEmitter` mixin

**15 files** implement identical callback registration pattern. Replace with:

```python
# src/core/events.py
class EventEmitter(Generic[T]):
    """Mixin for event emission. Replaces 15 copy-pasted implementations."""

    def __init__(self):
        self._event_callbacks: list[Callable[[T], None]] = []

    def on_event(self, callback: Callable[[T], None]) -> None:
        self._event_callbacks.append(callback)

    def _emit_event(self, event: T) -> None:
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")
```

**Files to refactor:** `context/manager.py`, `hub/coordinator.py`, `voice/manager.py`, `voice/tts/manager.py`, `nodes/base.py`, `knowledge/manager.py`, `intervention/engine.py`, `sensors/motion.py`, `sensors/fusion.py`, `sensors/location.py`, `sensors/audio.py`, `users/permissions.py`, `users/voice_id.py`, `users/guardian.py`, `users/manager.py`.

### 3.2 Standardize `get_stats()` pattern

**17 files** implement nearly identical stats methods. Create a protocol:

```python
# src/core/stats.py
class HasStats(Protocol):
    def get_stats(self) -> dict[str, Any]: ...
```

Standardize the return shape across all implementations. This enables a future `/stats` CLI command that aggregates all subsystem stats.

### 3.3 Refactor knowledge domains

**4 domain files (3,041 lines)** share identical structure: same imports, same init pattern, same query method, same caching. Extract the common structure:

```python
# src/knowledge/domains/base_domain.py
class BaseKnowledgeDomain(KnowledgeDomain):
    """Base class for knowledge domains with common query, cache, and init logic."""
    def __init__(self, domain_type: KnowledgeDomainType, data: dict[str, KnowledgeItem], ...):
        ...
    async def query(self, query: KnowledgeQuery) -> KnowledgeResult:
        # Common query logic currently duplicated across all 4 files
        ...
```

Each domain file reduces from ~750 lines to ~200 lines (mostly data definitions).

**Estimated savings:** ~2,000 lines eliminated.

### 3.4 Collapse single-implementation ABCs

`TTSProvider` in `src/voice/tts/base.py` has one implementation (`PiperProvider`). Merge the interface into the concrete class. If a second TTS engine is added later, extract the interface then. Same evaluation for other ABCs with single implementations.

### Phase 3 Milestone

Codebase is cleaner, with shared patterns extracted and domain duplication eliminated. No behavioral changes -- all existing tests continue to pass.

---

## Phase 4: Re-enable Deferred Systems (Week 7+)

**Goal:** Bring back shelved features, now that they have a working product to plug into.

### 4.1 Multi-user support

Re-integrate `src/users/` with the working pipeline:
- Voice identification to detect who is speaking
- Per-user context isolation (separate conversation histories)
- Role-based response filtering (minors get age-appropriate content)
- Guardian alerts when minors ask about certain topics

**Prerequisite:** Phase 1 pipeline must be working. Users module plugs into it, not the other way around.

### 4.2 Sensor fusion

Wire remaining sensors (location, motion, audio environment) into the main loop to enrich the context passed to the LLM. Location enables jurisdiction detection. Motion enables driving-mode responses. Audio environment enables noise-adaptive TTS volume.

**Prerequisite:** Phase 2 temporal sensor integration proves the pattern.

### 4.3 Hub-and-nodes architecture

Only after single-machine SAIL is stable. Room nodes (Raspberry Pi with mic) are the first target. Mobile and vehicle nodes come later, when there's something real to run on them.

**Prerequisite:** Phases 1-3 complete and stable.

---

## Dependency Graph

```
Phase 0: Clean Cut
    │
    ▼
Phase 1: End-to-End Pipeline  ← This is the critical path
    │
    ├──▶ Phase 2: Intervention & Knowledge  (builds on Phase 1)
    │
    └──▶ Phase 3: Code Quality  (can run in parallel with Phase 2)
              │
              ▼
         Phase 4: Re-enable Deferred  (requires Phase 1-3)
```

---

## Success Metrics

| Metric | Current | After Phase 1 | After Phase 4 |
|--------|---------|---------------|---------------|
| End-to-end queries possible | 0 | Yes (text + voice) | Yes (multi-user, multi-node) |
| Lines of dead/skeleton code | ~3,000 | 0 | 0 |
| Duplicate callback patterns | 15 files | 15 files | 1 mixin |
| Knowledge domain duplication | ~2,000 lines | ~2,000 lines | ~0 lines |
| Integration tests for pipeline | 0 | 1+ | 5+ |
| CLI commands that work | 4 of 5 | 5 of 5 | 5 of 5 |

---

## What NOT To Do

1. **Do not add new features until Phase 1 is complete.** No new knowledge domains, no new sensor types, no new node types.
2. **Do not refactor before wiring.** Phase 3 comes after Phase 1 because working code with duplication is better than clean code that doesn't run.
3. **Do not build mobile/vehicle nodes.** The mobile node has no app. The vehicle node has no real OBD-II. These are Phase 4+ if ever.
4. **Do not design for scale.** SAIL runs on one machine for one family. The hub-and-nodes architecture is deferred, not deleted, but it is not the priority.
5. **Do not write more ABCs.** Every new abstraction layer costs maintenance. Only abstract when you have 2+ concrete implementations that share real behavior.

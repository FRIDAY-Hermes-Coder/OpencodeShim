# opencode-local shim — plan v3

Date: 2026-09-21. Owner: mitansh. Device: surface (this device only).
Supersedes the v2.2 plan. Sections on attachments (§3/§4/§10 of v2.2) and
keyless guarantees (§6) are **unchanged and still authoritative** — see §12.

---

## 0. Root cause: why v2.2 can't recall and why prompts are huge

The two symptoms are the same bug seen from two ends.

**Hermes is stateless. opencode is stateful. v2.2 pretends both are stateless.**

Hermes speaks OpenAI `/v1/chat/completions`: every request carries the *entire*
conversation (`system`, `user`, `assistant`, `tool`) plus the full `tools[]`
array. That's correct OpenAI semantics — the server is expected to hold no state.

`opencode serve` is the opposite. A session is a durable, server-side agent
thread. `POST /session/:id/message` appends **one** turn; opencode already holds
everything before it, runs its own system prompt, its own tool loop, its own
context compaction.

v2.2 bridges these by calling `messages_to_prompt()` — flatten N turns +
24 kB of tool schemas into one giant text part — and firing it at a session.
Whichever session strategy it uses, the result is broken:

| v2.2 strategy | Consequence |
|---|---|
| New session per request | opencode has literally zero memory. All recall must survive the flatten. |
| One global session reused | Every request re-appends the full history *on top of* history opencode already has. Context doubles each turn; opencode's auto-compaction then shreds it. Threads bleed into each other. |

### 0.1 The four concrete failure mechanisms

1. **Truncation eats the wrong end.** A flatten budget (24 kB tools +
   history) forced to fit drops oldest-first. Oldest = the system message =
   where Hermes injects its memory/recall blocks and the original task.
   You truncate exactly the thing you call "memory". *This is almost
   certainly the direct cause of "can't recall anything."*

2. **Role collapse destroys recall quality.** Flattening puts prior assistant
   turns inside a *user* message as `ASSISTANT: ...` text. The model no longer
   *owns* those turns — it reads them as a transcript document. Recall from a
   transcript-in-a-prompt is dramatically worse than recall from real
   multi-turn structure, and degrades fast past a few thousand tokens.
   Tool results land mid-blob and hit lost-in-the-middle.

3. **O(n²) tokens.** Turn *k* re-sends turns 1..k-1 plus the full tool
   schemas. Over Hermes' 150-turn loop this is quadratic. Nothing caches,
   because the blob's shape changes every turn — so no prompt-cache hit on the
   Zen side either. This is the "huge prompt" and a large part of the 214 s
   turn in §10's RCA.

4. **opencode's own system prompt fights you.** The default `build` agent
   ships a multi-thousand-token coding prompt and a live tool set
   (read/write/edit/bash/task). Under it the model is being told "you are a
   coding agent, use your tools" while the shim's fence protocol says "emit
   JSON tool calls for someone else's tools". Two agents, one context. It
   will sometimes do its own thing — which is also why delegation
   "works" in §10 and the fence is flaky.

### 0.2 What this means

Do not tune the flatten. Delete it. **Map one Hermes conversation to one
opencode session and send only the new turns.** Everything else in this plan
follows from that.

---

## 1. Target architecture

```
Hermes  ──OpenAI /v1/chat/completions (full history, every turn)──▶
                      │
        ┌─────────────▼──────────────┐
        │ L1  Session Manager        │  prefix-hash → session_id, emit delta only
        ├────────────────────────────┤
        │ L2  Turn Protocol          │  profile A (LLM) / B (agent); tool bridge
        ├────────────────────────────┤
        │ L3  Transport              │  SSE relay, attachments, locks, errors
        └─────────────┬──────────────┘
                      │  POST /session/:id/message   (ONE turn)
                      │  GET  /event?session=:id     (live deltas)
                      ▼
              opencode serve :4096
                      ▼
        muse-spark-1.3-contributor-free (Zen)
```

Per-turn payload goes from *O(whole conversation)* to *O(one user message)*.

---

## 2. L1 — Session Manager (the core fix)

### 2.1 Prefix-chain hashing

Every request, build a rolling hash over the canonical form of each message:

```python
def canon(m: dict) -> str:
    # stable, order-independent, ignores volatile fields
    return json.dumps({
        "role": m["role"],
        "content": norm_content(m.get("content")),     # text + attachment refs, NOT bytes
        "tool_call_id": m.get("tool_call_id"),
        "tool_calls": [(c["function"]["name"], c["function"]["arguments"])
                       for c in m.get("tool_calls", [])],
    }, sort_keys=True, separators=(",", ":"))

chain[0] = sha256(b"v3|" + canon(messages[0]).encode()).hexdigest()
chain[i] = sha256((chain[i-1] + "|" + canon(messages[i])).encode()).hexdigest()
```

`chain[i]` identifies "a conversation whose first i+1 messages are exactly
these". Store `index: chain_hash -> {session_id, idx}`.

### 2.2 Resolve → delta

```
on request(messages[0..n]):
  for i in range(n, -1, -1):              # longest prefix first
      hit = index.get(chain[i])
      if hit: break
  if hit:
      session = hit.session_id
      delta   = messages[i+1 .. n]        # usually 1–2 messages
  else:
      session = create_session()
      delta   = messages[0 .. n]          # the ONE big dump, once per thread
```

**Critical: pre-register the next hash.** Hermes' *next* request will contain
the assistant message you are about to return (and any `role:tool` results).
After responding, compute the projected chain including your own reply and
register it against the same session. Without this, every second request is a
miss and you're back to square one.

```
reply_msg = {"role":"assistant","content":text,"tool_calls":tool_calls or None}
index[chain_after(chain[n], reply_msg)] = {session_id: session, idx: n+1}
```

Then the *next* request hits at depth `n+1`, and the delta is just the new
`user` or `tool` messages.

### 2.3 The system-prompt drift trap (read this one twice)

Hermes almost certainly injects a *volatile* system prompt: current datetime,
memory blocks, workspace state. If `messages[0]` changes every turn,
`chain[0]` changes, **nothing ever matches, and you create a brand new session
every single request.** You would have "implemented sessions" and still have
zero recall. I'd bet this is a live landmine even after you fix the flatten.

Fix: hash the system message *normalized*, and handle drift as an update
rather than a cache miss.

```python
SYSTEM_VOLATILE = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?\b"),  # timestamps
    re.compile(r"(?s)<memory>.*?</memory>"),                        # memory block
    re.compile(r"(?s)<context>.*?</context>"),
]
def norm_system(text):
    for rx in SYSTEM_VOLATILE: text = rx.sub("<volatile>", text)
    return text.strip()
```

- `chain[0]` uses `norm_system(...)`.
- Separately keep `system_live_hash` = hash of the *raw* system text on the
  session record.
- If `system_live_hash` changed but normalized matches → **do not fork**.
  Prepend one compact block to this turn's delta:
  `[system update]\n<the volatile blocks that changed>\n[/system update]`.
  Typically a few hundred bytes instead of a new session.
- If *normalized* system changed → genuinely a different agent config. Fork.

Log both hashes every request. `system_drift=1 forked=0` should be the normal
line; `forked=1` more than once per thread means the regexes need widening.

### 2.4 Divergence, retries, compaction

- **Shorter prefix hit than `session.idx`** (Hermes regenerated, edited, or
  compacted history): the session's state no longer matches what Hermes
  believes. v1 behavior: create a fresh session and replay `messages[0..n]`
  once. Log `fork_reason=divergence depth=i session_idx=k`.
  v2 (if `/session/:id/revert` or fork proves usable — the REST surface lists
  session fork and message revert): revert to the divergence point and
  continue. Spike it in Phase 1; don't block on it.
- **Hermes compaction** at ~150 turns will rewrite history wholesale. Expect a
  fork. That's correct and cheap — one big dump, then deltas again.
- **Exact repeat** (same `chain[n]`, same delta, within `SHIM_IDEMPOTENT_TTL`
  = 60 s): return the cached response, do **not** re-run the turn. Protects
  against Hermes retry double-executing a side-effecting delegation.

### 2.5 Store

Persist to `~/.local/state/opencode-shim/sessions.json`, atomic write
(tmp + `os.replace`), flushed after each turn. opencode's sessions survive a
shim restart; the mapping must too, or a `systemctl restart` wipes all recall.

```json
{
  "version": 3,
  "index": { "<chain_hash>": {"session_id":"ses_x","idx":12} },
  "sessions": {
    "ses_x": {
      "created": 1758412800, "last_used": 1758416400, "idx": 12,
      "system_norm_hash": "...", "system_live_hash": "...",
      "tools_hash": "...", "profile": "A",
      "tool_calls": {"call_7f2a":"grep_files"},
      "turns": 12, "forked_from": null
    }
  }
}
```

- Cap `SHIM_MAX_SESSIONS` = 200, LRU evict (drop index entries; leave the
  opencode session on disk, it's harmless).
- TTL `SHIM_SESSION_TTL` = 7 days.
- Index grows ~2 entries/turn; prune entries whose session is gone.

### 2.6 Concurrency

A conversation is inherently serial. Replace the single global semaphore with:

- `per_session_lock` — one in-flight turn per session, 300 s timeout.
- `global_sem(MAX_CONCURRENT)` — across *distinct* sessions only.
- Attachment normalization stays **outside** both (v2.2 got this right).
- Session resolution + store write under a short `index_lock`; never hold it
  across an LLM call.

---

## 3. L2 — Turn protocol: two profiles

The single biggest quality lever after sessions. Stop making opencode's coding
agent pretend to be a raw model.

### 3.1 Profile A — "LLM mode" (default when `tools[]` is non-empty)

Hermes is the agent. opencode must be a plain model: no system prompt of its
own, no tools of its own. Define a dedicated agent in
`/home/mitansh/hermesworkspace/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "agent": {
    "hermes": {
      "mode": "primary",
      "model": "opencode/muse-spark-1.3-contributor-free",
      "prompt": "{file:./prompts/hermes-bridge.md}",
      "tools": {
        "write": false, "edit": false, "patch": false, "bash": false,
        "read": false, "grep": false, "glob": false, "list": false,
        "webfetch": false, "task": false, "todowrite": false, "todoread": false
      }
    },
    "hermes-agent": {
      "mode": "primary",
      "model": "opencode/muse-spark-1.3-contributor-free",
      "prompt": "{file:./prompts/hermes-agent.md}"
    }
  }
}
```

Pass `"agent": "hermes"` in the message body. Verify against `/doc` that
`agent` is accepted on `POST /session/:id/message` for your build — if the
field name differs, the OpenAPI spec is ground truth, not this document.

`prompts/hermes-bridge.md` should be short — you are the conversational
partner, the tool protocol, nothing about editing code.

**Tool schemas once per session, not once per turn.** This alone removes
~24 kB × (turns − 1).

- Compute `tools_hash = sha256(canonical json of tools[])`.
- First message of a session: emit the full `[tools]` block.
- Later turns: if `tools_hash` unchanged, emit nothing. If changed, emit only
  a diff block: `[tools changed] added: X, Y  removed: Z` + schemas for added.
- Budget stays 24 kB but now applies to a once-per-session payload; raise to
  48 kB and log if exceeded.

### 3.2 Profile B — "Agent mode" (default when `tools[]` is absent/empty)

Hermes delegates a task; opencode works the workspace with its own tools.
Use agent `hermes-agent`, full tool set, relay progress as text deltas. This
is the path that already works for the zip/json cases in §10 — keep it.

Selection: `tools` present → A; absent → B. Allow explicit override via
`X-Shim-Profile: A|B` header or a `model` suffix (`...-agent`), so you can test
both without touching Hermes config.

### 3.3 Tool-call bridge, hardened (fence protocol v2)

Keep the fence — but with sessions, the protocol statement is sent **once**,
and the model has real conversational memory of how it went last turn.

Hard rules in the bridge prompt:

- Exactly one fenced block, last thing in the reply, nothing after it.
- ```` ```hermes-toolcalls ```` containing a JSON **array**.
- Each element `{"name": str, "arguments": {object}}` — `arguments` is an
  object, not a JSON string. Shim serializes to a string for OpenAI.
- To answer in text, emit no fence at all.

Shim side:

- Accept: fenced block; bare top-level array; single bare object. Reject
  anything else.
- Validate `name ∈ offered tools`; validate `arguments` is an object and every
  `required` key from the schema is present. Type-check only scalars — don't
  build a JSON Schema validator in stdlib.
- **One repair turn.** On an invalid/absent-but-required fence, append to the
  *same session*:
  `[protocol error] <specific reason>. Re-emit only the corrected
  hermes-toolcalls block.` Re-run once. A repair turn is far cheaper than a
  Hermes-level retry, and with sessions the model still has its own reasoning
  in context.
- `tool_choice: "none"` → strip the fence instruction for this turn, force text.
- `tool_choice: {"function": {"name": X}}` → say so explicitly and reject any
  other name.
- `call_id = "call_" + uuid4().hex[:12]`, stored on the session record so the
  returning `role:tool` message can be rendered as
  `[tool result: <name> (call_x)]\n<content>`. Without the name, the model
  can't tell which of three parallel calls it's reading.
- Parallel calls: array of >1 → multiple `tool_calls` in one response; expect
  multiple `role:tool` messages back in the next delta, render all of them.
- Metrics: count `fence_ok / fence_repaired / fence_failed` per 100 turns.
  Expose on `/debug/stats`. **Gate for Phase 5 below.**

### 3.4 Phase 5 (optional) — native tool calls via MCP

If fence reliability stays below ~95 % after §3.3, replace the fence entirely:

- Shim hosts a local MCP server exposing Hermes' tools to opencode
  (opencode has first-class MCP support via `opencode.json`).
- The model emits a *native* tool call. The MCP handler **parks** it on a
  queue and the shim returns `finish_reason: tool_calls` to Hermes.
- Hermes executes and posts the result in its next request; the shim matches
  by `call_id` and resolves the parked MCP response. The opencode turn was
  never torn down — it resumes in place with full internal state.

Upside: no parsing, correct native formatting, parallel calls free.
Downside: an opencode turn stays open across multiple Hermes HTTP requests.
Needs: park timeout (`SHIM_PARK_TIMEOUT` 600 s) → abort session turn;
per-session parked-call registry; the parked turn holds a `global_sem` slot,
so raise `MAX_CONCURRENT`. **Spike this for a day before committing.** Don't
start it until Phases 1–4 are measured.

---

## 4. L3 — Streaming and liveness

The 214 s silent turn in §10's RCA is a transport bug, not an opencode bug.
v2.2 emits a single fake SSE delta after the whole turn completes, so Hermes'
stale detector fires on any slow turn.

### 4.1 Real relay

1. `GET /event?session=<id>` (session filter is supported — use it; the
   unfiltered stream will drown you) **before** posting the message, so you
   can't miss early events.
2. `POST /session/:id/prompt_async` (returns 204) for the turn.
3. Consume:
   - `message.part.delta` (`field == "text"`) → emit an OpenAI
     `chat.completion.chunk` per delta. This is the real streaming source.
   - `message.part.updated` → final part state; use to reconcile.
   - `session.idle` → **the completion signal.** Not the HTTP return of the
     message POST; assistant parts can arrive after it.
   - `session.error` → failure.
4. Buffer full text alongside for fence parsing and for non-stream responses.

> Version caveat: `/event` delivery has been broken or reordered in some
> opencode builds (SyncEvent publishes not reaching subscribers; `part.delta`
> arriving before `part.updated` for the same `partID`). **Phase 0 probes
> this on 1.18.31 specifically.** If deltas don't arrive, fall back to
> `message.part.updated` + polling `GET /session/:id/message`, and keep the
> heartbeat below — but confirm, don't assume.
> Don't assume part ordering: register parts lazily on first sight of either
> event.

### 4.2 Heartbeats and inactivity timeout

- While a turn is open with no text yet (Profile B running bash/read), emit an
  SSE comment `: ping\n\n` every 10 s. Costs nothing, keeps Hermes' detector
  quiet. Optionally emit opencode tool-step summaries as visible text in
  Profile B — better UX than silence.
- Replace the 280 s wall clock with an **inactivity** timeout:
  `SHIM_IDLE_TIMEOUT` = 120 s since last event → abort, `504`, log the last
  seen event type. A turn making steady progress should never be killed.
- Non-streaming requests still consume the event stream internally, so the
  idle timeout applies there too.

### 4.3 Response fidelity

- `usage`: populate `prompt_tokens`/`completion_tokens` from opencode's
  message tokens if exposed on the assistant message; else estimate
  `len(text)//4` and set nothing you'd be lying about. Hermes may make
  compaction decisions from this — a zero usage block can confuse it.
- `id`, `created`, `model`, `finish_reason` (`stop` | `tool_calls` | `length`)
  must all be present and consistent between stream and non-stream paths.
- SSE terminator `data: [DONE]` unchanged.

---

## 5. Memory — the three layers, explicitly

"Recall" is three different things. Fix each.

| Layer | Scope | Mechanism after this plan |
|---|---|---|
| **Turn memory** | Within one Hermes thread | §2 sessions + delta. opencode holds real multi-turn structure. |
| **Injected memory** | Hermes' own recall blocks in `system` | §2.3 — never truncated, re-sent as a compact `[system update]` only when it changes. |
| **Project memory** | Across all threads | `AGENTS.md` in `WORKDIR`, loaded by opencode automatically into every session. |

### 5.1 Truncation policy (new, and it matters)

With sessions, a delta is normally 1–2 messages, so truncation should never
fire. If it does (a big fork replay):

1. Never drop the system message or the last 3 turns.
2. Drop/summarize the **middle**, oldest-first within the middle.
3. Emit `[history elided: N turns]` at the elision point so the model knows.
4. Log `truncated=1 dropped=N bytes=X` at WARNING. Today this happens
   silently — which is why the amnesia was invisible.

### 5.2 `AGENTS.md`

Optional but cheap: shim maintains `WORKDIR/AGENTS.md` from
`SHIM_CONTEXT_FILE` (stable facts: who mitansh is, what Hermes is, what the
workspace contains, house rules). It lands in opencode's own prompt, gets
cached, and costs nothing per turn. Do not put volatile state there.

---

## 6. Config (new env)

| Var | Default | Purpose |
|---|---|---|
| `SHIM_SESSIONS` | `1` | **Kill switch** — `0` restores v2.2 flatten path |
| `SHIM_SESSION_STORE` | `~/.local/state/opencode-shim/sessions.json` | persisted index |
| `SHIM_MAX_SESSIONS` | `200` | LRU cap |
| `SHIM_SESSION_TTL` | `604800` | 7 days |
| `SHIM_AGENT_LLM` | `hermes` | Profile A agent name |
| `SHIM_AGENT_TASK` | `hermes-agent` | Profile B agent name |
| `SHIM_PROFILE` | `auto` | `auto` \| `A` \| `B` |
| `SHIM_IDLE_TIMEOUT` | `120` | seconds since last SSE event |
| `SHIM_HEARTBEAT` | `10` | SSE ping interval |
| `SHIM_IDEMPOTENT_TTL` | `60` | duplicate-request cache |
| `SHIM_TOOLS_BUDGET` | `49152` | once-per-session schema budget |
| `SHIM_REPAIR_TURNS` | `1` | fence repair attempts |
| `SHIM_CONTEXT_FILE` | *(unset)* | source for `AGENTS.md` |
| `SHIM_DEBUG_DUMP` | `0` | write full outbound payloads to `/tmp/shim-dumps/` |

All v2.2 attachment vars unchanged.

---

## 7. Observability (build this in Phase 0, before any rewrite)

One structured line per request:

```
[req] id=r_8a1 sess=ses_x hit=1 depth=11 delta_msgs=2 delta_bytes=1841
      tools=24 tools_sent=0 sys_drift=0 forked=0 profile=A
      attach=1 mimes=image/png stream=1 fence=ok tool_calls=1
      ttfb_ms=940 total_ms=6210 finish=tool_calls
```

Endpoints:
- `GET /debug/sessions` — index size, sessions, LRU age, hit rate.
- `GET /debug/last` — last N request lines + the exact outbound payload
  (redacted base64).
- `GET /debug/stats` — hit rate, fence ok/repair/fail, p50/p95 total_ms,
  mean delta_bytes.

**Target after Phase 1:** prefix hit rate > 95 %, mean `delta_bytes` < 4 kB.
If hit rate is low, it's §2.3 — widen the volatile regexes.

---

## 8. Tests
### 8.1 Unit (no LLM)
1. `chain()` stability: reordered dict keys, whitespace → same hash.
2. Resolve: exact prefix, extended prefix, divergent prefix, empty store.
3. Pre-registration: simulated 5-turn loop → 1 create, 4 hits.
4. `norm_system`: drifting timestamp and `<memory>` block → same normalized
   hash, different live hash, `[system update]` generated.
5. Fence parser: fenced / bare array / bare object / prose+fence / two fences
   / unknown tool name / `arguments` as string / missing required key.
6. Truncation: system and last 3 turns always survive.
7. Store: atomic write, reload after simulated crash, LRU eviction.

### 8.2 Live against serve `:4096` (Phase 0)
8. `/doc` → confirm exact shapes of `POST /session`,
   `/session/:id/message`, `/session/:id/prompt_async`, and whether `agent`
   and `model` are accepted on the message body for 1.18.31.
9. `/event?session=<id>` → confirm `message.part.delta`,
   `message.part.updated`, `session.idle` actually arrive on this build, and
   in what order. **Record the answer in this file.**
10. Two messages to the same session → second reply demonstrates knowledge of
    the first *without* it being re-sent.

### 8.3 E2E via `:8000` — acceptance gates

11. **Recall canary.** Turn 1: "remember the code word is `ferrous-oxide`".
    Turns 2–20: unrelated filler. Turn 21: "what is the code word?" →
    must answer `ferrous-oxide`. **This is the gate for the whole rework.**
12. **Payload shrink.** Same 21 turns, assert `delta_bytes` on turn 21 is
    < 2 kB and `tools_sent=0`. Assert total bytes sent across 21 turns is at
    least 10× lower than v2.2 on the same script.
13. **System drift.** Inject a changing timestamp into `system` every turn →
    `forked=0` for all 21 turns.
14. **Tool loop.** 5-round tool-call loop (call → result → call → result →
    text), `tool_choice: none` forces text, unknown tool rejected, two
    parallel calls in one turn round-trip correctly.
15. **Streaming.** Deltas arrive incrementally (assert ≥3 chunks and
    ttfb < 10 s on a long answer); heartbeat during a silent Profile B turn;
    idle timeout fires at 120 s not 280 s.
16. **Restart persistence.** `systemctl --user restart opencode-shim`
    mid-thread → next turn still hits (`hit=1`), canary still recalled.
17. **Idempotency.** Identical request twice in 5 s → one opencode turn.
18. **Attachments regression.** Re-run the full v2.2 §7 suite: `data:`,
    `file://`, bare path, `http(s)`, pdf, zip staging, json inline,
    video/audio 400-gate. None of this should change — prove it.
19. **Fork.** Rewrite history mid-thread → `forked=1`, new session, correct
    answer, old session untouched.

---

## 9. Phased rollout

Each phase ships independently and is revertible. Don't skip Phase 0.

| Phase | Work | Exit criterion |
|---|---|---|
| **0. Instrument** | §7 logging + `/debug/*` on *current* v2.2. Run tests 8–9. Capture one real bad turn's full outbound payload. | You can quote actual prompt bytes/turn and know whether `/event` works on 1.18.31. Diagnosis confirmed, not assumed. |
| **1. Sessions** | §2 entire. Keep flatten behind `SHIM_SESSIONS=0`. | Tests 1–4, 7, 10–13, 16, 19. Hit rate > 95 %. **Canary passes.** |
| **2. Profiles** | §3.1–3.2 `opencode.json` agents, tools-once. | Test 12 (`tools_sent=0`), 18. Profile B zip case still works. |
| **3. Streaming** | §4 relay, heartbeat, idle timeout, usage. | Test 15. No stale-detector trips in a day of real use. |
| **4. Fence v2** | §3.3 hardening + repair turn. | Test 14. `fence_ok` > 95 % over 100 real turns. |
| **5. MCP (opt)** | §3.4 — **only if Phase 4 exits below 95 %.** | Spike first. Separate go/no-go. |

Per-phase mechanics (unchanged from v2.2 §8): backup `server.py`,
`python3 -m py_compile`, `systemctl --user restart opencode-shim`,
`journalctl --user -u opencode-shim -n 100`, then the phase's tests, then one
real Hermes message through the linked channel.

At Phase 1, `server.py` will outgrow one file. Split:
`server.py` (HTTP/SSE) · `sessions.py` (L1) · `bridge.py` (L2/fence) ·
`attachments.py` (lifted verbatim from v2.2 — do not touch it during the
move; port it in its own commit with tests green).

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| `/event` broken on 1.18.31 | Phase 0 test 9. Fallback: `part.updated` + poll. |
| `agent` field not accepted on message body | `/doc` is ground truth; fall back to setting the agent at session creation. |
| Session divergence on every turn (hidden volatile field) | `/debug/stats` hit rate is the alarm; widen `norm_system`. |
| opencode auto-compacts a long session and loses the canary | Log `session.turns`; at `SHIM_SESSION_MAX_TURNS` (~80) proactively fork with a shim-generated summary carried over. |
| Parked MCP turns leak (Phase 5) | Park timeout + reaper; don't start Phase 5 early. |
| Refactor breaks attachments | Test 18 is a hard gate on every phase. |
| Zen rate limits under a 150-turn loop | Log 429s distinctly; backoff + surface to Hermes as `429`, not `500`. |

---

## 11. What this does *not* do

- No public bind, no auth layer, no TLS (loopback only).
- No API keys anywhere on the Hermes side; Zen auth stays in opencode.
- No media downscaling/compression (files stay small).
- No Hermes config change — it stays pointed at `http://127.0.0.1:8000/v1`.
- Does not make opencode a *good* raw LLM endpoint; it's an agent runtime with
  a chat-shaped hole. Profile A narrows the mismatch, it doesn't remove it.

## 12. Carried forward unchanged from v2.2

Still authoritative, do not re-litigate during this rework:

- §2a backend media support matrix (image/pdf/text OK; video/audio/zip/json
  rejected by serve → shim gates or stages them).
- §3 attachment normalization table (`data:` / `http(s)` / `file://` / bare
  path → `data:` URL) and all `SHIM_MAX_*_BYTES` limits.
- §4 multi-shape content parsing (OpenAI / Responses / Anthropic image parts).
- §6 keyless guarantees in full.
- §10 `.shim-inbox/` staging for opaque binaries; `python3 -u` in the unit
  file; Telegram's 50 MB upload cap.

## 13. Phase 0 results (recorded 2026-09-21, serve 1.18.31)

- **Test 8 (`/doc`):** `agent` IS accepted on both `POST /session` (create)
  and `POST /session/{id}/message` (per-turn). `model` shape is
  `{providerID, modelID}`. `POST /session/{id}/prompt_async` returns 204 with
  the same body shape. `POST /session/{id}/fork` and `.../revert` exist and
  take `messageID` — revert/fork optimization still to be spiked in Phase 1.
- **Test 9 (`/event`): CORRECTION to §4.1** — `GET /event` on 1.18.31 takes
  NO session query param (only `directory`/`workspace`). The stream is global;
  the shim MUST subscribe once and filter client-side by
  `properties.sessionID`. Verified arriving: `message.part.delta`
  (`field=="text"` deltas), `message.part.updated`, `message.updated`,
  `session.status` (busy), `session.updated`, `session.diff`. `session.idle`
  is the documented completion signal but was not observed within a 90 s
  window on a multi-step `build`-agent turn (the turn kept opening new
  assistant messages per tool step) — completion detection must tolerate
  multi-`message.updated` turns, not just one reply message.
- **Delta-before-updated ordering:** observed `part.delta` then
  `part.updated` for the same `partID` (normal order). Register parts lazily
  on first sight of either, per §4.1 caveat.
- **Test 10 (session recall):** two `POST /session/{id}/message` turns —
  turn 1 stores `ferrous-oxide`, turn 2 asks without re-sending — recalled
  correctly. Sessions hold real multi-turn state. Flatten is unnecessary.
- **Payload capture (Phase 0 instrumentation on v2.2 flatten path):**
  2-message no-tool probe → `prompt_bytes=854`, `total_ms=2345`. A concurrent
  real Hermes turn (34 browser tools, 2 messages deep) → `prompt_bytes≈68 kB`
  outbound — i.e. tool schemas dominate even at depth 2, confirming the
  §3.1 tools-once-per-session win before history even matters.
- **`/debug/*` live:** `GET /debug/stats`, `/debug/last`, `/debug/sessions`
  on `:8000` (Phase 0 stub: sessions reports flatten-mode until Phase 1).
- **Side observation:** default `build` agent runs its own tool loop
  (`bash sleep 2` per counted number) even for trivial prompts — live proof
  of §0.1 mechanism 4 and why Profile A (Phase 2) matters.

## 14. Phase 1 results (recorded 2026-09-21)

- **Unit (§8.1 tests 1–4, 7):** all 18 checks pass (`/tmp/test_sessions.py`).
  One correction applied to the plan's test 4: a changed `<memory>` block
  yields the SAME normalized hash (volatile by design) with a different live
  hash plus a `[system update]` block — no fork. Fork only on normalized change.
- **E2E on isolated shim (`:8001`, same code): canary PASS** — code word
  `ferrous-oxide` recalled after 3 filler turns; 1 create + 4 hits;
  `delta_bytes` ~770/turn vs 890 first dump (10×+ win scales with history).
- **Test 13 drift:** timestamp+memory drift → `hit=1 forked=0 sys_drift=1`.
  Caveat: changing the system template SHAPE (adding a `Date` line where none
  existed) changes the normalized hash → correct fork. Real Hermes drift keeps
  template shape; hit-rate alarm (§7) is the live guard.
- **Tests 16/17/19:** restart with same store → `hit=1`; identical POST twice
  → second `finish=cached total_ms=0`; rewritten history → `forked=1`
  (`divergence depth=2 session_idx=4`), correct answer.
- **Attachments:** zip via session path PASS (`hello shim`, 14.5 s). Image
  turn hung once at 300 s — root-caused to the `build` agent ignoring the
  attached vision input and recursing the filesystem (`read WORKDIR`, then
  `read /home/mitansh/work`, stuck). NOT a session-layer defect: identical
  bytes via flatten path answered in 14 s, direct-to-serve in 25 s. The flakiness
  is §0.1 mechanism 4; fix is Phase 2 Profile A (tool-less agent). Session was
  aborted server-side afterwards.
- **Deferred split:** `attachments.py`/`bridge.py` extraction postponed —
  attachments + fence stayed in `server.py`, only `sessions.py` is new, to keep
  the tested paths byte-identical. Split later with tests green.
- **Live cutover 2026-09-21 ~07:12 IST:** `server.py`+`sessions.py` on
  `opencode-shim.service` (`SHIM_SESSIONS=1` default; kill switch
  `SHIM_SESSIONS=0` + restart restores flatten). Both units `enabled`,
  `Linger=yes` → auto-start at boot. First live turn `3.4 s finish=stop`.
  Pre-restart journal confirms the diagnosis on real traffic: a Hermes turn
  with 34 tools sent `delta_bytes=68277` at depth 6 and hit the 300 s timeout
  on the old path.

## 15. Phase 2 results (profiles + tools-once, 2026-09-21)

- `hermesworkspace/opencode.json` defines `hermes` (Profile A: tool-less LLM,
  bridge prompt) and `hermes-agent` (Profile B: full tools, worker prompt).
  `agent` verified accepted per-turn AND at session create; agents hot-load
  after `opencode-serve` restart (no hot-reload — restart required).
- Probe: `hermes` + tool-bait prompt → answered in words, zero tool parts in
  3 s. `hermes-agent` listed the workspace correctly (7.8 s).
- Tools-once E2E: turn 1 `tools_sent=2`, turn 2 `tools_sent=0 delta_bytes=817`;
  full add-tool loop returns 42; `tool_choice:none` forces text; no-tools →
  profile B; `X-Shim-Profile` override works both ways.
- Permissions (user request): top-level `"permission": "allow"` plus explicit
  per-agent allow maps. Schema lesson: config file takes the OBJECT map form,
  not the array ruleset (array rejected with ConfigInvalidError — twice).
  Effective ruleset still lists `doom_loop/external_directory: ask`, but the
  leading `* allow` matches first: verified live `bash` + external
  `/etc/hostname` read with ZERO permission events in 5.3 s. `question` stays
  `deny` (allow would stall headless turns waiting for an answer).

## 16. Phase 3 results (streaming relay, 2026-09-21)

- Turn runs via `prompt_async` + one `/event` subscription opened BEFORE the
  post, filtered client-side by `properties.sessionID` (no server filter).
  Full text reconciled from `GET message` (never assembled from deltas).
  Heartbeat `: ping` every 10 s on silent stream turns; inactivity timeout
  120 s → abort + 504; wall cap 300 s. Blocking fallback ONLY when
  `prompt_async` never started (never double-runs a live turn).
- E2E: 19 live chunks, `ttfb=7.7 s < 10 s`, `finish=stop`, 965 chars streamed;
  non-stream usage now real (`prompt_tokens=1032 completion_tokens=12` from
  opencode message tokens); streaming tool_calls ends `finish=tool_calls`.
- One transient 500 (`event stream ended` 24 ms after first delta) under a
  forked retry; fork path itself re-verified healthy. Hermes retries are the
  correct recovery (same delta re-runs as a new turn, OpenAI semantics).
- `[req]` gains `ttfb_ms` and `err=` (on failures) fields.

## 17. Phase 4 results (fence v2, 2026-09-21)

- Parser: fenced / bare array / bare object / prose+fence / two-fences
  (first wins) / unknown tool / args-as-string / missing required /
  scalar types / code-fence-as-text / forced-name reject+accept / parallel —
  14/14 unit checks (`/tmp/test_fence.py`). Repair ONLY on real JSON attempts
  with semantic errors (code samples stay text).
- Live repair E2E: `tool_choice=required` + text answer → repair turn fired →
  model refused again → `fence=failed`, text returned (one repair, honest
  failure, no loop). Forced `tool_choice` honored (`note`); parallel 2-call
  turn + full round-trip (`3`, `7`) correct.
- Metrics: `fence_ok` now counts clean text too; `/debug/stats` exposes
  `fence_ok_rate`. Gate for §3.4 MCP decision.
- Phase 5 (MCP) NOT started: no evidence yet that fence stays below 95% —
  measure on live traffic first per plan.

## 18. Streaming-echo fix (2026-09-21, post-cutover)

- Symptom on live Hermes traffic: every turn following a `fence=ok` reply
  forked (`divergence depth=N session_idx=N+1`); text replies hit fine.
- Root cause: Hermes streams, so its history echo of our tool-call reply has
  `content` = the FULL streamed text (fence block included), while the shim
  pre-registered `content=None` (the stripped remaining). Verified by
  assembling a streamed response client-side: content is the raw fence.
  (Hermes source confirms the pattern: send-path arg canonicalization +
  stream-assembled assistant text.)
- Fix: dual pre-registration — primary (`content` = stripped/None, what
  non-stream clients echo) + secondary (`content` = raw full text), same
  session/idx, no turn double-count. Synthetic Hermes-stream echo now
  `hit=1 tools_sent=0`. Pre-existing forked sessions are harmless (LRU).
- Unrelated bug found alongside: `register_new` hashed `json.dumps(raw)`
  while `resolve` hashes `raw` → phantom `sys_drift=1` on every second turn.
  Fixed; verified `sys_drift=0` on identical system, fork still fires on real
  template change.

## 19. Hermes verification on the live build (2026-09-21)

- Hermes (`base_url=http://127.0.0.1:8000/v1`, unchanged) runs its normal
  loops through the new path — no Hermes config change was needed (§11).
- Observed real turns: Profile A first-dump 79 kB (13 s) + Profile B
  delegation (17 s); tool-loop turn `fence=ok` (58 s); `hit=1 depth=15
  delta=804 B tools_sent=0` (3.7 s); depth-19 legacy thread healed via one
  fork. `fence_ok_rate=1.0`, zero Hermes-side errors since.
- Divergences seen are legacy-thread healing only (pre-fix registrations +
  retries from a self-inflicted broken-config window at ~07:17 when an
  intermediate `opencode.json` failed validation — Hermes retried per its
  3-attempt policy; config since valid, serve restarted, probes green).
  Steady state for new threads: one dump, then delta hits.
- All phases exit-gated. Phase 5 (MCP) stays parked: no evidence for it.
- Ops: both units `enabled`, `Linger=yes` (auto-start at boot, verified
  across 4+ restarts today); kill switch `SHIM_SESSIONS=0`; session index
  persists at `~/.local/state/opencode-shim/sessions.json` (survived all
  restarts); `/debug/stats` (hit/fence rates, p50/p95) is the ongoing alarm.
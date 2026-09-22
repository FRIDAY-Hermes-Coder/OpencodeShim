# opencode-hermes-shim — OpenAI Chat Completions compat gap analysis

Date: 2026-09-22. Based on direct review of `server.py` (2544 lines, commit
`3586b00`+) and `sessions.py`, against OpenAI's current Chat Completions API
(`POST /v1/chat/completions`, `developers.openai.com/api/reference`). This is
a single-provider, keyless, loopback shim — the goal isn't "be OpenAI," it's
"never surprise a client that expects OpenAI shape." Every gap below is
framed that way: what breaks, for whom, and how cheaply it can close.

Method: every documented request parameter, message/content-part shape, and
response field was checked against an actual `grep`/`view` of the code, not
assumed. Where the fix depends on something only opencode's own API can
answer (does `/session/:id/message` accept sampling params at all?), that's
flagged explicitly rather than guessed at.

---

## The shape of the gap

Three categories, because they need different treatment:

1. **Silently wrong** — the client sends a real parameter, the shim ignores
   it, and the response looks completely valid. This is the dangerous
   category: nothing errors, so nothing gets noticed until someone's `n=3`
   quietly returns one choice, or a `stop` sequence is sent and never
   honored. **Fix these first.**
2. **Missing but honest** — a capability opencode/Zen likely can't provide
   at all (`logprobs`, `seed`, `logit_bias`). The right fix usually isn't
   "implement it," it's "reject it loudly" instead of ignoring it, so a
   client relying on it fails fast instead of silently getting wrong data.
3. **Cosmetic/metadata** — fields OpenAI clients send that a personal
   loopback shim has no reason to act on (`user`, `store`, `service_tier`,
   `metadata`, `prompt_cache_key`). Already handled correctly today by
   virtue of being ignored — unknown JSON fields are harmless. Not gaps.

---

## A. Request parameters — full pass

Confirmed by `grep -n "\"n\"\|max_tokens\|temperature\|top_p"` returning
**zero matches** anywhere in `server.py`. The entire body-parsing surface,
today, is:

```python
messages = body.get("messages", [])
tools = body.get("tools")
tool_choice = body.get("tool_choice")
stream = bool(body.get("stream"))
req_model = body.get("model") or MODEL_ID
```

Everything else OpenAI defines is invisible to the shim. Full table:

| Param | Category | Current state | Fix |
|---|---|---|---|
| `stop` | **Silently wrong** | Ignored entirely | Client-side post-truncation (§A.1) |
| `max_tokens` / `max_completion_tokens` | **Silently wrong** | Ignored; no `finish_reason:"length"` ever possible | Client-side cap + `length` (§A.2) |
| `n` | **Silently wrong** | Ignored; always returns exactly 1 choice regardless of value | Reject `n>1` explicitly (§A.3) — don't try to fake it |
| `parallel_tool_calls: false` | **Silently wrong** | Ignored; fence bridge can already return multiple calls | Truncate to first call when `false` (§A.4) — cheap |
| `temperature`, `top_p` | Needs upstream check | Ignored | Research opencode's message-body schema first (§A.5) |
| `response_format` | Missing but honest→could fill | Ignored | Prompt-enforced JSON via existing repair-turn machinery (§A.6) |
| `logprobs`, `top_logprobs` | Missing but honest | Ignored | Reject with clear 400 if requested (§A.7) |
| `seed` | Missing but honest | Ignored | Reject with clear 400 if requested (§A.7) |
| `logit_bias` | Missing but honest | Ignored | Reject with clear 400 if requested (§A.7) |
| `presence_penalty`, `frequency_penalty` | Needs upstream check | Ignored | Same bucket as temperature/top_p (§A.5) |
| `functions` / `function_call` (deprecated) | Missing, low priority | Not read at all — only `tools`/`tool_choice` | Alias `functions`→`tools` if `tools` absent (§A.8) |
| `stream_options.include_usage` | Missing | Ignored; no final usage-only chunk ever sent | Add terminal usage chunk (§A.9) |
| `user`, `store`, `metadata`, `service_tier`, `prompt_cache_key`, `safety_identifier` | Cosmetic | Ignored | Correct as-is — no action |

### A.1 `stop` — client-side truncation (cheap, do first)

opencode/Zen almost certainly has no decode-time stop-string support exposed
through the shim's request shape, but the shim doesn't need one — it already
has the full reconciled text before it ever reaches the client (the same
`out`/`final_text` used by the Bug-A catch-up path). Truncate there:

```python
def _apply_stop(text, stop):
    if not stop or not text:
        return text, False
    seqs = [stop] if isinstance(stop, str) else list(stop or [])
    cut_at = None
    for s in seqs:
        if s:
            i = text.find(s)
            if i != -1 and (cut_at is None or i < cut_at):
                cut_at = i
    if cut_at is None:
        return text, False
    return text[:cut_at], True
```

Apply once, right before `chat_completion_response`/`chat_completion_tool_response`
are built (both the streaming and non-streaming paths funnel through the same
`out` variable, so this is a single call site each). If a stop sequence
fires, `finish_reason` becomes `"stop"` — which, conveniently, is already
the hardcoded default, so no other change needed there. For the streaming
path, since the whole reconciled text arrives via the catch-up-emit
mechanism (Bug A's fix), truncate *before* calling `_emit_live`, not after —
otherwise the stop sequence itself gets streamed to the client first.

### A.2 `max_tokens` / `max_completion_tokens` — cap + honest `finish_reason`

`finish_reason` in this codebase is hardcoded to `"stop"` or `"tool_calls"` —
grep confirms `"length"` never appears as a value anywhere. That's a real
gap: a client that sends `max_completion_tokens: 200` to keep costs bounded
gets a response that may run arbitrarily long, silently.

Since opencode does the actual generation (this shim can't interrupt
mid-token the way a real inference server can), the honest implementation is
post-hoc: truncate the reconciled text to the token budget and set
`finish_reason: "length"` when truncation actually happened.

```python
def _apply_max_tokens(text, limit, chars_per_token=4):
    if not limit or not text:
        return text, False
    budget = int(limit) * chars_per_token
    if len(text) <= budget:
        return text, False
    return text[:budget], True
```

This is an estimate (same `~4 chars/token` heuristic already used for the
Bug-E usage fallback — consistent with existing precedent in this codebase),
not an exact token cut. Good enough to bound runaway output and tell the
client the truth about why it stopped; not good enough to bill against.
Document that distinction wherever usage/limits are surfaced.

### A.3 `n` — reject, don't fake

Silently returning 1 choice when `n=3` was requested is worse than erroring:
a client written against real OpenAI semantics (`response.choices[2]`) gets
an `IndexError` deep in its own code with no signal from the shim about why.
opencode's session model produces one assistant turn per message — there's
no cheap way to actually produce `n` independent samples without running the
turn `n` times (real cost, real latency, and each run would need its own
session-state handling since they'd diverge). Given this shim's one real
client is Hermes, and Hermes has no apparent use for `n>1`:

```python
n = body.get("n", 1)
if isinstance(n, int) and n > 1:
    self._json(400, {"error": {"message": "n>1 is not supported by this shim (opencode serve produces one completion per turn)",
                                "type": "invalid_request_error", "param": "n", "code": "unsupported_parameter"}})
    return
```

Fail loudly at the top of `do_POST`, before any opencode call — cheapest
possible rejection point.

### A.4 `parallel_tool_calls: false`

The fence bridge already supports multiple calls per turn (an array in the
`hermes-toolcalls` fence). If the client explicitly opts out of parallel
calls, honor it by keeping only the first parsed call:

```python
if body.get("parallel_tool_calls") is False and calls and len(calls) > 1:
    calls = calls[:1]
```

One line, at the point `calls` is finalized after `parse_fence`.

### A.5 `temperature` / `top_p` / `presence_penalty` / `frequency_penalty` — needs upstream research first

These control sampling at the model layer. Whether they're fillable at all
depends entirely on whether `opencode serve`'s `POST /session/:id/message`
body accepts sampling overrides, or whether sampling is fixed per-agent in
`opencode.json` and not overridable per-message. **This needs one probe
against the live `/doc` OpenAPI spec before writing any code** — don't guess
the field name. If opencode does expose it (plausibly under the same
`model: {providerID, modelID, ...}` object the shim already builds), thread
it through the same way `model_override` was threaded through in the
fallback-chain work. If it doesn't, these join `logprobs`/`seed`/`logit_bias`
in the "reject don't ignore" bucket (§A.7) rather than being silently eaten.

### A.6 `response_format` — this one's worth actually building

`{"type": "json_object"}` and `{"type": "json_schema", "json_schema": {...}}`
are common enough (structured output for downstream parsing) that silently
ignoring it is a real functional gap, and this codebase already has the
exact machinery needed to fill it: the same prompt-injection + parse +
repair-turn pattern used for the `hermes-toolcalls` fence.

- Inject an instruction when `response_format` is present: for `json_object`,
  "respond with a single valid JSON object and nothing else"; for
  `json_schema`, include the schema and instruct strict conformance —
  reuse the existing tool-schema-budgeting logic (`SHIM_TOOLS_BUDGET`
  pattern) rather than inventing a new budget.
- After reconciliation, attempt `json.loads(out)`. On success, ship as-is.
- On failure, run **one repair turn** — the exact same mechanism already
  built for Bug D's empty-completion guard: `"[system] Your last response
  was not valid JSON. Reply with only the corrected JSON, nothing else."`
- If the repair also fails to parse, this is a genuine unsatisfiable
  request for this model — return `502` with a clear message, same pattern
  as the Bug-D exhausted-nudge failure, rather than shipping malformed JSON
  as a "successful" 200.

This is the single highest-value gap to close in this whole document: it's
a real, commonly-used feature, and the codebase's existing repair-turn
infrastructure makes it cheap.

### A.7 `logprobs` / `top_logprobs` / `seed` / `logit_bias` — reject, don't ignore

None of these are things opencode/Zen plausibly exposes through this
shim's abstraction layer (token-level logprobs and logit biasing require
direct access to the model's sampling step, which opencode's session
abstraction doesn't surface). Silently ignoring `logprobs: true` means a
client parses `choice.logprobs` expecting a real object and gets `None` —
a `NoneType` error in their code, unexplained. One check at the top of
`do_POST`:

```python
UNSUPPORTED_PARAMS = {"logprobs": None, "top_logprobs": None, "seed": None, "logit_bias": None}
for p in UNSUPPORTED_PARAMS:
    if body.get(p) is not None:
        self._json(400, {"error": {"message": f"'{p}' is not supported by this shim (opencode/Zen backend does not expose it)",
                                    "type": "invalid_request_error", "param": p, "code": "unsupported_parameter"}})
        return
```

Cheap, and turns a silent correctness bug into an immediate, actionable
error for anyone (including future-you) who tries to use one of these.

### A.8 Legacy `functions`/`function_call` — low priority, cheap

Pre-2023 OpenAI clients (and some SDKs' legacy code paths) still send
`functions=[...]` + `function_call` instead of `tools`/`tool_choice`. Given
Hermes is the only real client and almost certainly uses the modern shape,
this is low priority — but a one-line alias costs nothing:

```python
if not tools and body.get("functions"):
    tools = [{"type": "function", "function": f} for f in body["functions"]]
    tool_choice = tool_choice or body.get("function_call")
```

### A.9 `stream_options: {"include_usage": true}`

Real OpenAI streaming sends a final chunk with `choices: []` and a populated
`usage` object, after the content chunks and before `[DONE]`. Currently
absent — the terminal chunk always has empty `choices[0].delta` and no
`usage` key at all. Since usage (real or Bug-E's estimate) is already
computed by the time the terminal chunk is sent, this is one more chunk:

```python
if body.get("stream_options", {}).get("include_usage") and stream_live:
    usage_chunk = {"id": cid, "object": "chat.completion.chunk", "created": created,
                   "model": req_model, "choices": [], "usage": relay_meta.get("usage")}
    handler.wfile.write(f"data: {json.dumps(usage_chunk)}\n\n".encode())
```

Right before the existing `data: [DONE]` write, in every SSE terminal branch
(there are a few — the plain-text, tool-calls, and cached-response paths all
have their own).

---

## B. Message / content-part parsing — one real bug found

`extract_text_and_images` (line 1459) is genuinely thorough — it already
handles `image_url`, `input_image` (Responses API), Anthropic-style
`{"type":"image","source":{...}}`, `video_url`/`video`, `file`/`document`/
`pdf` with both `file_data` and raw `data`, and several fallback shapes.
One real gap and two low-priority ones:

### B.1 `input_audio` content parts are silently dropped (real gap)

OpenAI's actual schema for audio input is:

```json
{"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}
```

The code's audio branch (line 1530-1537) checks `t in ("audio_url", "audio")`
— it never checks `t == "input_audio"`, and even if it matched by accident
via a fallback branch, it looks for `p.get("audio_url", p.get("url", ...))`,
never `p.get("input_audio", {}).get("data")`. A client sending genuine
OpenAI-shaped audio input gets it dropped on the floor — no error, no
attachment, just silently gone (falls through every `elif`, matches
nothing, contributes nothing to `texts` or `atts`).

```python
elif t == "input_audio":
    ia = p.get("input_audio") or {}
    b64 = ia.get("data")
    fmt = ia.get("format") or "wav"
    if isinstance(b64, str) and b64:
        mime = f"audio/{fmt}"
        url = b64 if b64.startswith("data:") else f"data:{mime};base64,{b64}"
        atts.append((mime, url, None))
```

This lands in the existing audio pipeline (§2a's video/audio 400-gate from
the original plan still applies — audio still isn't forwarded to serve by
default — but at least it'll now hit that gate with a clear message instead
of vanishing silently before ever reaching it).

### B.2 `developer` role isn't recognized as "the system message" (low priority, Hermes-specific)

`sessions.py`'s system-drift immunity (`norm_system`, `system_live_hash`)
keys off `messages[0].get("role") == "system"`. o1-and-newer-style clients
send `role: "developer"` instead. If Hermes ever switches conventions, the
whole system-drift/carryover mechanism silently stops applying to the first
message, and volatile content in it (timestamps, memory blocks) would cause
needless forks every turn — exactly the landmine flagged in the original
session-design plan. Cheap fix, low urgency since Hermes controls its own
client and presumably isn't switching conventions unprompted:

```python
if messages and isinstance(messages[0], dict) and messages[0].get("role") in ("system", "developer"):
```

wherever that check currently lives in `sessions.py`.

### B.3 `refusal` content parts are silently dropped (cosmetic, near-impossible in practice)

An assistant message's `content` array can include
`{"type": "refusal", "refusal": "..."}`. The parsing loop has no branch for
`type == "refusal"` and it has no `"text"` key, so it falls through every
`elif` untouched. In practice this can only appear if a client echoes back
a previous *OpenAI* refusal into history — since this shim never emits
`refusal` itself, and Hermes' own history is built from this shim's
responses, the scenario basically can't arise here. Documenting it, not
prioritizing it.

### B.4 `file_id` (OpenAI Files API reference) has no handling

`{"type": "file", "file": {"file_id": "file-abc123"}}` — a reference to a
previously-uploaded file, no inline data. The code checks `file_data`/`data`
but never `file_id`, so this pattern is silently dropped rather than
rejected. Since this shim has no Files API backing it, there's no way to
*resolve* the reference — but silently dropping it is worse than saying so:

```python
if not url and isinstance(fobj, dict) and fobj.get("file_id"):
    print(f"[attach] file_id reference not resolvable by this shim: {fobj['file_id']}")
    # fall through with no attachment; consider surfacing to the client
    # via a 400 if this ever shows up in real traffic
```

Low priority unless it actually shows up in logs — Hermes presumably always
sends inline data given it manages its own attachment storage.

---

## C. Response object shape

| Field | Current state | Gap |
|---|---|---|
| `id`, `object`, `created`, `model` | Present, correct shape | None |
| `choices[].index` | Always `0` | Correct — matches §A.3's "no n>1" decision |
| `choices[].message.content` | Present | None |
| `choices[].message.tool_calls` | Present when applicable | None |
| `choices[].finish_reason` | Only ever `"stop"` / `"tool_calls"` | Add `"length"` (§A.2); `"content_filter"` is out of scope — no moderation layer to trigger it, correctly never emitted |
| `choices[].logprobs` | Field absent entirely | Should be explicit `null` when not requested, not just missing — some strict JSON-schema-validating clients treat a missing key differently from a null value. One-line addition to both response builders. |
| `usage.prompt_tokens/completion_tokens/total_tokens` | Present (real or Bug-E estimate) | None functionally; see below for detail fields |
| `usage.prompt_tokens_details` / `completion_tokens_details` | Absent | Cosmetic — these carry cached-token and reasoning-token breakdowns that don't apply to this backend. Not worth adding; a client that reads them optionally will just see them missing, which is valid. |
| `system_fingerprint` | Absent | Cosmetic for a single-backend shim — add only if a client hard-requires the key to exist; cheap to add as a static string (e.g. `"shim-<git-sha>"`) if it ever matters. |

The `choices[].logprobs: null` addition is the only item here worth actually
doing — it's a one-line change to both `chat_completion_response` and
`chat_completion_tool_response`, and it removes a class of "missing key"
bugs in stricter client-side deserializers (e.g. Pydantic models with
`logprobs: Optional[...]` but no `default=None` would choke on a genuinely
absent key depending on config).

---

## D. Streaming / SSE shape

This is the area the last several rounds of work already hardened
significantly (role-delta on open, real heartbeat chunks, part-boundary
separators, catch-up-emit, fence-safe non-streaming for tool turns). What's
still missing relative to real OpenAI streaming:

- **`stream_options.include_usage`** — covered in §A.9.
- **`include_obfuscation`** — real OpenAI adds small random padding to
  streaming deltas as a side-channel-timing mitigation. Purely a security
  hardening detail for OpenAI's own infrastructure; irrelevant on a
  loopback-only shim with no network-observable timing surface. Correctly
  not worth implementing.
- **Chunk `id` stability** — already consistent (`cid` computed once,
  reused across all chunks in a turn). No gap.

Streaming is in genuinely good shape at this point; the remaining item
(§A.9) is metadata, not structural.

---

## E. Error response shape

Confirmed via grep: every error in this codebase uses one of exactly four
`type` strings — `invalid_request`, `not_found`, `backend_error`,
`rate_limit` — and **no error anywhere includes `param` or `code`** (zero
matches in the whole file). Real OpenAI's error `type` values are
`invalid_request_error`, `authentication_error`, `permission_error`,
`not_found_error`, `rate_limit_error`, and `api_error` (5xx) — note the
`_error` suffix on every one, which this shim omits throughout.

**How much this matters depends on what actually branches on it.** Most
OpenAI-SDK-derived clients pick their exception class primarily from the
**HTTP status code**, which this shim already gets right (400/404/429/
500/502/504 are all used correctly per the grep above) — so this isn't a
"requests silently fail differently" bug the way the request-parameter gaps
are. It matters if Hermes (or any client) does its own string-matching on
`error.type`/`error.code` for retry/backoff logic, which — given the
sophistication of Hermes' own retry/backoff behavior seen throughout this
project — is plausible enough to be worth the trivial cost of aligning:

```python
_TYPE_MAP = {"invalid_request": "invalid_request_error",
             "not_found": "not_found_error",
             "backend_error": "api_error",
             "rate_limit": "rate_limit_error"}
```

Rename at the four call sites (or wrap `_json`'s error-shaping in one small
helper that does the rename centrally — cheaper to maintain than four
find-and-replaces if a fifth type string ever gets added). Add `param` on
every 400 that's about a specific field (all of §A's new rejections already
include it in the sample code above) and `code` as a short machine-readable
slug (`"unsupported_parameter"`, `"context_length_exceeded"` if that ever
becomes detectable, etc.) — both are optional per the OpenAI spec but cheap
to include and strictly additive for any client that reads them.

---

## F. Model listing endpoint

`GET /v1/models`, `GET /v1/models/{id}` are implemented (line 2215-2240) and
reasonably complete — object type, id lookup by short or full name, 404 on
unknown. Compared to the real endpoint:

- Missing `owned_by` on the list endpoint (present on the single-model
  retrieve at line 2238, absent from the list-building in `_ranked_models()`
  — worth checking that function directly, wasn't in the reviewed range).
- No `permission`/`root`/`parent` fields — these are legacy OpenAI
  fine-tuning-era fields most modern clients never read. Not worth adding.

Low priority relative to everything above — the models endpoint is mostly
used for discovery/dropdowns, not hot-path correctness.

---

## Prioritized rollout

Ordered by (impact if left broken) × (cost to fix), cheapest-and-most-
dangerous first:

| Phase | Items | Why this order |
|---|---|---|
| **1** | §A.3 (`n` rejection), §A.7 (`logprobs`/`seed`/`logit_bias` rejection), §E (error type alignment) | Pure guard-rail additions at the top of `do_POST` — no interaction with the relay/session/streaming machinery, essentially zero regression risk, and they convert several silent-wrongness bugs into loud, correct ones immediately. |
| **2** | §A.1 (`stop`), §A.4 (`parallel_tool_calls`) | Small, self-contained, touch the already-well-tested "finalize `out`/`calls`" section right before response-building — same section already hardened across Bugs A/D/E. |
| **3** | §A.2 (`max_tokens`/`length`) | Same section as Phase 2, slightly more involved since it needs a token-estimate pass (reuse Bug E's `//4` heuristic — don't invent a second one). |
| **4** | §A.9 (`stream_options.include_usage`) | Touches every SSE terminal branch (three of them) — more surface area than Phases 1-3, still low risk since it's purely additive (one more chunk before `[DONE]`). |
| **5** | §B.1 (`input_audio` parsing) | Isolated to `extract_text_and_images`, no interaction with anything else — safe any time, just not urgent since it currently only matters if a client sends real OpenAI-shaped audio, which Hermes may not do today. |
| **6** | §A.6 (`response_format`) | The biggest single addition — reuses existing repair-turn infrastructure but touches the request-parsing, prompt-building, *and* response-validation paths. Do this once Phases 1-5 are stable and tested, not alongside them. |
| **7** | §A.5 (temperature/top_p/penalties) | Blocked on external research — spike opencode's `/doc` OpenAPI spec first to find out if these are even fillable before writing any shim code. Don't schedule real work here until that answer exists. |
| **Skip** | §A.9's `include_obfuscation`, `usage.*_details`, `system_fingerprint`, `functions` legacy alias unless it shows up in real logs | Confirmed cosmetic or effectively unreachable given Hermes is the only real client. Revisit only if a second client ever talks to this shim. |

### Tests to add per phase

- **Phase 1**: three requests with `n:2`, `logprobs:true`, `seed:42`
  respectively → each gets a `400` with the right `param`/`code`, and
  critically, **no opencode call is made** (verify via a call-count mock —
  these should fail before `_do_session_turn` is ever reached).
- **Phase 2**: a turn where the model's natural output contains a
  client-supplied stop string mid-sentence → response truncates exactly at
  the first occurrence, `finish_reason:"stop"`. A tool-call turn with
  `parallel_tool_calls:false` and a fence containing 2 calls → response has
  exactly 1.
- **Phase 3**: a turn with `max_completion_tokens:10` against a normally
  long response → truncated to roughly the token budget, `finish_reason:
  "length"`, and confirm a turn that *doesn't* exceed the budget still
  reports `finish_reason:"stop"` (no false positives).
- **Phase 4**: `stream:true, stream_options:{include_usage:true}` → assert
  the chunk immediately before `[DONE]` has `choices:[]` and a populated
  `usage`, and every earlier chunk has no `usage` key (or `null`, per spec —
  confirm which the current implementation does and be consistent).
- **Phase 5**: a message with a genuine `{"type":"input_audio",...}` part →
  confirm it now reaches the existing audio 400-gate with a clear message,
  where today it silently vanishes with no attachment and no error.
- **Phase 6**: `response_format:{"type":"json_object"}` against a prompt
  where the model would naturally answer in prose → confirm the repair-turn
  fires, and the final response parses as valid JSON. A schema-conformance
  test if `json_schema` mode is implemented, not just `json_object`.

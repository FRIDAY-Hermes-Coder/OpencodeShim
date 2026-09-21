# opencode-hermes-shim — v4: streaming & empty-response fixes

Date: 2026-09-21. Based on direct review of the deployed `server.py`
(`server_version = "OpencodeShim/2.6-phase4"`, `SHIM_SESSIONS=1`, so every
request runs through `_do_session_turn`). Supersedes nothing in v3 — this is
a bug-fix pass on top of it. All five bugs below are confirmed from the code
itself, with exact line references, not inferred from symptoms alone.

Your reports are **five separate bugs**, and they're all inside
`_do_session_turn` / `_relay_turn`:

| Symptom | Bug |
|---|---|
| Narration sentences glued together with no spaces | **B** — no separator between opencode "parts" |
| "waiting for stream response (90s, first_chunk)" | **C** — heartbeats don't count as a first chunk |
| "Model returned empty after tool calls" | **A** — live content silently dropped |
| "Model returned no content after all retries" | **D** — no repair path for a genuinely empty turn |
| `/status` shows "Context: 0 / 1,048,576 (0%)" | **E** — usage tokens never populate outside one fragile extraction |

---

## Bug A — the completion is empty even though the shim had the text

**This is the one causing "empty after tool calls" / "empty from model."** It
is a certain code defect, not opencode flakiness.

### Where

`_do_session_turn`, the `stream_live` branch (~line 862):

```python
if stream_live:
    # Headers sent before the turn; text deltas already forwarded live.
    # Emit only the terminal chunk (no content duplication).
    cid, created = resp["id"], resp["created"]
    try:
        if calls:
            _sse_send_tool_calls(handler, cid, created, req_model, None, calls, tcs=tcs)
        else:
            done = {"id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": req_model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            handler.wfile.write(f"data: {json.dumps(done)}\n\ndata: [DONE]\n\n".encode())
    ...
```

The comment says "text deltas already forwarded live" — but that's only true
if `_emit_live` was actually called with real text during `_relay_turn`
(`~line 366-379`), which only happens when a `message.part.delta` event with
`field == "text"` arrives on `/event`. `/event` delivery for
`SyncEvent`-derived publishes is known to be unreliable across opencode
builds (some versions drop them entirely, some reorder them relative to
`message.part.updated`). Whenever that happens for a given turn, `_emit_live`
never fires, `stream_first` stays `True`, and the code above sends **nothing
but an empty terminal chunk** — even though `out` (the fully reconciled text,
fetched via `GET /session/:id/message` right after `session.idle`) is sitting
right there in scope, correct and non-empty. Hermes receives a "successful"
200 stream with zero content, which is exactly the shape of "Model returned
empty after tool calls."

### Fix

Track what was actually streamed, and if it doesn't cover the reconciled
text, send the difference before the terminal chunk.

**In `_relay_turn`** (~line 331), add an accumulator to `meta`:

```python
meta = {"ttfb_ms": 0, "usage": None, "events": 0, "streamed_chunks": 0, "streamed_text": ""}
```

**In the delta-handling block** (~line 366-379), accumulate what was sent:

```python
if etype == "message.part.delta":
    d = props.get("delta", "")
    if props.get("field") == "text" and d:
        if ttfb is None:
            ttfb = now
            meta["ttfb_ms"] = int((ttfb - t_start) * 1000)
        if on_delta:
            try:
                on_delta(d)
                meta["streamed_chunks"] += 1
                meta["streamed_text"] += d          # NEW
            except Exception:
                fatal = "client disconnected"
                break
        last_forward = now
```

**In `_do_session_turn`**, immediately before the `if stream_live:` block
(~line 862), add the catch-up:

```python
if stream_live and not calls:
    streamed = relay_meta.get("streamed_text", "")
    final_text = out or ""
    if not streamed and final_text.strip():
        # Zero live deltas arrived (event stream dropped them) but the
        # reconciled text exists — send it now instead of an empty 200.
        _emit_live(final_text)
        print(f"[stream] catch-up: 0 live chunks, sent {len(final_text)} reconciled chars")
    elif streamed and final_text.startswith(streamed) and len(final_text) > len(streamed):
        tail = final_text[len(streamed):]
        if tail.strip():
            _emit_live(tail)   # only the part that never streamed
            print(f"[stream] catch-up: tail of {len(tail)} chars not covered by live deltas")
```

`_emit_live` is already an in-scope closure at this point in the function, so
this is a pure addition — no signature changes needed elsewhere.

---

## Bug B — sentences glued together with no whitespace

**This is your first report** (`...contents.Pulling the file list...`).

### Where

Same delta loop, `~line 366-379`. `on_delta(d)` forwards `props.get("delta")`
verbatim with no awareness of which opencode *part* it belongs to. When
opencode narrates in short, separately-generated sentences (one per internal
step — extract, list, read, compare), each is very likely its own `partID`.
Consecutive parts get concatenated with nothing between them, because nothing
in the relay ever inserts a boundary.

### Fix

Track `partID` and insert a separator whenever it changes:

```python
# in _relay_turn, before the while-loop:
last_part_id = None

# inside the message.part.delta branch, before calling on_delta:
part_id = props.get("partID") or (props.get("part") or {}).get("id")
if part_id and part_id != last_part_id and meta["streamed_text"] and not meta["streamed_text"].endswith("\n"):
    sep = "\n\n"
    if on_delta:
        on_delta(sep)
    meta["streamed_text"] += sep
last_part_id = part_id
```

Put this right before the existing `if on_delta: on_delta(d); ...` block.
This alone fixes the readability of case 1 regardless of whether delivery is
smooth or bursty.

---

## Bug C — "waiting for stream response (90s, first_chunk)"

### Where

Two places conspire:

1. `_do_session_turn` opens the SSE stream (~line 652-660) and then calls
   `_run_once` → `_relay_turn`, but sends **no chunk at all** until the first
   real text delta arrives. If opencode spends the first 60-90s doing silent
   tool work (unzip, read, grep) before its first narrated sentence, Hermes
   sees zero bytes for that whole window.
2. The heartbeat in `_relay_turn` (~line 392-398) calls `on_delta(None)`,
   which `_emit_live` (~line 606-609) turns into a bare SSE comment:
   `handler.wfile.write(b": ping\n\n")`. An SSE comment line (`:`-prefixed)
   is not a `data:` frame — most OpenAI-client stream parsers, Hermes' almost
   certainly included, only reset a "time to first chunk" watchdog on an
   actual `chat.completion.chunk` object. The shim *is* sending heartbeats
   every 10s exactly as designed, but they're invisible to the timer that's
   actually killing the turn.

### Fix

Two small changes:

**a. Send an immediate role-delta the instant the stream opens**, before
`_run_once` is even called (~right after line 660, `stream_live = True`):

```python
if stream:
    try:
        handler.send_response(200)
        ...
        handler.end_headers()
        stream_live = True
        _emit_live("")   # NEW: zero-width role-delta, satisfies TTFB instantly
    except (BrokenPipeError, ConnectionResetError):
        ...
```

Check `_emit_live`'s existing logic — with `stream_first=True` it already
emits `{"role":"assistant","content":""}` for the first call regardless of
whether `text` is empty, since the only special-cased value is `None`
(heartbeat). Empty string `""` takes the normal branch. Confirm this in your
copy; if `_emit_live` treats falsy text as a no-op, special-case `""`
explicitly to still emit the role frame.

**b. Make ongoing heartbeats real (content-empty) chunks instead of SSE
comments**, so Hermes' watchdog — if it tracks "any chunk," not just the
first — keeps resetting through long silent tool stretches too:

```python
def _emit_live(text):
    nonlocal stream_first
    if text is None:
        chunk = {"id": stream_cid, "object": "chat.completion.chunk",
                 "created": stream_created, "model": req_model,
                 "choices": [{"index": 0, "delta": {}, "finish_reason": None}]}
        handler.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        handler.wfile.flush()
        return
    ...
```

This replaces the `: ping\n\n` comment with a spec-legal, content-empty
`data:` chunk — costs nothing, and is far more likely to be recognized by
whatever client library Hermes uses.

---

## Bug D — genuinely empty turns have no safety net

**This is "no content after all retries" on the agentic status you pasted**
(iteration 3/150, long tool loop). Distinct from Bug A: here `out` really is
empty after reconciliation — opencode's turn ended with a tool call but no
narration text at all.

### Where

The only repair path (`~line 761-814`) is gated by:

```python
if tools and tool_choice != "none":
    ...
    if ferr and SHIM_REPAIR_TURNS > 0:
        # repair turn
```

`ferr` is only set when `must_call` is true (forced tool_choice) and no
fence was found, or fence-parsing itself failed. If `tools` is empty
(Profile B — no tools offered, which is the agentic-delegation case your
robot-balance-control status shows) or `tool_choice == "none"`, this entire
block is skipped. A turn that ends with `calls == []` and `out == ""` sails
straight through as a "successful" `finish_reason: "stop"` completion with
empty content. Hermes retries three times against a shim that has no
mechanism to notice or correct this, then gives up.

### Fix

Add a universal empty-completion guard, independent of `tools`/`tool_choice`,
right after fence-parsing settles `calls`/`out` (~line 822, after the
`elif fence == "text" and must_call:` line):

```python
if not calls and not (out or "").strip() and SHIM_REPAIR_TURNS > 0:
    print(f"[repair] {sid[:14]} empty completion, no tool call — nudging once")
    nudge = ("[system] Your previous turn produced no visible output. "
             "Respond now — either plain text, or a ```hermes-toolcalls "
             "block if a tool call is needed.")
    ok_n, out_n, err_n = False, "", ""
    if _sem.acquire(blocking=True, timeout=60):
        try:
            slock_n = L1.session_lock(sid)
            if slock_n.acquire(blocking=True, timeout=60):
                try:
                    ok_n, out_n, err_n, meta_n, _ = _run_once(sid, nudge, [])
                    relay_meta.update({k: v for k, v in meta_n.items() if v and k != "ttfb_ms"})
                finally:
                    slock_n.release()
        finally:
            _sem.release()
    if ok_n and out_n.strip():
        out = raw_out = out_n
        fence = "repaired-empty"
        if tools and tool_choice != "none":
            try:
                calls2, remaining2, _ = parse_fence(out, tools, forced_name)
                if calls2:
                    calls, out = calls2, remaining2
                    tcs = _openai_tool_calls(calls)
                    store.note_tool_calls(sid, {tc["id"]: tc["function"]["name"] for tc in tcs})
            except Exception:
                pass
    else:
        fail(502, "opencode produced no output for this turn, even after a nudge", "502", "backend_error")
        return
```

`502` here is deliberate: it lets Hermes' own retry/backoff treat this as a
real upstream failure instead of quietly accepting an empty `200` and
grinding through its own three retries against a shim that already knows the
turn failed.

---

## Bug E — `/status` shows "Context: 0 / 1,048,576 (0%)"

**This is the context/lifetime-tokens regression.** Distinct from A-D — this
one is about response metadata, not content.

### Where

`chat_completion_response` / `chat_completion_tool_response` both default
`usage` to `{"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}`
(~line 1743, ~line 1804) whenever the caller passes `usage=None`. Across the
whole file there is exactly **one** place that ever sets it to something
else — `_relay_turn`'s post-`session.idle` reconciliation (~line 415-419):

```python
toks = last_m.get("info", {}).get("tokens", {}) or {}
if isinstance(toks, dict) and (toks.get("input") or toks.get("output")):
    meta["usage"] = {"prompt_tokens": int(toks.get("input") or 0),
                     "completion_tokens": int(toks.get("output") or 0),
                     "total_tokens": int(toks.get("input") or 0) + int(toks.get("output") or 0)}
```

`run_opencode` (the old flatten path, ~line 1646) never sets usage at all —
it was always zero there too. So this isn't a pure regression from the
session rework; it's a gap that the rework made visible, because the
session/delta model is what makes an accurate "context size" figure
actually meaningful and worth showing correctly.

Two independent reasons this extraction can come up empty:

1. **Wrong or version-drifted field path.** opencode's token/cost schema
   has changed shape across releases (input/output/reasoning/cache split
   differently release to release), and there's a documented case of
   custom/OpenAI-compatible providers — which is exactly what Zen-via-opencode
   looks like from opencode's side — never getting usage populated by
   opencode at all, with cost staying at $0 regardless of real token
   consumption. If that's what's happening here, `info.tokens` is genuinely
   empty at the source and no path fix will find real data.
2. **Wrong scope even when present.** `last_m` is only the *last* assistant
   message. Its token count (if populated) reflects that one message, not
   the full session context Hermes wants to display. In the old flatten
   world this didn't matter (every request carried full history, so "this
   turn's tokens" ≈ "context size"); in the delta model it does — a correct
   per-message extraction would still under-report the true window fill.

### Fix

Diagnose first, one line, cheap: log the full `info` blob on a real turn to
see which of the two you have.

```python
print(f"[usage-debug] {json.dumps(last_m.get('info', {}))[:1000]}")
```

Then, regardless of what that shows, add a fallback so `/status` never shows
a flat, permanent zero again. Estimate cumulative session size from
everything opencode is holding (already fetched as `msgs` in the same
reconciliation call, no extra request needed) rather than just the last
message:

```python
# in _relay_turn, right after the existing toks extraction, still inside
# the `try:` block that has `msgs` in scope:
if not meta["usage"]:
    all_text = sum(len(p.get("text", "")) for m in msgs for p in m.get("parts", [])
                   if p.get("type") == "text")
    est_total = max(1, all_text // 4)          # ~4 chars/token, rough but honest
    meta["usage"] = {"prompt_tokens": est_total,
                     "completion_tokens": max(1, len(full) // 4),
                     "total_tokens": est_total + max(1, len(full) // 4),
                     "estimated": True}         # extra key; harmless, OpenAI clients ignore unknown fields
```

This gives Hermes a real, monotonically-growing number instead of a
permanent zero. It's an estimate, not an exact token count — fine for a
context-fill percentage, not fine if anything downstream bills against it.
If test 6 below shows real `tokens.input`/`tokens.output` are actually
present under a different key, replace the estimate with the correct
extraction instead of keeping the fallback as primary.

---

## Optional — surface tool-step narration during silent stretches

Not a bug fix, closer to the original UX ask ("show it progressively like
before"). Currently `_relay_turn` only acts on `message.part.delta` events;
`message.part.updated` events for non-text parts (tool calls) are counted
toward `last_event`/`meta["events"]` but never surfaced. Turning a completed
tool part into a short status line, forwarded the same way as text, gives
visible progress during the silent gaps that also trigger Bug C:

```python
elif etype == "message.part.updated":
    part = props.get("part") or {}
    if SHIM_NARRATE_TOOLS and part.get("type") == "tool" and part.get("state", {}).get("status") == "completed":
        tool_name = part.get("tool") or "tool"
        summary = f"\n⚙ {tool_name}\n"
        if on_delta:
            on_delta(summary)
        meta["streamed_text"] += summary
```

Gate behind `SHIM_NARRATE_TOOLS=1` (default off) since it changes visible
output shape — turn on, watch a few real turns, decide if it's worth keeping
on by default. This doesn't force Hermes to render separate chat bubbles
(that's a Hermes-side rendering choice this shim can't control), but it does
mean Hermes has something real to display instead of silence, which is most
of what "showing progress like before" was actually asking for.

---

## Test additions

Add to the existing suite:

1. **Bug A regression**: mock `/event` to never emit `message.part.delta`
   for a turn, but let `GET /session/:id/message` return real text after
   `session.idle`. Assert the SSE stream to the client still contains that
   text before `[DONE]`.
2. **Bug A partial-coverage**: mock `/event` to emit deltas covering only
   the first half of the reconciled text. Assert only the tail is sent as
   catch-up (no duplication).
3. **Bug B**: mock two consecutive `message.part.delta` events with
   different `partID`s and no leading/trailing whitespace in either delta.
   Assert the forwarded stream contains a separator between them.
4. **Bug C**: assert a `data:` chunk (not an SSE comment) is written within
   the first event loop tick after headers open, before any real content
   exists.
5. **Bug D**: mock a turn that ends with `session.idle`, zero text parts,
   zero tool call fence. Assert one nudge turn is attempted, and if the nudge
   also comes back empty, assert `502` — not a silent empty `200`.
6. **Bug E**: log `last_m.get("info")` in full on one real live turn against
   the actual opencode/Zen backend (not mocked) to determine whether
   `tokens` is present-but-misnamed, present-under-a-different-parent-key, or
   absent entirely. This determines whether the fallback estimate should stay
   primary or become a backstop behind a corrected real extraction. Separately,
   assert `usage.prompt_tokens` grows turn-over-turn across a multi-turn
   session (proves the estimate is cumulative, not per-delta).

## Rollout

1. Patch `_relay_turn` (Bugs A, B — accumulator + separator) and `_emit_live`
   (Bug C — real heartbeat chunks) together; they touch the same functions.
2. Patch `_do_session_turn` catch-up block (Bug A) and empty-guard (Bug D)
   together; both sit in the same post-relay section.
3. Patch the usage debug log + fallback estimate (Bug E) in the same
   `_relay_turn` edit pass as Bugs A/B, since all three touch the
   reconciliation block right after `session.idle`.
4. `python3 -m py_compile server.py`, restart, run tests 1-5 above against a
   mocked `/event` before touching the real Zen backend — these are exactly
   the failure modes that are cheap to simulate and expensive to reproduce
   live.
5. One real Hermes conversation through the linked channel with a task that
   involves at least one silent tool-only stretch (e.g. reprise the archive
   analysis from your report) — confirm no glued sentences, no empty
   completions, no 90s first-chunk stalls, and check `/status` shows a
   nonzero, growing context figure (test 6).
6. Watch `/debug/stats` for a day: `streamed_chunks == 0` on turns where the
   catch-up path fired tells you how often `/event` is actually dropping
   deltas on this opencode build — worth knowing regardless of whether Bug A
   is now masked correctly. Also check the `[usage-debug]` log line once to
   settle Bug E's root cause, then remove that print.

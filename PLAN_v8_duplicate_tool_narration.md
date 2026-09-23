# opencode-hermes-shim — v8: duplicate tool-narration triggers false repetition loops

Date: 2026-09-23. Root cause confirmed by direct review of `_relay_turn`'s
`message.part.updated` handling in the current `server.py`.

## Symptom

Two linked transcript pieces from the same session:
1. Hermes' own "Response Stopped — Repetition Detected" safety net fired and
   discarded a partial response.
2. On retry, the visible output was ~10 near-identical lines:
   `⚙ bash… curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool
   | head -n 200 echo "---SHIM HEALTH---" curl -s http://127…` — repeated
   back to back, plus one `⚙ invalid… {"tool":"default.terminal",...}` line.

## Root cause

`SHIM_NARRATE_TOOLS` (default `"1"`, on) turns opencode's internal tool
activity into visible narration lines for Profile B (agentic, no
Hermes-offered tools) turns. The `ptype == "reasoning"` branch right below it
correctly dedupes by part ID:

```python
elif ptype == "reasoning" and SHIM_NARRATE_TOOLS:
    _pid = part.get("id")
    _txt = part.get("text") if isinstance(part.get("text"), str) else ""
    if _txt and _pid not in seen_reasoning_parts:
        seen_reasoning_parts.add(_pid)
        ...
```

The `ptype == "tool"` branch immediately above it has **no such dedup at
all** — every `message.part.updated` event for a tool part emits a fresh
narration line, unconditionally:

```python
if ptype == "tool" and SHIM_NARRATE_TOOLS:
    st = part.get("state") if isinstance(part.get("state"), dict) else {}
    status = (st or {}).get("status")
    ...
    if status == "running":
        ...
        line = f"\n⚙ {tool_name}… {title}\n" if title else f"\n⚙ {tool_name}…\n"
    elif status == "completed":
        ...
```

opencode fires `message.part.updated` for a tool part **more than once while
it's still `status:"running"`** — most plausibly as the tool call's own
input/arguments stream in progressively (a `bash` command string filling in
character by character is a normal shape for a tool-calling model's
output). Each of those incremental updates re-enters this branch, and since
nothing tracks "have I already narrated this part's `running` state,"
every one of them produces a fresh, near-identical `⚙ bash… curl -s ...`
line — for what is genuinely **one** tool call.

This is not the model looping. It's the shim manufacturing the appearance of
a loop out of a single tool call's normal incremental status updates. Two
consequences, both visible in the transcript:

- Hermes' own repetition detector operates on the text it actually
  receives — ten near-identical narration lines in a row is exactly what it
  exists to catch, so it correctly (from its own point of view) flagged and
  discarded the response. The detector isn't malfunctioning; it's reacting
  correctly to shim-manufactured repetition.
- The `⚙ invalid… {"tool":"default.terminal",...}` line is a separate,
  genuine event (the model did try a real but unavailable tool name once) —
  that one's real, and unrelated to this bug. It's a one-off worth noting
  but not the cause of the repeated-lines pattern.

## Fix

Track which lifecycle stage has already been narrated per tool part ID,
mirroring the pattern the `reasoning` branch already uses:

```python
# near the other per-turn trackers, alongside seen_reasoning_parts:
seen_tool_parts = {}   # part_id -> set of statuses already narrated
```

```python
elif etype == "message.part.updated":
    part = props.get("part") or {}
    ptype = part.get("type")
    if ptype == "tool" and SHIM_NARRATE_TOOLS:
        _tpid = part.get("id")
        st = part.get("state") if isinstance(part.get("state"), dict) else {}
        status = (st or {}).get("status")
        tool_name = part.get("tool") or "tool"
        _seen = seen_tool_parts.setdefault(_tpid, set()) if _tpid else None
        line = None

        def _once(tag):
            # No part id (shouldn't happen, but don't silently drop the
            # narration if it does) -> always allow; otherwise narrate a
            # given lifecycle stage for this part exactly once.
            if _seen is None:
                return True
            if tag in _seen:
                return False
            _seen.add(tag)
            return True

        if status == "running" and _once("running"):
            preview = _tool_input_preview((st or {}).get("input"))
            title = (st or {}).get("title") or preview
            line = f"\n⚙ {tool_name}… {title}\n" if title else f"\n⚙ {tool_name}…\n"
        elif status == "completed" and _once("completed"):
            title = (st or {}).get("title") or _tool_input_preview((st or {}).get("input"))
            dur = ""
            try:
                _t = (st or {}).get("time") or {}
                if _t.get("start") and _t.get("end"):
                    dur = f" ({float(_t['end'] - _t['start']) / 1000:.1f}s)"
            except Exception:
                dur = ""
            tail = f" — {title}" if title else ""
            line = f"\n⚙ {tool_name}{tail}{dur}\n"
        elif status == "error" and _once("error"):
            err_s = str((st or {}).get("error") or "")[:200].replace("\n", " ")
            line = f"\n⚙ {tool_name} failed: {err_s}\n" if err_s else f"\n⚙ {tool_name} failed\n"

        if line:
            if ttfb is None:
                ttfb = now
                meta["ttfb_ms"] = int((ttfb - t_start) * 1000)
            if on_delta:
                try:
                    on_delta(line)
                except Exception:
                    fatal = "client disconnected"
                    break
            meta["displayed_text"] += line
            last_forward = now
```

One deliberate design choice: `_once("running")` only allows the **first**
`running` narration per part, even if the tool's title/preview text changes
on later `running` updates (e.g. the command string growing as it streams
in). Showing the evolving partial command would be nice-to-have but isn't
worth the complexity here — the goal is eliminating the false-repetition
symptom, not building a live-updating tool-call card. `completed`/`error`
are guarded the same way defensively, in case opencode's event bus ever
redelivers a terminal state (e.g. on an SSE reconnect that replays a
backlog) — a genuine one-time transition being narrated twice is a much
smaller problem than what's happening today, but there's no reason to leave
it unguarded when the fix is the same three lines.

## What this doesn't explain, and shouldn't be conflated with it

The `default.terminal` invalid-tool call is a real, separate event — the
model actually attempted a tool name outside its offered set (`bash`,
`edit`). That's either a model-side confusion (plausible for a free-tier
model under agentic pressure) or a prompt/tool-list mismatch worth watching
via `/debug/stats` if it recurs, but it is not caused by this bug and
patching the narration dedup won't touch it. If `default.terminal` shows up
again after this fix, it's worth its own investigation — likely which agent
config (`hermes` vs `hermes-agent` from the Profile A/B split) the model
believes it's running under versus which tools are actually registered.

## Tests to add

1. **Streaming-input tool call**: mock a single tool part receiving 5
   `message.part.updated` events, all `status:"running"`, with a growing
   `input.command` string, followed by one `status:"completed"` event.
   Assert exactly **one** `running` narration line and **one** `completed`
   narration line reach `on_delta` — not five.
2. **Multiple distinct tool calls in one turn**: two different tool parts
   (different `part.id`), each with their own running→completed sequence.
   Assert each gets its own one-running/one-completed pair — confirm the
   dedup is keyed per part ID, not globally (a global dedup would wrongly
   suppress the second tool call's narration entirely).
3. **Error after running**: a part that goes running → error (no
   completed). Assert both the running and error lines appear once each,
   and no completed line is fabricated.
4. **Reconnect replay defense**: mock the SSE subscriber reconnecting mid-turn
   and replaying an already-seen `completed` event for a part already
   marked complete. Assert no duplicate `completed` line is emitted —
   validates the defensive half of the fix, not just the streaming-input
   case that motivated it.

## Rollout

1. Add `seen_tool_parts = {}` next to the existing `seen_reasoning_parts`
   initialization, and apply the `_once()`-gated branch above — isolated to
   the `message.part.updated`/`ptype=="tool"` block, no other code path
   touched.
2. `python3 -m py_compile server.py`, restart, run tests 1-4 against a
   mocked `/event` stream.
3. Re-run the exact task from the report (asking the shim/model to inspect
   its own model-list wiring, which is what triggered the original
   multi-`curl` bash sequence) and confirm the narration now shows one line
   per real tool call instead of one line per incremental status update.
4. Watch whether Hermes' repetition detector still fires on any real,
   unrelated shim traffic over the next few days — if the false-positive
   pattern is gone, that's strong confirmation this was the actual cause
   rather than a coincidence with this one transcript.

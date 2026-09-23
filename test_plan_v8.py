#!/usr/bin/env python3
"""PLAN v8 acceptance tests (stdlib unittest, mocked /event stream, no backend).

Regression: every `message.part.updated` for a tool part emitted a fresh
narration line, so one tool call whose input streamed in progressively
looked like ~10 near-identical `bash...` calls and tripped Hermes'
repetition detector. The fix dedupes narration per (part_id, status) via
`seen_tool_parts` + `_once()` in `_relay_turn`.

These tests drive the real `_relay_turn` with scripted events through
monkeypatched `_prompt_async` / `_event_subscriber` / `_serve_call` and
assert what reaches `on_delta`.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server

SID = "ses_v8test"


def _upd(part):
    return ("event", {"type": "message.part.updated",
                      "properties": {"sessionID": SID, "part": part}})


def _idle():
    return ("event", {"type": "session.idle",
                      "properties": {"sessionID": SID}})


def _tool_part(pid, tool, status, command=None, title=None, err=None):
    state = {"status": status}
    if command is not None:
        state["input"] = {"command": command}
    else:
        state["input"] = {}
    if title is not None:
        state["title"] = title
    if status == "completed":
        state.update({"output": "ok", "metadata": {},
                      "time": {"start": 1000, "end": 2500}})
    if err is not None:
        state["error"] = err
    part = {"id": pid, "sessionID": SID, "messageID": "msg_1",
            "type": "tool", "tool": tool, "state": state}
    if pid is None:
        del part["id"]
    return part


def _assistant_msg():
    return {"info": {"role": "assistant",
                     "tokens": {"input": 100, "output": 10, "reasoning": 0,
                                "cache": {"read": 500, "write": 0}}},
            "parts": [{"type": "text", "text": "done"}]}


def _run_scripted(script):
    """Run _relay_turn against a scripted event list. Returns (ok, lines)."""
    lines = []

    def on_delta(d):
        if d is not None:
            lines.append(d)

    orig_prompt = server._prompt_async
    orig_sub = server._event_subscriber
    orig_call = server._serve_call

    def fake_prompt(sid, body):
        return True, ""

    def fake_sub(stop_flag, q):
        for item in script:
            q.put(item)

    def fake_call(method, path, payload=None, timeout=30):
        return True, [_assistant_msg()]

    server._prompt_async = fake_prompt
    server._event_subscriber = fake_sub
    server._serve_call = fake_call
    try:
        ok, full, err, meta = server._relay_turn(
            SID, {"parts": []}, 30, on_delta=on_delta)
    finally:
        server._prompt_async = orig_prompt
        server._event_subscriber = orig_sub
        server._serve_call = orig_call
    return ok, err, lines


class TestV8Dedup(unittest.TestCase):
    def setUp(self):
        self._orig_narrate = server.SHIM_NARRATE_TOOLS
        server.SHIM_NARRATE_TOOLS = True

    def tearDown(self):
        server.SHIM_NARRATE_TOOLS = self._orig_narrate

    def test_streaming_input_single_call_narrated_once_per_stage(self):
        # Test 1 (plan §Tests.1): one tool call, 5 running updates as the
        # command streams in, then completed -> 1 running + 1 completed.
        base = "curl -s http://127.0.0.1:8000/v1/models"
        script = [_upd(_tool_part("p1", "bash", "running",
                                  command=base[:i]))
                  for i in (10, 20, 30, len(base) - 5, len(base))]
        script.append(_upd(_tool_part("p1", "bash", "completed",
                                      command=base, title=base)))
        script.append(_idle())
        ok, err, lines = _run_scripted(script)
        self.assertTrue(ok, msg=err)
        running = [l for l in lines if "…" in l]
        done = [l for l in lines if "—" in l and "bash" in l]
        self.assertEqual(len(running), 1, msg=lines)
        self.assertEqual(len(done), 1, msg=lines)
        self.assertEqual(len(lines), 2, msg=lines)

    def test_distinct_parts_each_narrated(self):
        # Test 2: dedup is keyed per part ID, not globally.
        script = [
            _upd(_tool_part("p1", "bash", "running", command="ls -la")),
            _upd(_tool_part("p2", "read", "running", command="/tmp/a.txt")),
            _upd(_tool_part("p1", "bash", "completed", command="ls -la")),
            _upd(_tool_part("p2", "read", "completed",
                            command="/tmp/a.txt")),
            _idle(),
        ]
        ok, err, lines = _run_scripted(script)
        self.assertTrue(ok, msg=err)
        self.assertEqual(len(lines), 4, msg=lines)
        self.assertEqual(sum("bash" in l for l in lines), 2, msg=lines)
        self.assertEqual(sum("read" in l for l in lines), 2, msg=lines)

    def test_error_after_running(self):
        # Test 3: running -> error gives one running + one error, no
        # completed line fabricated.
        script = [
            _upd(_tool_part("p1", "bash", "running", command="rm -rf /")),
            _upd(_tool_part("p1", "bash", "error", err="denied by policy")),
            _idle(),
        ]
        ok, err, lines = _run_scripted(script)
        self.assertTrue(ok, msg=err)
        self.assertEqual(len(lines), 2, msg=lines)
        self.assertTrue(any("…" in l for l in lines), msg=lines)
        self.assertTrue(any("failed" in l for l in lines), msg=lines)
        self.assertFalse(any("—" in l for l in lines), msg=lines)

    def test_reconnect_replay_no_duplicate_completed(self):
        # Test 4: redelivered terminal state must not re-narrate.
        script = [
            _upd(_tool_part("p1", "bash", "running", command="echo hi")),
            _upd(_tool_part("p1", "bash", "completed", command="echo hi")),
            _upd(_tool_part("p1", "bash", "completed", command="echo hi")),
            _idle(),
        ]
        ok, err, lines = _run_scripted(script)
        self.assertTrue(ok, msg=err)
        self.assertEqual(len(lines), 2, msg=lines)

    def test_missing_part_id_fails_open(self):
        # No part id (shouldn't happen): narration must not be silently
        # dropped — both updates still emit.
        script = [
            _upd(_tool_part(None, "bash", "running", command="echo hi")),
            _upd(_tool_part(None, "bash", "running", command="echo hi")),
            _idle(),
        ]
        ok, err, lines = _run_scripted(script)
        self.assertTrue(ok, msg=err)
        self.assertEqual(len(lines), 2, msg=lines)


if __name__ == "__main__":
    unittest.main(verbosity=2)

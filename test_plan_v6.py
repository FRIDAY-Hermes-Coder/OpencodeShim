#!/usr/bin/env python3
"""PLAN v6 acceptance tests (stdlib unittest, no backend needed).

Regression: live streaming was disabled whenever stop/max_tokens was merely
present (not when a cutoff fired). The fix wraps on_delta in
_server._make_gated_delta_ so deltas forward normally until a cutoff hits.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server


def _collect():
    out = []

    def emit(d):
        out.append(d)

    return out, emit


def _run_gated(deltas, stop=None, max_tokens=None):
    out, emit = _collect()
    gated = server._make_gated_delta(emit, stop, max_tokens)
    for d in deltas:
        gated(d)
    return out, gated


class TestGatedDelta(unittest.TestCase):
    def test_noop_generous_budget_streams_normally(self):
        # Regression case: short text well under a generous budget must
        # stream exactly like no budget at all.
        deltas = ["hello ", "world", "!"]
        out_gated, _ = _run_gated(deltas, stop=None, max_tokens=4096)
        out_plain, _ = _run_gated(deltas, stop=None, max_tokens=None)
        self.assertEqual(out_gated, deltas)
        self.assertEqual(out_plain, deltas)
        self.assertEqual(len(out_gated), len(out_plain))
        self.assertEqual(len(out_gated), 3)

    def test_stop_cutoff(self):
        stop = "STOP"
        full = "hello wo STOP rld!!!"
        expected, hit = server._apply_stop(full, stop)
        self.assertTrue(hit)
        deltas = ["hello ", "wo STOP rld", "!!!"]
        out, gated = _run_gated(deltas, stop=stop, max_tokens=None)
        self.assertEqual("".join(x for x in out if x is not None), expected)
        # Further deltas suppressed after cutoff fired.
        gated("LATE DATA THAT MUST NOT LEAK")
        self.assertEqual("".join(x for x in out if x is not None), expected)

    def test_split_sequence_across_deltas(self):
        # Stop string split as "STO" + "P" across two deltas must still
        # trigger the cutoff via the accumulated buffer.
        stop = "STOP"
        out, emit = _collect()
        gated = server._make_gated_delta(emit, stop, None)
        gated("abc STO")
        gated("P def")
        # Cutoff must have fired on the second delta; trailing content
        # after STOP (" def") must not have been forwarded as a new chunk.
        gated(" MORE THAT MUST NOT LEAK")
        joined = "".join(x for x in out if x is not None)
        self.assertNotIn("def", joined.replace("STO", ""))
        self.assertNotIn("MORE", joined)
        # No further visible bytes after the cutoff delta.
        n = len(out)
        gated("EVEN MORE")
        self.assertEqual(len(out), n)

    def test_max_tokens_cutoff_matches_apply(self):
        max_tokens = 5  # budget = 20 chars, same heuristic as _apply_max_tokens
        deltas = ["x" * 10, "y" * 10, "z" * 10]
        full = "".join(deltas)
        expected, hit = server._apply_max_tokens(full, max_tokens)
        self.assertTrue(hit)
        self.assertEqual(len(expected), 20)
        out, gated = _run_gated(deltas, stop=None, max_tokens=max_tokens)
        joined = "".join(x for x in out if x is not None)
        self.assertEqual(joined, expected)
        # Post-cutoff deltas suppressed.
        n = len(out)
        gated("LATE")
        self.assertEqual(len(out), n)

    def test_response_format_still_buffers(self):
        # Call-site rule: whenever response_format is set, on_delta is None
        # regardless of stop/max_tokens. Mirror the production condition
        # (do not refactor production code just to test it).
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "server.py")) as f:
            src = f.read()
        self.assertIn("_needs_full_buffer = bool(response_format)", src)
        self.assertIn("_make_gated_delta", src)
        self.assertNotIn(
            "_buffered = bool(stop or max_tokens or response_format)", src)

        def select_on_delta(stream, tools, response_format, stop, max_tokens):
            _needs_full_buffer = bool(response_format)
            on_delta = None
            if stream and not tools and not _needs_full_buffer:
                on_delta = (server._make_gated_delta(None, stop, max_tokens)
                            if (stop or max_tokens) else "LIVE")
            return on_delta

        self.assertIsNone(select_on_delta(True, False, {"type": "json_object"},
                                          "STOP", 4096))
        self.assertIsNone(select_on_delta(True, False, {"type": "json_object"},
                                          None, None))
        # Sanity: without response_format the gated/live path is selected.
        self.assertIsNotNone(select_on_delta(True, False, None, "STOP", None))
        self.assertIsNotNone(select_on_delta(True, False, None, None, 100))
        self.assertEqual(select_on_delta(True, False, None, None, None), "LIVE")
        # Tool-offered turns still buffer (Bug F fence).
        self.assertIsNone(select_on_delta(True, True, None, "STOP", 100))

    def test_heartbeat_passthrough_after_cutoff(self):
        out, emit = _collect()
        gated = server._make_gated_delta(emit, "STOP", None)
        gated("hello STOP world")
        # Cutoff fired; heartbeats must still forward so the connection
        # stays alive until the turn finishes server-side.
        gated(None)
        self.assertIn(None, out)
        n_none = sum(1 for x in out if x is None)
        gated(None)
        self.assertEqual(sum(1 for x in out if x is None), n_none + 1)
        # Heartbeat with no inner_emit must not crash.
        g2 = server._make_gated_delta(None, "STOP", 1)
        g2(None)
        g2("abc")

    def test_bad_max_tokens_means_no_budget(self):
        # Non-int / falsy / <=0 max_tokens behaves like no budget (mirrors
        # _apply_max_tokens): streams normally, never cuts.
        for bad in ("bad", 0, -3, None):
            out, _ = _run_gated(["hello ", "world"], stop=None,
                                max_tokens=bad)
            self.assertEqual(out, ["hello ", "world"], msg=repr(bad))


if __name__ == "__main__":
    unittest.main(verbosity=2)

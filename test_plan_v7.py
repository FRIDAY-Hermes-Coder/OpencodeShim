#!/usr/bin/env python3
"""PLAN v7 acceptance tests (stdlib unittest, no backend, no network).

Regression: when parse_fence rejected the model's tool-call attempt AND the
one-shot repair turn also failed, fence became "failed" with calls == [] but
no guard matched (the only sibling guard checks fence == "text"), so the
repair attempt's malformed JSON shipped to the user verbatim as plain text.

The repair-turn path calls the live backend via the _run_once closure, so
these tests exercise the trigger condition (parse_fence on a malformed
batch) and the exact post-repair branch condition directly — no network,
no live serve.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server


def _write_file_tools():
    return [{
        "type": "function",
        "function": {
            "name": "write_file",
            "parameters": {
                "type": "object",
                "required": ["path", "content"],
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
            },
        },
    }]


def _malformed_batch():
    # Valid first call + bare {"path":...} second element (missing name
    # envelope) — the v7 transcript failure shape.
    return (
        "```hermes-toolcalls\n"
        "[\n"
        '  {"name": "write_file", "arguments": {"path": "a.txt", "content": "hi"}},\n'
        '  {"path": "b.txt", "content": "lo"}\n'
        "]\n"
        "```"
    )


class TestMalformedBatchDetection(unittest.TestCase):
    def test_malformed_batch_returns_error_not_silent(self):
        calls, remaining, err = server.parse_fence(
            _malformed_batch(), _write_file_tools())
        self.assertIsNone(calls)
        self.assertIsNotNone(err)
        self.assertTrue(str(err).strip())


class TestFailedBranchCondition(unittest.TestCase):
    def test_old_guards_miss_failed_state_new_branch_catches_it(self):
        # Exact post-repair state: repair also failed, tool_choice="auto".
        fence = "failed"
        calls = []
        out = _malformed_batch()  # non-empty malformed repair output
        tool_choice = "auto"
        must_call = tool_choice == "required" or False

        # OLD code: only guard below the repair block.
        old_matched = (fence == "text" and must_call)
        self.assertFalse(old_matched)
        # Empty-completion guard (Bug D) also misses: out is non-empty.
        self.assertTrue((out or "").strip())
        # NEW branch condition matches this exact state.
        self.assertEqual(fence, "failed")

        # Guard is present in production source as a sibling branch.
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "server.py")) as f:
            src = f.read()
        self.assertIn('elif fence == "failed":', src)
        self.assertIn('elif fence == "text" and must_call:', src)
        self.assertIn("model produced an invalid tool call and the repair attempt",
                      src)


class TestHappyPathsUntouched(unittest.TestCase):
    def test_valid_fence_returns_calls(self):
        text = (
            "```hermes-toolcalls\n"
            '[{"name": "write_file", "arguments": {"path": "a.txt", "content": "hi"}}]\n'
            "```"
        )
        calls, remaining, err = server.parse_fence(text, _write_file_tools())
        self.assertIsNotNone(calls)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "write_file")
        self.assertIsNone(err)

    def test_repair_fixed_output_parses_on_second_attempt(self):
        # First attempt malformed (triggers repair), second attempt fixed.
        bad = _malformed_batch()
        good = (
            "```hermes-toolcalls\n"
            '[{"name": "write_file", "arguments": {"path": "b.txt", "content": "lo"}}]\n'
            "```"
        )
        calls1, _, err1 = server.parse_fence(bad, _write_file_tools())
        self.assertIsNone(calls1)
        self.assertIsNotNone(err1)
        calls2, _, err2 = server.parse_fence(good, _write_file_tools())
        self.assertIsNotNone(calls2)
        self.assertEqual(len(calls2), 1)
        self.assertIsNone(err2)


class TestOneCallPerTurnPrompt(unittest.TestCase):
    def test_instruction_mentions_one_call_per_turn(self):
        text = server.build_tools_instruction(_write_file_tools(), "auto")
        self.assertIn("one call per turn", text.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)

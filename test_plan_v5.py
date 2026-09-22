#!/usr/bin/env python3
"""PLAN v5 acceptance tests (stdlib unittest, no backend needed).

Covers every plan phase that is unit-testable without a live opencode serve:
guard rejections, post-hoc shaping, parallel-call truncation, response_format
validation (incl. json_schema subset), content-part parsing, error shape,
response shape, and session canon stability.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server
import sessions


class TestCompatGuards(unittest.TestCase):
    def test_n_int_rejected(self):
        code, body = server._compat_guard({"n": 2})
        self.assertEqual(code, 400)
        self.assertEqual(body["error"]["param"], "n")
        self.assertEqual(body["error"]["code"], "unsupported_parameter")
        self.assertEqual(body["error"]["type"], "invalid_request_error")

    def test_n_float_rejected(self):
        self.assertIsNotNone(server._compat_guard({"n": 2.0}))

    def test_n_one_and_bool_allowed(self):
        self.assertIsNone(server._compat_guard({"n": 1}))
        self.assertIsNone(server._compat_guard({"n": True}))
        self.assertIsNone(server._compat_guard({}))

    def test_logprobs_truthy_rejected_falsy_allowed(self):
        self.assertIsNone(server._compat_guard({"logprobs": False}))
        code, body = server._compat_guard({"logprobs": True})
        self.assertEqual(code, 400)
        self.assertEqual(body["error"]["param"], "logprobs")

    def test_top_logprobs(self):
        self.assertIsNone(server._compat_guard({"top_logprobs": 0}))
        code, body = server._compat_guard({"top_logprobs": 5})
        self.assertEqual(code, 400)
        self.assertEqual(body["error"]["param"], "top_logprobs")

    def test_seed_any_value_rejected_absent_allowed(self):
        self.assertIsNone(server._compat_guard({}))
        for v in (0, 42):
            code, body = server._compat_guard({"seed": v})
            self.assertEqual(code, 400)
            self.assertEqual(body["error"]["param"], "seed")

    def test_logit_bias_empty_allowed_nonempty_rejected(self):
        self.assertIsNone(server._compat_guard({"logit_bias": {}}))
        code, body = server._compat_guard({"logit_bias": {"42": 1}})
        self.assertEqual(code, 400)
        self.assertEqual(body["error"]["param"], "logit_bias")


class TestShaping(unittest.TestCase):
    def test_stop_str_and_list(self):
        out, hit = server._apply_stop("hello STOP world", "STOP")
        self.assertEqual((out, hit), ("hello ", True))
        out, hit = server._apply_stop("a.X b.Y", [".Y", ".X"])
        self.assertEqual((out, hit), ("a", True))
        out, hit = server._apply_stop("clean", "ZZZ")
        self.assertEqual((out, hit), ("clean", False))

    def test_max_tokens_budget(self):
        out, hit = server._apply_max_tokens("x" * 100, 10)
        self.assertTrue(hit)
        self.assertEqual(len(out), 40)  # 10 tokens x ~4 chars
        out, hit = server._apply_max_tokens("short", 10)
        self.assertFalse(hit)
        self.assertEqual(server._apply_max_tokens("abc", 0), ("abc", False))
        self.assertEqual(server._apply_max_tokens("abc", "bad"), ("abc", False))
        self.assertEqual(server._apply_max_tokens("abc", None), ("abc", False))

    def test_parallel_truncation(self):
        calls = [{"name": "a"}, {"name": "b"}]
        self.assertEqual(len(server._limit_parallel_calls(calls, False)), 1)
        self.assertEqual(server._limit_parallel_calls(calls, False)[0]["name"], "a")
        self.assertEqual(len(server._limit_parallel_calls(calls, True)), 2)
        self.assertEqual(len(server._limit_parallel_calls(calls, None)), 2)


class TestResponseFormat(unittest.TestCase):
    def test_json_object(self):
        ok, _ = server._validate_response_format('{"a": 1}', {"type": "json_object"})
        self.assertTrue(ok)
        ok, err = server._validate_response_format('nope', {"type": "json_object"})
        self.assertFalse(ok)
        self.assertIn("valid JSON", err)

    def test_json_schema_full_subset(self):
        rf = {"type": "json_schema",
              "json_schema": {"schema": {
                  "type": "object",
                  "required": ["name", "tags"],
                  "properties": {
                      "name": {"type": "string"},
                      "age": {"type": "integer"},
                      "tags": {"type": "array", "items": {"type": "string"}},
                      "role": {"type": "string", "enum": ["a", "b"]},
                  },
                  "additionalProperties": False}}}
        ok, err = server._validate_response_format(
            '{"name": "x", "tags": ["t"], "age": 3, "role": "a"}', rf)
        self.assertTrue(ok, err)
        for bad in ('{"tags": []}',
                    '{"name": 1, "tags": []}',
                    '{"name": "x", "tags": [1]}',
                    '{"name": "x", "tags": [], "role": "z"}',
                    '{"name": "x", "tags": [], "extra": 1}'):
            ok, err = server._validate_response_format(bad, rf)
            self.assertFalse(ok, bad)

    def test_bool_is_not_number(self):
        ok, _ = server._validate_json_schema(True, {"type": "number"})
        self.assertFalse(ok)
        ok, _ = server._validate_json_schema(True, {"type": "integer"})
        self.assertFalse(ok)
        ok, _ = server._validate_json_schema(True, {"type": "boolean"})
        self.assertTrue(ok)

    def test_unknown_format_type_passes(self):
        ok, _ = server._validate_response_format("anything", {"type": "text"})
        self.assertTrue(ok)


class TestContentParts(unittest.TestCase):
    def test_input_audio_parsed(self):
        texts, atts = server.extract_text_and_images([
            {"type": "input_audio",
             "input_audio": {"data": "QUJD", "format": "wav"}}])
        self.assertEqual(len(atts), 1)
        self.assertTrue(atts[0][1].startswith("data:audio/wav;base64,"))

    def test_refusal_kept_as_text(self):
        texts, atts = server.extract_text_and_images([
            {"type": "refusal", "refusal": "sorry"}])
        self.assertIn("sorry", texts)
        self.assertEqual(atts, [])

    def test_file_id_does_not_crash_or_attach(self):
        texts, atts = server.extract_text_and_images([
            {"type": "file", "file": {"file_id": "file-abc"}}])
        self.assertEqual(atts, [])


class TestErrorAndResponseShape(unittest.TestCase):
    def test_type_map(self):
        for before, after in (("invalid_request", "invalid_request_error"),
                              ("not_found", "not_found_error"),
                              ("backend_error", "api_error"),
                              ("rate_limit", "rate_limit_error")):
            self.assertEqual(server._openai_error("m", before)["error"]["type"], after)

    def test_param_code_present(self):
        err = server._openai_error("m", "invalid_request", param="n",
                                   code="unsupported_parameter")["error"]
        self.assertEqual(err["param"], "n")
        self.assertEqual(err["code"], "unsupported_parameter")

    def test_logprobs_null_in_builders(self):
        r1 = server.chat_completion_response("m", "hi")
        self.assertIn("logprobs", r1["choices"][0])
        self.assertIsNone(r1["choices"][0]["logprobs"])
        r2 = server.chat_completion_tool_response("m", None, [{"name": "f", "arguments": "{}"}])
        self.assertIsNone(r2["choices"][0]["logprobs"])
        self.assertEqual(r2["choices"][0]["finish_reason"], "tool_calls")

    def test_warnings_header(self):
        h = server._warnings_header(["temperature ignored: x"])
        self.assertTrue(h.startswith("299 "))
        self.assertIn("temperature", h)


class TestSessionCanon(unittest.TestCase):
    def test_developer_role_normalized_like_system(self):
        a = {"role": "developer", "content": "sys 2026-09-22T10:00:00Z"}
        b = {"role": "developer", "content": "sys 2026-09-23T11:00:00Z"}
        self.assertEqual(sessions.canon(a), sessions.canon(b))
        s1 = {"role": "system", "content": "sys 2026-09-22T10:00:00Z"}
        s2 = {"role": "system", "content": "sys 2026-09-23T11:00:00Z"}
        self.assertEqual(sessions.canon(s1), sessions.canon(s2))


if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python3
"""PLAN v10 acceptance tests (stdlib unittest, no backend, no network).

reasoning_effort was silently swallowed: Hermes sends a level on every
request believing it is honored. opencode serve 1.18.31 exposes
provider-defined per-call `variant` names (minimal/low/medium/high/xhigh
with reasoningEffort, confirmed in its /doc schema + /config/providers),
so known levels map to variants and unknown levels are a loud 400 —
same shape as the existing _compat_guard rejections.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server


class TestReasoningVariantMapping(unittest.TestCase):
    def test_known_levels_map(self):
        for level, want in (("low", "low"), ("medium", "medium"),
                            ("high", "high"), ("xhigh", "xhigh")):
            variant, err = server._reasoning_variant(level)
            self.assertIsNone(err, msg=level)
            self.assertEqual(variant, want, msg=level)

    def test_noop_levels_omit_variant(self):
        for level in (None, "", "none", "NONE", "minimal", "  Minimal "):
            variant, err = server._reasoning_variant(level)
            self.assertIsNone(err, msg=repr(level))
            self.assertIsNone(variant, msg=repr(level))

    def test_unknown_level_errors(self):
        # Plan test 4: Hermes-internal "ultra" (supposed to be clamped
        # before a custom endpoint) must not crash — it errors cleanly.
        for bad in ("ultra", "max", "off", 42, True, ["high"]):
            variant, err = server._reasoning_variant(bad)
            self.assertIsNone(variant, msg=repr(bad))
            self.assertTrue(err, msg=repr(bad))

    def test_dict_shape_unwrapped(self):
        # Responses-style {"effort": ...} mapping, defensively accepted.
        variant, err = server._reasoning_variant({"effort": "high"})
        self.assertIsNone(err)
        self.assertEqual(variant, "high")


class TestCompatGuardReasoning(unittest.TestCase):
    def test_known_level_passes_guard(self):
        # "high" is a real variant: passes the guard (returns None),
        # mapping happens downstream in do_POST.
        self.assertIsNone(
            server._compat_guard({"reasoning_effort": "high"}))

    def test_guard_passes_known_and_absent(self):
        for payload in ({}, {"reasoning_effort": None},
                        {"reasoning_effort": "none"},
                        {"reasoning_effort": "minimal"},
                        {"reasoning_effort": "low"},
                        {"reasoning_effort": "xhigh"}):
            self.assertIsNone(server._compat_guard(payload),
                              msg=repr(payload))

    def test_guard_rejects_unknown(self):
        for bad in ("ultra", "max", "banana"):
            res = server._compat_guard({"reasoning_effort": bad})
            self.assertIsNotNone(res, msg=repr(bad))
            code, body = res
            self.assertEqual(code, 400)
            self.assertEqual(body["error"].get("param"), "reasoning_effort")
            self.assertEqual(body["error"].get("code"), "unsupported_parameter")


class TestVariantBodyShape(unittest.TestCase):
    def test_variant_present_when_set(self):
        # Plan test 3 (unit half): the outgoing serve body actually differs.
        body = {"model": {"providerID": "opencode", "modelID": "m"},
                "parts": []}
        server._attach_variant(body, "xhigh")
        self.assertEqual(body.get("variant"), "xhigh")

    def test_variant_omitted_when_none(self):
        body = {"model": {"providerID": "opencode", "modelID": "m"},
                "parts": []}
        server._attach_variant(body, None)
        self.assertNotIn("variant", body)


if __name__ == "__main__":
    unittest.main(verbosity=2)

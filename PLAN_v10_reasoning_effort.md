# opencode-hermes-shim — v10: reasoning-effort variant selection (low/medium/high/xhigh)

Date: 2026-09-23. Short answer to "can I change the variant, or is it stuck
at default": **stuck at default today, and worse than that — it's silently
stuck**, which contradicts Hermes' own documented contract for this
parameter. Fixable, but there's a verification step that has to come first,
because it might turn out to not matter at all for this specific model.

## What's happening today

Confirmed by grep — `server.py` has zero handling of `reasoning_effort`
anywhere. It's not read, not rejected, not passed through. It falls into
the same bucket as any other unrecognized JSON field: silently ignored.

That matters more than most ignored fields because of how Hermes itself
behaves (`hermes-agent.nousresearch.com/docs/integrations/providers`):

> When no effort is configured at all, chat_completions requests carry
> `reasoning_effort: medium` — rather than leaving the endpoint's own
> default in charge... An endpoint that rejects the level answers with an
> HTTP 400 instead of Hermes silently downgrading it.

So two things are true simultaneously right now:
1. Hermes is almost certainly sending `reasoning_effort: medium` (or
   whatever `/reasoning <level>` was last set to) on every single request
   to the shim already, believing it's being honored.
2. The shim throws it away without a trace, and reports success. There's
   no 400, no downgrade notice — Hermes has no way to know the level was
   never applied. `/reasoning xhigh` looks like it worked and silently
   didn't.

This is the same category of gap the whole OpenAI-compat pass (v5) was
built to close — the goal there was "fail loudly instead of silently
ignoring," and this parameter slipped through that pass entirely.

## The real question: does the model even support this?

Before wiring anything, this needs an answer, because it changes what "the
fix" even is:

opencode's own message API supports selecting a named **variant** per
call — verified against opencode's OpenAPI schema (per a third-party
integration's documented findings): `POST /session/{id}/message` accepts a
per-call `variant` string alongside the existing `model` object. But
`variant` is not a raw reasoning-effort value — it's the *name* of a preset
block defined per-model in `opencode.json`:

```json
{
  "provider": {
    "opencode": {
      "models": {
        "muse-spark-1.3-contributor-free": {
          "variants": {
            "low":    { "reasoningEffort": "low" },
            "medium": { "reasoningEffort": "medium" },
            "high":   { "reasoningEffort": "high" },
            "xhigh":  { "reasoningEffort": "xhigh" }
          }
        }
      }
    }
  }
}
```

If `muse-spark-1.3-contributor-free` doesn't actually expose an adjustable
reasoning-effort field on the Zen side at all — plausible for a
free/contributor-tier model — then defining variants like this changes
nothing no matter how correctly it's wired; opencode would just pass a
field the model ignores. Check this **before** building anything:

```bash
curl -s http://127.0.0.1:4096/config/providers | python3 -m json.tool | grep -A20 muse-spark
```

Look for a `reasoning`/`variants`/`thinking` capability block on the model
entry itself (already-fetched by `_fetch_models`'s existing
`/config/providers` call — this is just inspecting the same response by
hand). No such block, or Zen's model card documents a fixed reasoning
behavior → skip straight to the "reject, don't silently ignore" fallback
below and stop there; the feature genuinely isn't available for this model,
and that's a fact worth having rather than a half-built variant table that
does nothing.

## If the model does support it: wiring plan

1. **Define the variants in `opencode.json`** for the model — one block
   per level Hermes might send (`none`, `minimal`, `low`, `medium`, `high`,
   `xhigh` — Hermes' vocabulary per `/reasoning <level>`), mapped to
   whatever field name Zen's `reasoningEffort`-equivalent actually expects.
   Use the earlier probe's output to get the field name right rather than
   guessing — different providers behind opencode use different key names
   (`reasoningEffort`, `thinking.type`, `reasoning_effort` at request-root)
   even though opencode's *config* schema is consistent.

2. **Read `reasoning_effort` from the incoming request body** in
   `do_POST`, alongside the other params already extracted (`stop`,
   `max_tokens`, `response_format`).

3. **Map Hermes' level to an opencode variant name** and attach it to the
   message body wherever `model`/`format` are already set:

   ```python
   _REASONING_LEVELS = {"none", "minimal", "low", "medium", "high", "xhigh"}
   reasoning_effort = body.get("reasoning_effort")
   variant = None
   if reasoning_effort in _REASONING_LEVELS:
       # none/minimal collapse to omitting the variant (model's own zero-effort
       # default) unless a dedicated "none" variant is defined in opencode.json
       variant = reasoning_effort if reasoning_effort not in ("none", "minimal") else None
   ```

   Then thread `variant` into the message body the same way
   `model_override`/`serve_format` are already threaded through `_run_once`
   → `_relay_turn` → the actual `POST /session/:id/message` call — same
   pattern, one more optional field.

4. **Unknown/unsupported levels**: if Hermes ever sends something outside
   the six documented values (shouldn't happen, but don't trust it blindly),
   or a level with no matching variant defined in `opencode.json`, treat it
   the same as the fallback case below rather than silently dropping it.

## Fallback (do this regardless, and immediately — it's cheap)

Whether or not the model turns out to support adjustable effort, stop
silently swallowing the parameter today. Per Hermes' own documented
expectation ("rejects... instead of silently downgrading"), the honest
shim behavior for a level it can't actually honor is a `400`, not a quiet
no-op — consistent with how `logprobs`/`seed`/`logit_bias` are already
handled in `_compat_guard`:

```python
# Until variants are wired (or if the model turns out not to support them):
# don't let Hermes believe reasoning_effort was honored when it wasn't.
_reasoning = body.get("reasoning_effort")
if _reasoning not in (None, "none", "minimal"):
    # "none"/"minimal" are indistinguishable from just not asking for
    # extra reasoning, so let those through as a no-op; anything asking
    # for real effort (low/medium/high/xhigh) that the shim can't yet
    # deliver should say so.
    self._json(400, _openai_error(
        f"reasoning_effort='{_reasoning}' is not yet wired to opencode's "
        f"variant system for this model — see PLAN_v10.",
        "invalid_request", param="reasoning_effort", code="unsupported_parameter"))
    return
```

This is the same shape as the existing `_compat_guard` rejections — put it
there rather than as a separate check, so it shows up in the same place
future-you already knows to look. Once the real wiring from the section
above lands, delete this block (or narrow it to only reject levels that
genuinely have no variant defined, so newly-added levels don't need a code
change to stop being rejected).

## Tests to add

### Probe result (recorded 2026-09-24, serve 1.18.31)

`GET /config/providers` shows `muse-spark-1.3-contributor-free` (and 1.2)
with `capabilities.reasoning: true` and provider-defined `variants:
{minimal, low, medium, high, xhigh}`, each carrying that level's
`reasoningEffort`. opencode's own `/doc` schema confirms
`POST /session/:id/prompt_async` (and `/message`) accept a per-call
`variant: string` next to `model`/`agent`/`parts`. So the model DOES
support adjustable effort → the wiring path below is the real fix (not
the permanent-reject fallback), and no `opencode.json` variant block is
needed — the names are already defined server-side.

1. **Capability probe result recorded**: whatever the `/config/providers`
   check above shows for `muse-spark-1.3-contributor-free`, write the
   answer here as a comment once known — this determines which of the two
   paths above is real work vs. already done.
2. **Reject-until-wired**: `reasoning_effort: "high"` with no variant
   support yet → `400`, `param: "reasoning_effort"`. `reasoning_effort:
   "none"` or `"minimal"` → passes through as a no-op, `200` as normal.
3. **If wired**: a request with `reasoning_effort: "low"` vs one with
   `"xhigh"` against the same prompt → confirm the `variant` field actually
   differs in the outgoing `POST /session/:id/message` body (mock the
   serve call and assert on the payload — cheaper and more reliable than
   trying to infer effort from output length/quality).
4. **Unknown level**: `reasoning_effort: "ultra"` (a Hermes-internal
   level that's supposed to get clamped before reaching a custom endpoint,
   per the docs) — confirm the shim doesn't crash on an unrecognized value
   either way, whichever path is active.

## Rollout

1. Run the capability probe first — this is a `curl`, not a code change,
   and it decides which of the next two steps is real.
2. Ship the reject-until-wired guard immediately regardless of the probe
   result — it's small, isolated, and stops the current silent-no-op
   behavior today rather than waiting on the full variant wiring.
3. If the probe confirms real support: define variants in `opencode.json`,
   wire `reasoning_effort` → `variant` through the existing
   model_override/serve_format threading pattern, then narrow or remove
   the reject-guard for the levels now actually handled.
4. If the probe shows no support: leave the reject-guard as the permanent
   answer for this model, and note it plainly rather than revisiting this
   later assuming it was never tried — check `/debug/stats` occasionally
   for how often `reasoning_effort` requests are actually being rejected,
   since a persistently low rate might mean this was never worth building
   in the first place, and a high rate is the signal to prioritize it.

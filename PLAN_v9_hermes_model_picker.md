# opencode-hermes-shim — v9: making Hermes' /model picker actually work

Date: 2026-09-23. This one isn't a `server.py` bug — the shim's side of this
is already solid. Confirmed by reading `_fetch_models`/`_ranked_models`
directly: discovery merges two sources (`GET /config/providers` +
`opencode models opencode` CLI), dedupes, ranks by power, and only falls
back to a single-model list if *both* sources genuinely fail. That's not
where "the picker still sees just one" comes from.

## Root cause

Per Hermes' own docs (`hermes-agent.nousresearch.com/docs/reference/slash-commands`,
`cli-commands`), **`/model` with no arguments only shows providers already
registered in `~/.hermes/config.yaml`** — it does not add new providers or
pull a fresh model list on its own:

> If you've only configured OpenRouter, `/model` will only show OpenRouter
> models. To add another provider, exit your session and run `hermes model`
> from the terminal.

The project's original config (from the very first `PLAN.md`) set:

```yaml
model:
  default: muse-spark-1.3-contributor-free
  provider: custom
  base_url: http://127.0.0.1:8000/v1
```

This is enough to make the shim **the active model** — which is why chat
works at all — but it's just a single `model.default` value, not a
provider *inventory*. There's no `providers:` (or `custom_providers:`) block
telling Hermes "this custom endpoint has more than one model on it." With
only the active-model fields set, `/model` has exactly one thing to show:
whatever `model.default` currently is. That matches the transcript exactly
— the shim's `/v1/models` was already returning a real multi-model list at
the time this was being debugged, but the debugging session checked
opencode's own config (`/home/mitansh/.config/opencode/opencode.jsonc` —
which controls what model opencode's *backend* talks to) rather than
Hermes' `~/.hermes/config.yaml` — which is what actually governs the
picker. Right side of the wire, wrong config file.

Two secondary gotchas worth knowing before touching config:

- **Stale cache**: even a registered custom provider's model list isn't
  necessarily live — `/model custom --refresh` explicitly exists to
  re-fetch it. If the shim's `/v1/models` changed since registration
  (e.g. after the recent multi-source discovery work), a plain `/model`
  can still show an outdated list until refreshed once.
- **Credential gating**: Hermes hides "API-key-style" providers from
  `/model` unless a configured key env var exists — even a placeholder.
  For a genuinely keyless local endpoint, the documented pattern is a
  non-empty dummy value (`api_key: local`), not an empty string or the key
  omitted entirely, or the provider may not show up in the picker at all
  regardless of how the rest of the config looks.

## Fix

### Recommended: `hermes model` from the terminal (not `/model` in-session)

This is Hermes' own guided path and sidesteps guessing at config schema by
hand — the wizard "can add new providers... and configure endpoints"
directly:

```bash
# from a terminal, OUTSIDE any active Hermes session
hermes model
```

Walk it through adding a **custom endpoint**: base URL
`http://127.0.0.1:8000/v1`, a placeholder API key (`local` or `dummy` — the
shim ignores it entirely, loopback-only + no auth is the actual guarantee,
per the project's original design constraint), and let it probe
`/v1/models` to populate the picker. Give it a name (e.g. `opencode-shim`)
so it's addressable as `/model opencode-shim:<model-id>` later.

### Reference: manual `config.yaml` (if the wizard doesn't detect a keyless local endpoint cleanly)

Hermes' docs show this config key under two names across different doc
pages — `providers:` (singular block) and `custom_providers:` (list form).
Rather than guess which your installed version expects, check first:

```bash
hermes config path      # find the real config.yaml
hermes config check     # flags stale/unrecognized keys for your version
```

Reference shape (list form, more commonly documented for self-hosted/local
endpoints):

```yaml
model:
  provider: custom
  default: muse-spark-1.3-contributor-free   # must be the raw id exactly as
                                              # /v1/models returns it — no
                                              # "custom/" or "opencode/" prefix
  base_url: http://127.0.0.1:8000/v1
  api_key: local                             # non-empty placeholder; shim ignores it
  context_length: 1048576                    # matches the 1M window shown in the TUI image

custom_providers:
  - name: "opencode-shim"
    base_url: "http://127.0.0.1:8000/v1"
    api_key: local
    # models: block is optional — Hermes can discover from /v1/models
    # directly; only pin explicit entries here if you want fixed
    # context_length per model rather than relying on discovery.
```

### After setup: using `/model` mid-session

Per Hermes' documented syntax, once the shim is registered as a named
custom provider:

```
/model                                  # interactive picker — provider, then model, fuzzy-filterable
/model custom                           # auto-detect/refresh from the currently active custom endpoint
/model custom:muse-spark-1.3-...        # jump straight to a specific model on it
/model opencode-shim:muse-spark-1.3-... # same, addressed by the name given during setup
/model custom --refresh                 # force re-fetch if the list looks stale
/model <model-id> --global              # persist the choice as the new session default
```

Plain `/model <name>` is session-only unless
`model.persist_switch_by_default: true` is set — worth knowing if a
switch mid-chat isn't surviving `/new` or a restart the way expected.

## What NOT to do

Don't have the model debug this by curling its own shim and reading
opencode's config again — that's the exact path the earlier transcript
took, and it never touches the file that actually controls this
(`~/.hermes/config.yaml`'s provider inventory). It also runs directly into
the v8 duplicate-tool-narration bug on any turn involving a few `curl`/`cat`
calls in a row, so land that fix first if more agentic self-diagnosis is
needed here. This one's a five-minute manual `hermes model` run, not a
debugging task for the agent itself.

## Verification

1. `curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool` — confirm
   this returns multiple entries today (per the code review above, it
   should already, assuming `opencode serve` and the `opencode` CLI are
   both reachable when `_fetch_models` runs). If it doesn't, the discovery
   layer needs its own look — check the `[models] ... discovery failed`
   log lines for which of the two sources is failing.
2. Run `hermes model` (or edit config + `hermes config check`), register
   the endpoint.
3. Inside a session, `/model` with no args — confirm the picker now lists
   more than one model, matching step 1's output.
4. `/model custom:<some-other-model-id>` — confirm it actually switches
   (check `/status` or the next turn's model field) and that a subsequent
   turn correctly routes through the shim's fallback-chain/model_override
   plumbing to the newly-selected model.
5. If the list still shows stale/single after registration, try
   `/model custom --refresh` before assuming it's broken again — this is
   the documented cache-refresh path, not a bug report waiting to happen.

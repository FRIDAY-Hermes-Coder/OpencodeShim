#!/usr/bin/env python3
"""Keyless local OpenAI-compatible shim for Hermes -> on-device opencode.

Hermes (keyless, dummy Bearer) -> http://127.0.0.1:8000/v1 -> warm `opencode serve`
with opencode/muse-spark-1.3-contributor-free (Zen free, auth stored in opencode).

Stdlib only. Binds 127.0.0.1 (localhost-only, no auth needed).
Media (image/video/pdf/audio) is normalized to data: URLs because serve accepts
data: and file:// but rejects bare paths (500) and http(s) URLs (400).
"""
import base64
import collections
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sessions as L1

HOST = os.environ.get("SHIM_HOST", "127.0.0.1")
PORT = int(os.environ.get("SHIM_PORT", "8000"))
MODEL_ID = os.environ.get("SHIM_MODEL_ID", "muse-spark-1.3-contributor-free")
SHIM_OPENCODE_MODEL = os.environ.get("SHIM_OPENCODE_MODEL", "opencode/muse-spark-1.3-contributor-free")
OPENCODE_MODEL = SHIM_OPENCODE_MODEL

# Dynamic model resolution: resolve client-requested model to an opencode model.
_model_resolution_cache = {}
def _resolve_model(req_model):
    """Map a client-requested model name to an available opencode model.

    Returns the modelID (short, without provider prefix) to use for the
    opencode request body's modelID field. Provider is always derived as
    ``opencode`` or the prefix before ``/`` if present.
    """
    if not req_model:
        # Use configured default (strip provider prefix if present).
        return SHIM_OPENCODE_MODEL.split("/")[-1] if "/" in SHIM_OPENCODE_MODEL else SHIM_OPENCODE_MODEL
    # If client asks for the configured SHORT id, map to its opencode model.
    if req_model == MODEL_ID:
        return SHIM_OPENCODE_MODEL.split("/")[-1] if "/" in SHIM_OPENCODE_MODEL else SHIM_OPENCODE_MODEL
    # Check discovered list — handle both "provider/model" and short names.
    try:
        available = _fetch_models()
    except Exception:
        available = []
    # Build lookup for both full and short forms.
    short_to_full = {}
    full_ids = set()
    for m in available:
        fid = m.get("id", "")
        full_ids.add(fid)
        if "/" in fid:
            short_to_full[fid.split("/")[-1]] = fid
        short_to_full[fid] = fid
    if req_model in full_ids:
        return req_model.split("/")[-1] if "/" in req_model else req_model
    if req_model in short_to_full:
        fid = short_to_full[req_model]
        return fid.split("/")[-1] if "/" in fid else fid
    # Suffix match (e.g. client sends short, available is provider/short)
    for fid in full_ids:
        if fid.endswith("/" + req_model) or fid == req_model:
            return fid.split("/")[-1] if "/" in fid else fid
    # Unknown model — fall back to default rather than failing the turn.
    return SHIM_OPENCODE_MODEL.split("/")[-1] if "/" in SHIM_OPENCODE_MODEL else SHIM_OPENCODE_MODEL
OPENCODE_BIN = os.environ.get("SHIM_OPENCODE_BIN", "/home/mitansh/.opencode/bin/opencode")
SERVE_URL = os.environ.get("SHIM_SERVE_URL", "http://127.0.0.1:4096").rstrip("/")
WORKDIR = os.environ.get("SHIM_WORKDIR", "/home/mitansh/hermesworkspace")
TIMEOUT = int(os.environ.get("SHIM_TIMEOUT", "300"))
MAX_CONCURRENT = int(os.environ.get("SHIM_MAX_CONCURRENT", "4"))
IMAGE_TIMEOUT = int(os.environ.get("SHIM_IMAGE_TIMEOUT", "20"))

_sem = threading.Semaphore(MAX_CONCURRENT)

# --- Phase 1 L1 Session Manager (§2 plan_enhanced.md) ---
SHIM_REPAIR_TURNS = int(os.environ.get("SHIM_REPAIR_TURNS", "1"))  # fence repair attempts (§3.3)
SHIM_SESSIONS = os.environ.get("SHIM_SESSIONS", "1") == "1"  # kill switch: 0 = v2.2 flatten
SHIM_SESSION_STORE = os.environ.get("SHIM_SESSION_STORE", "") or None
SHIM_MAX_SESSIONS = int(os.environ.get("SHIM_MAX_SESSIONS", "200"))
SHIM_SESSION_TTL = int(os.environ.get("SHIM_SESSION_TTL", str(7 * 24 * 3600)))
SHIM_IDEMPOTENT_TTL = int(os.environ.get("SHIM_IDEMPOTENT_TTL", "60"))
SHIM_SESSION_LOCK_TIMEOUT = int(os.environ.get("SHIM_SESSION_LOCK_TIMEOUT", "300"))
SHIM_MODELS_CACHE_TTL = int(os.environ.get("SHIM_MODELS_CACHE_TTL", "300"))  # seconds
SHIM_SESSION_MAX_TURNS = int(os.environ.get("SHIM_SESSION_MAX_TURNS", "80"))
_models_cache = {"models": [], "updated": 0}
_models_cache_lock = threading.Lock()
SHIM_MODEL_FALLBACK = os.environ.get("SHIM_MODEL_FALLBACK", "1") == "1"
SHIM_MODEL_FALLBACK_RETRIES = int(os.environ.get("SHIM_MODEL_FALLBACK_RETRIES", "3"))

# --- OpenAI compat (PLAN v5) ---
# Error type alignment (§E): OpenAI uses the `_error`-suffixed values. The
# shim historically emitted bare names; map centrally so new code can pass
# either form.
_TYPE_MAP = {"invalid_request": "invalid_request_error",
             "not_found": "not_found_error",
             "backend_error": "api_error",
             "rate_limit": "rate_limit_error"}


def _openai_error(message, etype="invalid_request", param=None, code=None):
    """Build an OpenAI-shaped {"error": {...}} body with aligned type + param/code."""
    t = _TYPE_MAP.get(etype, etype)
    err = {"message": str(message)[:4000], "type": t}
    if param is not None:
        err["param"] = param
    if code is not None:
        err["code"] = code
    return {"error": err}


# Params the opencode/Zen backend cannot provide through the session
# abstraction (token-level logprobs / sampling control). Requesting them is
# a loud 400, never a silent ignore (§A.7).
_UNSUPPORTED_PARAMS = ("logprobs", "top_logprobs", "seed", "logit_bias")


def _compat_guard(body):
    """Phase-1 guards (§A.3/A.7). Returns (code, error_body) or None.

    Runs before any opencode call so silent-wrongness becomes a loud 400.
    Falsy/disabled values (logprobs:false, top_logprobs:0, logit_bias:{})
    are harmless and allowed — only values that actually request the
    unsupported capability are rejected.
    """
    _n = body.get("n", 1)
    if isinstance(_n, (int, float)) and _n > 1:
        return (400, _openai_error(
            "n>1 is not supported by this shim (opencode serve produces one "
            "completion per turn)", "invalid_request",
            param="n", code="unsupported_parameter"))
    for _p in _UNSUPPORTED_PARAMS:
        _v = body.get(_p)
        if _p == "seed":
            if _v is None:  # any explicit seed requests determinism
                continue
        elif not _v:  # logprobs/top_logprobs/logit_bias: falsy = disabled
            continue
        return (400, _openai_error(
            f"'{_p}' is not supported by this shim (opencode/Zen backend "
            "does not expose it)", "invalid_request",
            param=_p, code="unsupported_parameter"))
    return None


def _limit_parallel_calls(calls, parallel_tool_calls):
    """Honor parallel_tool_calls:false (§A.4) — keep only the first call."""
    if parallel_tool_calls is False and calls and len(calls) > 1:
        return calls[:1]
    return calls


def _apply_stop(text, stop):
    """Truncate text at the first occurrence of any stop sequence (§A.1)."""
    if not stop or not text:
        return text, False
    seqs = [stop] if isinstance(stop, str) else list(stop or [])
    cut_at = None
    for s in seqs:
        if s and isinstance(s, str):
            i = text.find(s)
            if i != -1 and (cut_at is None or i < cut_at):
                cut_at = i
    if cut_at is None:
        return text, False
    return text[:cut_at], True


def _apply_max_tokens(text, limit, chars_per_token=4):
    """Post-hoc token-budget cap (§A.2). Estimate via ~4 chars/token, same
    heuristic as the Bug-E usage fallback. Returns (text, truncated)."""
    if not limit or not text:
        return text, False
    try:
        n = int(limit)
    except Exception:
        return text, False
    if n <= 0:
        return text, False
    budget = n * chars_per_token
    if len(text) <= budget:
        return text, False
    return text[:budget], True


def _response_format_instruction(rf):
    """Prompt instruction enforcing response_format (§A.6). Returns "" if none."""
    if not isinstance(rf, dict):
        return ""
    t = rf.get("type")
    if t == "json_object":
        return ("\n\n[Response format: respond with a single valid JSON object "
                "and nothing else. No prose, no fences, no commentary.]")
    if t == "json_schema":
        js = rf.get("json_schema") or {}
        name = js.get("name") if isinstance(js, dict) else None
        schema = (js.get("schema") if isinstance(js, dict) else None) or rf.get("schema")
        try:
            schema_s = json.dumps(schema)[:4000] if schema is not None else ""
        except Exception:
            schema_s = str(schema)[:4000] if schema is not None else ""
        nm = f" named '{name}'" if name else ""
        return ("\n\n[Response format: respond with a single valid JSON value"
                f"{nm} conforming strictly to this JSON Schema and nothing else: "
                f"{schema_s}. No prose, no fences, no commentary.]")
    return ""


def _validate_json_schema(obj, schema, _path="$"):
    """Subset JSON Schema validator (§A.6, json_schema mode).

    Supports: type (object/array/string/number/integer/boolean/null),
    required, properties (recursive), additionalProperties (bool),
    items (single schema), enum, const. Unknown keywords are ignored.
    Returns (ok, err).
    """
    if not isinstance(schema, dict):
        return True, None
    if "const" in schema:
        if obj != schema["const"]:
            return False, f"{_path}: value does not match const"
    if "enum" in schema:
        try:
            if obj not in schema["enum"]:
                return False, f"{_path}: value not in enum"
        except Exception:
            return False, f"{_path}: value not in enum"
    t = schema.get("type")
    if t is not None:
        _types = [t] if isinstance(t, str) else (t if isinstance(t, list) else None)
        if _types is not None:
            _ok = False
            for _t in _types:
                if _t == "object" and isinstance(obj, dict):
                    _ok = True
                elif _t == "array" and isinstance(obj, list):
                    _ok = True
                elif _t == "string" and isinstance(obj, str):
                    _ok = True
                elif _t == "number" and isinstance(obj, (int, float)) and not isinstance(obj, bool):
                    _ok = True
                elif _t == "integer" and isinstance(obj, int) and not isinstance(obj, bool):
                    _ok = True
                elif _t == "boolean" and isinstance(obj, bool):
                    _ok = True
                elif _t == "null" and obj is None:
                    _ok = True
            if not _ok:
                return False, f"{_path}: expected type {_types}, got {type(obj).__name__}"
    if isinstance(obj, dict):
        for _k in schema.get("required") or []:
            if isinstance(_k, str) and _k not in obj:
                return False, f"{_path}: missing required key '{_k}'"
        _props = schema.get("properties") or {}
        if isinstance(_props, dict):
            for _k, _sub in _props.items():
                if _k in obj:
                    _ok, _e = _validate_json_schema(obj[_k], _sub, f"{_path}.{_k}")
                    if not _ok:
                        return False, _e
        if schema.get("additionalProperties") is False and isinstance(_props, dict):
            _extra = [k for k in obj if k not in _props]
            if _extra:
                return False, f"{_path}: additional properties not allowed: {sorted(_extra)[:5]}"
    if isinstance(obj, list) and isinstance(schema.get("items"), dict):
        for _i, _item in enumerate(obj):
            _ok, _e = _validate_json_schema(_item, schema["items"], f"{_path}[{_i}]")
            if not _ok:
                return False, _e
    return True, None


def _validate_response_format(text, rf):
    """Check reconciled text against response_format. Returns (ok, err)."""
    if not isinstance(rf, dict):
        return True, None
    t = rf.get("type")
    if t not in ("json_object", "json_schema"):
        return True, None
    try:
        obj = json.loads(text or "")
    except Exception as e:
        return False, f"response is not valid JSON ({e})"
    if t == "json_schema":
        js = rf.get("json_schema") or {}
        schema = (js.get("schema") if isinstance(js, dict) else None) or rf.get("schema")
        if isinstance(schema, dict):
            return _validate_json_schema(obj, schema)
    return True, None


def _serve_format_for_response_format(rf):
    """Native opencode `format` body for response_format (§A.6) — DISABLED.

    Probed 2026-09-22 against serve 1.18.31: a format-bearing
    POST /session/:id/prompt_async returns 204 and the turn runs, but the
    very next GET /session/:id/message (the reconciliation step every turn
    depends on) fails 400 "Expected OutputFormatJsonSchema, got {...}".
    So native format breaks 100% of turns that use it. This stays a
    single kill-switch returning None — prompt-injection + repair-turn is
    the sole enforcement path until a fixed serve version is confirmed
    with the same direct probe. Re-enable by restoring the json_schema
    passthrough below."""
    return None

# Power ranking — most powerful first. Clients see this order in /v1/models.
# Override via SHIM_MODEL_ORDER="model-a,model-b,..." (comma-separated, case-insensitive substrings).
_MODEL_POWER_ORDER = [s.strip().lower() for s in os.environ.get("SHIM_MODEL_ORDER", "").split(",") if s.strip()] or [
    "claude-4.5-opus", "claude-4-opus", "claude-4-sonnet", "claude-4",
    "claude-3.5-sonnet", "claude-3-opus", "claude-3-sonnet", "claude-3-haiku",
    "gpt-4o", "gpt-4-turbo", "gpt-4", "o1", "o3",
    "gemini-2.0-pro", "gemini-2.0-flash", "gemini-1.5-pro", "gemini",
    "deepseek-r1", "deepseek-v3", "deepseek",
    "llama-3.3-70b", "llama-3.1-405b", "llama-3.1-70b", "llama-3.1-8b", "llama",
    "qwen2.5-72b", "qwen2.5", "qwen",
    "muse-spark", "muse",
]

def _model_power_score(mid):
    """Lower score = more powerful. Scores by earliest substring match in _MODEL_POWER_ORDER."""
    low = (mid or "").lower()
    for idx, pat in enumerate(_MODEL_POWER_ORDER):
        if pat and pat in low:
            return idx
    # Heuristic: larger param count ~ more powerful (405b > 70b > 8b)
    m = re.search(r"(\d+)b", low)
    if m:
        try:
            # invert: bigger b -> lower (better) offset after unknown bucket
            return 900 - int(m.group(1))
        except Exception:
            pass
    return 999

def _ranked_models():
    """All discovered models sorted most-powerful -> least. Uses cached fetch."""
    try:
        models = _fetch_models()
    except Exception:
        models = []
    return sorted(models, key=lambda m: (_model_power_score(m.get("id","")), m.get("id","")))

def _fallback_chain(req_model):
    """Ordered list of model IDs to try: honors explicit request first, then
    all ranked most-powerful -> least. Keeps *all* discovered models (never
    filters). When the request is the default (MODEL_ID or empty), the global
    power order wins so the most powerful model is first.
    """
    ranked = _ranked_models()
    ranked_ids = [m["id"] for m in ranked]
    # Default / empty request -> pure power order (most powerful first)
    if not req_model or req_model == MODEL_ID:
        chain = list(ranked_ids)
        if MODEL_ID not in chain:
            chain.append(MODEL_ID)
        short_def = SHIM_OPENCODE_MODEL.split("/")[-1] if "/" in SHIM_OPENCODE_MODEL else SHIM_OPENCODE_MODEL
        if short_def not in chain:
            chain.append(short_def)
        return chain
    # Explicit request -> honor it first, then power-ordered rest
    chain = []
    seen = set()
    chain.append(req_model)
    seen.add(req_model)
    if "/" in req_model:
        seen.add(req_model.split("/")[-1])
    else:
        for fid in ranked_ids:
            if fid.split("/")[-1] == req_model:
                seen.add(fid)
                break
    for mid in ranked_ids:
        if mid in seen or mid.split("/")[-1] in seen:
            continue
        chain.append(mid)
        seen.add(mid)
        seen.add(mid.split("/")[-1] if "/" in mid else mid)
    return chain

# --- Phase 2 profiles (§3 plan_enhanced.md) ---
SHIM_AGENT_LLM = os.environ.get("SHIM_AGENT_LLM", "hermes")       # Profile A: plain model
SHIM_AGENT_TASK = os.environ.get("SHIM_AGENT_TASK", "hermes-agent")  # Profile B: agentic worker
SHIM_PROFILE = os.environ.get("SHIM_PROFILE", "auto")  # auto | A | B
# Delta tool-result cap (per-result, sent once — no quadratic blowup).
SHIM_DELTA_TOOL_BYTES = int(os.environ.get("SHIM_DELTA_TOOL_BYTES", "24000"))

_store = None
_store_guard = threading.Lock()


def get_store():
    global _store
    with _store_guard:
        if _store is None:
            _store = L1.SessionStore(path=SHIM_SESSION_STORE or L1.default_store_path(),
                                     max_sessions=SHIM_MAX_SESSIONS, ttl=SHIM_SESSION_TTL)
        return _store


def _system_raw_of(messages):
    if messages and isinstance(messages[0], dict) and messages[0].get("role") in ("system", "developer"):
        c = messages[0].get("content")
        return c if isinstance(c, str) else json.dumps(c or "")
    return ""


def _delta_text_and_atts(m, session_id):
    """Render one Hermes delta message -> (text, atts_raw)."""
    if isinstance(m, str):
        m = {"role": "user", "content": m}
    if not isinstance(m, dict):
        return str(m), []
    role = m.get("role") or "user"
    text, atts = extract_text_and_images(m.get("content"))
    if role == "tool" or m.get("tool_call_id"):
        call_id = m.get("tool_call_id") or ""
        name = None
        if session_id:
            try:
                name = get_store().tool_name(session_id, call_id)
            except Exception:
                name = None
        name = name or m.get("name") or call_id or "tool"
        if len(text) > SHIM_DELTA_TOOL_BYTES:
            text = text[:SHIM_DELTA_TOOL_BYTES] + "...(truncated)"
        return f"[tool result: {name} ({call_id})]\n{text}", atts
    if role == "assistant" and m.get("tool_calls"):
        try:
            summ = []
            for tc in m["tool_calls"]:
                fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                summ.append(f"{tc.get('id', '?')}={fn.get('name', '?')}({str(fn.get('arguments', ''))[:800]})")
            text += "\n[assistant tool_calls in history: " + "; ".join(summ) + "]"
        except Exception:
            text += f"\n[tool_calls in history: {json.dumps(m.get('tool_calls'))[:2000]}]"
        return text, atts
    if role == "system":
        return text, atts
    return text, atts


def _tools_hash(tools):
    try:
        return hashlib.sha256(json.dumps(tools, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    except Exception:
        return hashlib.sha256(str(tools).encode()).hexdigest()


def _offered_names(tools):
    return tool_names(tools)


def select_profile(tools, header_override=None, model=None):
    """Profile A (LLM) when Hermes offers tools; B (agent) otherwise. §3.2."""
    ov = (header_override or "").strip().upper()
    if ov in ("A", "LLM"):
        return "A"
    if ov in ("B", "AGENT"):
        return "B"
    if SHIM_PROFILE in ("A", "B"):
        return SHIM_PROFILE
    m = (model or "")
    if m.endswith("-agent"):
        return "B"
    if m.endswith("-llm"):
        return "A"
    return "A" if tools else "B"


def profile_agent(profile):
    return SHIM_AGENT_LLM if profile == "A" else SHIM_AGENT_TASK


def _make_session_carryover(messages, rec):
    """One-paragraph carryover for proactive max-turns fork.

    Keeps recent context so new session does not lose recall, without relying
    on opencode's own compaction which silently truncates.
    """
    turns = rec.get("turns", 0)
    idx = rec.get("idx", 0)
    # recent messages compact rendering
    recent = messages[-12:] if len(messages) > 12 else messages
    lines = []
    for m in recent:
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "user")
        txt, _ = extract_text_and_images(m.get("content"))
        txt = (txt or "").strip().replace("\n", " ")
        if not txt:
            continue
        # keep per-message truncated to avoid bloat
        lines.append(f"{role[:4]}: {txt[:300]}")
    body = " | ".join(lines[-6:]) if lines else "(no recent text)"
    return (f"[system carryover: prior session {idx+1} msgs / {turns} turns "
            f"exceeded SHIM_SESSION_MAX_TURNS={SHIM_SESSION_MAX_TURNS}. "
            f"Forked to new session to avoid opencode compaction. "
            f"Recent context (truncated, last {len(lines)} msgs): {body[:2000]}]")


def build_tools_compact(tools, tool_choice):
    """One-liner used when schemas were already sent once (§3.1: emit nothing extra)."""
    names = _offered_names(tools)
    if tool_choice == "none":
        return ("\n\n[Tools defined but tool_choice='none': answer directly in plain "
                "text. Do NOT emit tool calls.]")
    if tool_choice == "required":
        return ("\n\n[Tools available (schemas sent earlier). You MUST call a tool "
                "via the ```hermes-toolcalls fence; do not answer directly.]")
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function", {})
        want = fn.get("name") if isinstance(fn, dict) else None
        if want:
            return (f"\n\n[Tools available (schemas sent earlier). You MUST call the tool "
                    f"named '{want}' via the ```hermes-toolcalls fence.]")
    return ""


def build_tools_diff(tools, old_names):
    """Compact diff block when the tool set changed mid-session."""
    new_names = _offered_names(tools)
    old = set(old_names or [])
    new = set(new_names)
    added = [n for n in new_names if n not in old]
    removed = sorted(old - new)
    schemas = []
    by_name = {}
    for t in tools or []:
        if isinstance(t, dict):
            fn = t.get("function")
            if isinstance(fn, dict) and fn.get("name"):
                by_name[str(fn["name"])] = t
            elif isinstance(t.get("name"), str):
                by_name[t["name"]] = t
    for n in added:
        try:
            schemas.append(json.dumps(by_name.get(n, {}))[:4000])
        except Exception:
            schemas.append(str(by_name.get(n, ""))[:4000])
    block = ("\n\n[tools changed] added: " + (", ".join(added) or "none") +
             "  removed: " + (", ".join(removed) or "none") +
             ". Use the ```hermes-toolcalls fence with these names: " +
             ", ".join(new_names[:100]) + ".]")
    if schemas:
        block += "\nNew tool schemas: [" + ", ".join(schemas) + "]"
    return block, len(added)


def _build_delta_prompt(delta, session_id, system_update, tools_section):
    """Prompt for a session hit: only new turns + optional system update."""
    parts, atts = [], []
    if system_update:
        parts.append(system_update)
    for m in delta:
        t, a = _delta_text_and_atts(m, session_id)
        if t.strip():
            parts.append(t)
        atts.extend(a)
    prompt = "\n\n".join(parts).strip() or "Continue."
    if atts:
        kinds = sorted({("image" if mm.startswith("image/") else "video" if mm.startswith("video/")
                         else "audio" if mm.startswith("audio/") else "pdf") for mm, _r, _f in atts})
        prompt += f"\n\n[{len(atts)} attached file(s) follow as vision/file input ({', '.join(kinds)}). Inspect them via the attached files.]"
    if tools_section:
        prompt += tools_section
    elif tools_section is None:
        prompt += "\n\n[Context: no usable Hermes tool names found; answer in plain text.]"
    prompt += build_search_discipline()
    return prompt, atts


def _run_session_turn(sid, prompt, file_parts, agent=None, model_override=None,
                      serve_format=None):
    """POST one turn to an existing opencode session (no create/delete)."""
    ok, health = _serve_call("GET", "/global/health", timeout=15)
    if not ok:
        return False, "", f"opencode serve down ({health}). Check `systemctl --user status opencode-serve`."
    parts = [{"type": "text", "text": prompt}] + list(file_parts or [])
    _eff = model_override if model_override is not None else MODEL_ID
    resolved_model = _resolve_model(_eff)
    body = {"model": {"providerID": "opencode", "modelID": resolved_model},
            "parts": parts}
    if agent:
        body["agent"] = agent
    if serve_format is not None:
        body["format"] = serve_format
    ok, resp = _serve_call(
        "POST", f"/session/{sid}/message", body,
        timeout=TIMEOUT,
    )
    if not ok:
        # Stale mapping (serve pruned the session?) -> signal caller to re-create.
        if "404" in str(resp):
            return False, "", f"SESSION_GONE: {resp}"
        return False, "", str(resp)
    info = resp.get("info", {}) if isinstance(resp, dict) else {}
    info_err = info.get("error") if isinstance(info, dict) else None
    if info_err:
        try:
            detail = json.dumps(info_err)[:2000]
        except Exception:
            detail = str(info_err)[:2000]
        return False, "", f"serve model error: {detail}"
    texts = [p.get("text", "") for p in resp.get("parts", []) if p.get("type") == "text"]
    out = "\n".join(t for t in texts if t).strip()
    return True, out, ""


# --- Phase 3 streaming relay (§4 plan_enhanced.md) ---
SHIM_IDLE_TIMEOUT = int(os.environ.get("SHIM_IDLE_TIMEOUT", "120"))  # s since last event
SHIM_HEARTBEAT = int(os.environ.get("SHIM_HEARTBEAT", "10"))  # SSE ping interval (stream path)
SHIM_NARRATE_TOOLS = os.environ.get("SHIM_NARRATE_TOOLS", "0") == "1"


def _prompt_async(sid, body):
    """POST /session/:id/prompt_async (204, empty body)."""
    import http.client
    from urllib.parse import urlparse as _up
    try:
        u = _up(SERVE_URL)
        conn = http.client.HTTPConnection(u.hostname, u.port, timeout=30)
        conn.request("POST", f"/session/{sid}/prompt_async",
                     body=json.dumps(body), headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = resp.read()
        if resp.status == 204:
            return True, ""
        return False, f"prompt_async HTTP {resp.status}: {data[:500]!r}"
    except Exception as e:
        return False, f"prompt_async failed: {e}"


def _abort_turn(sid):
    try:
        _serve_call("POST", f"/session/{sid}/abort", {}, timeout=10)
    except Exception:
        pass


def _event_subscriber(stop_flag, q):
    """GET /event (global stream, 1.18.31 has no session filter) -> queue.

    Items: ("event", evdict) | ("tick", None) on read timeout | ("eof", None).
    """
    import socket
    try:
        req = urllib.request.Request(SERVE_URL + "/event", method="GET")
        with urllib.request.urlopen(req, timeout=30) as r:
            lines = []
            while not stop_flag.is_set():
                try:
                    raw = r.readline(65536)
                except (socket.timeout, TimeoutError):
                    q.put(("tick", None))
                    continue
                except Exception as e:
                    q.put(("eof", f"read: {e}"))
                    return
                if not raw:
                    q.put(("eof", "closed"))
                    return
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                if line == "":
                    if not lines:
                        continue
                    data_text = "\n".join(
                        l[5:].lstrip() for l in lines if l.startswith("data:"))
                    lines = []
                    if not data_text or data_text == "[DONE]":
                        continue
                    try:
                        q.put(("event", json.loads(data_text)))
                    except Exception:
                        continue
                elif line.startswith(":"):
                    continue  # SSE comment / heartbeat from serve
                else:
                    lines.append(line)
    except Exception as e:
        q.put(("eof", str(e)[:200]))


def _relay_turn(sid, msg_body, wall_timeout, on_delta=None, should_stop=None):
    """Run one turn via prompt_async + /event relay.

    on_delta(text) forwards live text (stream path) else buffered only.
    on_delta(None) means heartbeat ping requested.
    Returns (ok, full_text, err, meta{ttfb_ms, usage, events, streamed_chunks}).
    Full text is reconciled from GET message (never assembled from deltas).
    """
    import queue as _queue
    t_start = time.time()
    meta = {"ttfb_ms": 0, "usage": None, "events": 0, "streamed_chunks": 0, "streamed_text": "", "displayed_text": ""}
    q = _queue.Queue()
    stop_flag = threading.Event()
    sub = threading.Thread(target=_event_subscriber, args=(stop_flag, q), daemon=True)
    sub.start()
    # Connect the subscriber before prompt_async so early events aren't missed.
    time.sleep(0.5)
    last_part_id = None
    ok, err = _prompt_async(sid, msg_body)
    if not ok:
        stop_flag.set()
        return False, "", err, meta
    last_event = time.time()
    last_forward = time.time()
    ttfb = None
    fatal = None
    while True:
        now = time.time()
        if wall_timeout and now - t_start > wall_timeout:
            fatal = f"turn wall timeout after {int(now - t_start)}s"
            break
        if should_stop and should_stop():
            fatal = "client disconnected"
            break
        try:
            kind, payload = q.get(timeout=1)
        except Exception:
            kind, payload = ("tick", None)
        now = time.time()
        if kind == "event":
            props = payload.get("properties", {}) if isinstance(payload, dict) else {}
            if props.get("sessionID") != sid:
                continue
            meta["events"] += 1
            last_event = now
            etype = payload.get("type", "")
            if etype == "message.part.delta":
                d = props.get("delta", "")
                if props.get("field") == "text" and d:
                    if ttfb is None:
                        ttfb = now
                        meta["ttfb_ms"] = int((ttfb - t_start) * 1000)
                    part_id = props.get("partID") or (props.get("part") or {}).get("id")
                    if part_id and part_id != last_part_id and meta["streamed_text"] and not meta["streamed_text"].endswith("\n"):
                        sep = "\n\n"
                        if on_delta:
                            on_delta(sep)
                        meta["streamed_text"] += sep
                        meta["displayed_text"] += sep
                    last_part_id = part_id
                    if on_delta:
                        try:
                            on_delta(d)
                            meta["streamed_chunks"] += 1
                            meta["streamed_text"] += d
                            meta["displayed_text"] += d
                        except Exception:
                            fatal = "client disconnected"
                            break
                    last_forward = now
            elif etype == "message.part.updated":
                part = props.get("part") or {}
                if SHIM_NARRATE_TOOLS and part.get("type") == "tool" and part.get("state", {}).get("status") == "completed":
                    tool_name = part.get("tool") or "tool"
                    summary = f"\n\u2699 {tool_name}\n"
                    if on_delta:
                        on_delta(summary)
                    meta["displayed_text"] += summary
            elif etype == "session.idle":
                break  # success; reconcile below
            elif etype == "session.error":
                fatal = f"session error: {json.dumps(props.get('error', {}))[:500]}"
                break
        elif kind == "eof":
            fatal = f"event stream ended ({payload})"
            break
        idle_for = now - last_event
        if idle_for > SHIM_IDLE_TIMEOUT:
            fatal = f"inactivity timeout ({SHIM_IDLE_TIMEOUT}s since last event)"
            break
        if on_delta and now - last_forward >= SHIM_HEARTBEAT:
            try:
                on_delta(None)  # heartbeat ping
            except Exception:
                fatal = "client disconnected"
                break
            last_forward = now
    stop_flag.set()
    if fatal:
        _abort_turn(sid)
        return False, "", fatal, meta
    try:
        ok2, msgs = _serve_call("GET", f"/session/{sid}/message", timeout=30)
        if not ok2 or not isinstance(msgs, list):
            return False, "", f"message list failed: {msgs}", meta
        assistants = [m for m in msgs
                      if isinstance(m, dict) and m.get("info", {}).get("role") == "assistant"]
        if not assistants:
            return False, "", "no assistant message after idle", meta
        last_m = assistants[-1]
        texts = [p.get("text", "") for p in last_m.get("parts", [])
                 if p.get("type") == "text" and p.get("text")]
        full = "\n".join(texts).strip()
        toks = last_m.get("info", {}).get("tokens", {}) or {}
        if isinstance(toks, dict) and (toks.get("input") or toks.get("output")):
            meta["usage"] = {"prompt_tokens": int(toks.get("input") or 0),
                             "completion_tokens": int(toks.get("output") or 0),
                             "total_tokens": int(toks.get("input") or 0) + int(toks.get("output") or 0)}
        if not meta["usage"]:
            with _stats_lock:
                _stats["usage_estimated"] += 1
            print(f"[usage-debug] {json.dumps(last_m.get('info', {}))[:1000]}")
            all_text = sum(len(p.get("text", "")) for m in msgs for p in m.get("parts", [])
                           if p.get("type") == "text")
            est_total = max(1, all_text // 4)
            meta["usage"] = {"prompt_tokens": est_total,
                             "completion_tokens": max(1, len(full) // 4),
                             "total_tokens": est_total + max(1, len(full) // 4),
                             "estimated": True}
        return True, full, "", meta
    except Exception as e:
        return False, "", f"reconcile failed: {e}", meta


def _openai_tool_calls(calls):
    """Stable OpenAI tool_calls (ids generated once, reused for hashing + reply)."""
    tcs = []
    for c in calls:
        tcs.append({
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])},
        })
    return tcs


def _fetch_models():
    """Discover available models from opencode serve. Returns list of model dicts."""
    global _models_cache
    now = time.time()
    with _models_cache_lock:
        if _models_cache["models"] and (now - _models_cache["updated"] < SHIM_MODELS_CACHE_TTL):
            return _models_cache["models"]
    ok, resp = _serve_call("GET", "/global/models", timeout=10)
    if not ok:
        # Fallback: try /global/health to check serve is up, then return default
        return [{"id": MODEL_ID, "object": "model",
                 "created": int(time.time()), "owned_by": "opencode-shim"}]
    if isinstance(resp, dict):
        models = resp.get("models") or resp.get("data") or resp.get("models")
        if isinstance(models, list) and models:
            normalized = []
            for m in models:
                if isinstance(m, dict):
                    mid = m.get("id") or m.get("modelID") or m.get("name")
                    if mid:
                        normalized.append({
                            "id": str(mid),
                            "object": "model",
                            "created": m.get("created", int(time.time())),
                            "owned_by": m.get("owned_by", "opencode"),
                        })
            if normalized:
                # Rank most powerful first (stable sort)
                normalized.sort(key=lambda m: (_model_power_score(m["id"]), m["id"]))
                with _models_cache_lock:
                    _models_cache["models"] = normalized
                    _models_cache["updated"] = now
                print(f"[models] discovered {len(normalized)} ranked: { [m['id'] for m in normalized[:8]] }")
                return normalized
    # Serve returned something unexpected; use the configured model as fallback
    return [{"id": MODEL_ID, "object": "model",
             "created": int(time.time()), "owned_by": "opencode-shim"}]
# --- Phase 0 observability (§7 plan_enhanced.md): structured req log + debug ---
SHIM_DEBUG_DUMP = os.environ.get("SHIM_DEBUG_DUMP", "0") == "1"
_SHIM_START = time.time()
_req_log = collections.deque(maxlen=50)  # each: dict with req line fields
_last_payload = {"redacted": None, "prompt_bytes": 0, "time": 0}  # exact outbound text, base64 redacted
_stats = {
    "requests": 0,
    "fence_ok": 0, "fence_repaired": 0, "fence_failed": 0,
    "tool_calls_total": 0,
    "total_ms": collections.deque(maxlen=200),
    "prompt_bytes": collections.deque(maxlen=200),
    "errors": 0,
    "catchup_fires": 0,
    "empty_nudges": 0,
    "usage_estimated": 0,
    "forks_max_turns": 0,
    "zen_429": 0,
}
_stats_lock = threading.Lock()


def _log_req_line(entry):
    _record_request(entry)
    print(f"[req] id={entry.get('id')} sess={entry.get('sess')} hit={int(bool(entry.get('hit')))} "
          f"depth={entry.get('depth')} delta_msgs={entry.get('delta_msgs')} "
          f"delta_bytes={entry.get('delta_bytes')} tools={entry.get('tools')} "
          f"tools_sent={entry.get('tools_sent')} sys_drift={int(bool(entry.get('sys_drift')))} "
          f"forked={int(bool(entry.get('forked')))} profile={entry.get('profile')} "
          f"attach={entry.get('attach')} stream={entry.get('stream')} fence={entry.get('fence')} "
          f"tool_calls={entry.get('tool_calls')} "
          f"ttfb_ms={entry.get('ttfb_ms', 0)} total_ms={entry.get('total_ms')} "
          f"finish={entry.get('finish')}" + (f" fork_reason={entry.get('fork_reason')}" if entry.get('fork_reason') else "") + (f" err={entry.get('err')}" if entry.get('err') else ""))


def _warnings_header(warnings):
    """Single Warning header value for sampling no-ops (§A.5)."""
    return '299 opencode-shim "' + "; ".join(warnings)[:500] + '"'


def _do_session_turn(handler, t0, req_id, messages, tools, tool_choice,
                     stream, req_model, n_msgs, n_tools, profile_override=None,
                     stop=None, max_tokens=None, parallel_tool_calls=None,
                     response_format=None, include_usage=False, serve_format=None,
                     warnings=None):
    """L1 session path (§2): one Hermes conversation -> one opencode session.

    Sends only the delta (new turns). Miss/fork -> create + one full dump.
    stop / max_tokens are applied post-hoc to the reconciled text (§A.1/A.2);
    response_format is prompt-enforced with one repair turn (§A.6).
    """
    store = get_store()
    resolution = store.resolve(messages)
    chain = resolution["chain"]
    full_hash = chain[-1]
    sid = resolution["session_id"]
    hit = resolution["hit"]
    forked = resolution["forked"]
    fork_reason = resolution["fork_reason"]
    sys_drift = resolution["sys_drift"]
    sys_update = resolution["system_update"]
    delta = resolution["delta"]
    profile = select_profile(tools, profile_override, req_model)
    agent = profile_agent(profile)
    # --- Session length cap (§10): proactive fork before opencode compaction ---
    carryover = None
    if hit and not forked and sid:
        try:
            info = store.session_info(sid)
        except Exception:
            info = None
        if info and info.get("turns", 0) >= SHIM_SESSION_MAX_TURNS:
            carryover = _make_session_carryover(messages, info)
            print(f"[sessions] {sid[:14]} at {info['turns']} turns >= max {SHIM_SESSION_MAX_TURNS}, forking with carryover")
            with _stats_lock:
                _stats["forks_max_turns"] += 1
            forked, fork_reason = True, f"max_turns {info['turns']}>={SHIM_SESSION_MAX_TURNS}"
            # inject carryover as sys_update so it reaches the model
            sys_update = (sys_update + "\n\n" + carryover) if sys_update else carryover
            # force new session; keep original messages for chain but mark miss
            sid, hit = None, False
            delta = list(messages)
    new_tools_hash = _tools_hash(tools) if tools else ""
    new_tool_names = _offered_names(tools) if tools else []
    tools_sent = 0

    def fail(code, msg, finish, ftype="invalid_request", param=None, code_slug=None):
        entry = {"id": req_id, "sess": (sid[:14] if sid else "-"), "session_id": sid or "-",
                 "hit": hit, "depth": n_msgs - 1,
                 "delta_msgs": len(delta), "delta_bytes": 0, "prompt_bytes": 0,
                 "tools": n_tools, "tools_sent": 0,
                 "sys_drift": sys_drift, "forked": forked, "fork_reason": fork_reason,
                 "profile": profile, "attach": 0, "stream": int(bool(stream)),
                 "fence": "-", "tool_calls": 0,
                 "total_ms": int((time.time() - t0) * 1000), "finish": finish}
        _log_req_line(entry)
        handler._json(code, _openai_error(msg, ftype, param=param, code=code_slug))

    # Exact repeat within TTL -> cached response, no new turn ( Generale retry guard).
    cached = store.check_idempotent(full_hash, SHIM_IDEMPOTENT_TTL)
    if cached is not None:
        entry = {"id": req_id, "sess": (sid[:14] if sid else "-"), "session_id": sid or "-",
                 "hit": hit, "depth": n_msgs - 1,
                 "delta_msgs": len(delta), "delta_bytes": 0, "prompt_bytes": 0,
                 "tools": n_tools, "tools_sent": n_tools,
                 "sys_drift": sys_drift, "forked": False, "fork_reason": "",
                 "profile": profile, "attach": 0, "stream": int(bool(stream)),
                 "fence": "cached", "tool_calls": 0,
                 "total_ms": int((time.time() - t0) * 1000), "finish": "cached"}
        _log_req_line(entry)
        print(f"[idempotent] {req_id} served from cache")
        _serve_cached(handler, cached, stream, req_model, include_usage=include_usage,
                      warnings=warnings)
        return

    # Empty delta on a hit = exact-duplicate of an earlier prefix (e.g. two
    # threads opening with identical messages). Fork safely, never 400.
    if hit and not delta and not sys_update:
        forked, fork_reason = True, "empty-delta collision"
        sid, hit = None, False
        delta = list(messages)

    if tools:
        try:
            print(f"[tools] n={len(tools)} names={tool_names(tools)[:15]} choice={str(tool_choice)[:120]}")
        except Exception:
            pass
    print(f"[profile] {profile} agent={agent} tools={n_tools}")

    # Build prompt: delta-only on hit, one full dump on miss/fork.
    # Tools-once-per-session (§3.1): full schemas on first dump, diff on change,
    # compact one-liner (or nothing) when unchanged.
    if hit:
        if tools:
            info = store.session_info(sid) or {}
            if info.get("tools_hash") == new_tools_hash:
                tools_section = build_tools_compact(tools, tool_choice)
                tools_sent = 0
            else:
                tools_section, n_added = build_tools_diff(tools, info.get("tool_names"))
                tools_section += build_tools_compact(tools, tool_choice)
                tools_sent = n_added
                store.update_tools(sid, new_tools_hash, new_tool_names, profile)
        else:
            tools_section = ""
        prompt, atts_raw = _build_delta_prompt(delta, sid, sys_update, tools_section)
        first_dump = False
        if carryover and carryover not in prompt:
            prompt = carryover + "\n\n" + prompt
    else:
        prompt, atts_raw = messages_to_prompt(messages, tools, tool_choice)
        if carryover and carryover not in prompt:
            prompt = carryover + "\n\n" + prompt
        tools_sent = n_tools
        first_dump = True

    # response_format (§A.6): prompt-enforced JSON via existing repair-turn
    # machinery. This is deliberately the SOLE enforcement path: native
    # serve `format` is kill-switched (confirmed regression, see
    # _serve_format_for_response_format), so there is no second machinery to
    # conflict with. Coexists with tools by construction — validation is
    # skipped for tool-call turns, where content is incidental.
    if response_format:
        prompt += _response_format_instruction(response_format)

    # Normalize attachments BEFORE acquiring any LLM slot (no hold during I/O).
    try:
        normed = normalize_attachments(atts_raw)
    except ValueError as e:
        fail(400, e, "400", "invalid_request", param="messages", code_slug="invalid_attachment")
        return
    except Exception as e:
        fail(400, f"bad attachment: {e}", "400", "invalid_request", param="messages", code_slug="invalid_attachment")
        return
    try:
        file_parts, file_notes = prepare_file_inputs(normed)
    except ValueError as e:
        fail(400, e, "400", "invalid_request", param="messages", code_slug="unsupported_media")
        return
    except Exception as e:
        fail(400, f"bad attachment: {e}", "400", "invalid_request", param="messages", code_slug="invalid_attachment")
        return
    if file_notes:
        prompt += "\n\n" + "\n".join(file_notes)
    if file_parts or file_notes:
        try:
            total_b64 = sum(len(p.get("url", "")) for p in file_parts)
            mimes = sorted({p.get("mime", "?") for p in file_parts})
            print(f"[attachments] parts={len(file_parts)} notes={len(file_notes)} mimes={mimes} b64chars={total_b64}")
        except Exception:
            pass

    prompt_bytes = len(prompt.encode())
    try:
        with _stats_lock:
            _last_payload.update({"redacted": _redact_payload(prompt[:8000], file_parts),
                                  "prompt_bytes": prompt_bytes, "time": int(time.time())})
        if SHIM_DEBUG_DUMP:
            os.makedirs("/tmp/shim-dumps", exist_ok=True)
            with open(f"/tmp/shim-dumps/{req_id}.json", "w") as f:
                json.dump({"prompt": prompt, "file_parts": file_parts,
                           "n_msgs": n_msgs, "n_tools": n_tools}, f)
    except Exception as e:
        print(f"[debug] payload capture failed: {e}")

    # Acquire: global slots first, then the per-session turn lock (fixed order).
    if not _sem.acquire(blocking=True, timeout=280):
        fail(429, "busy: another opencode run in progress, retry shortly", "429", "rate_limit")
        return
    # Stream path: pre-generate completion id so live chunks match the final.
    stream_cid = f"chatcmpl-{uuid.uuid4().hex[:12]}" if stream else None
    stream_created = int(time.time()) if stream else None
    stream_live = False
    stream_first = True
    relay_meta = {"ttfb_ms": 0, "usage": None, "events": 0, "streamed_chunks": 0, "streamed_text": "", "displayed_text": ""}

    def _emit_live(text):
        """Write one OpenAI chunk (or heartbeat ping when text is None)."""
        nonlocal stream_first
        if text is None:
            chunk = {"id": stream_cid, "object": "chat.completion.chunk",
                     "created": stream_created, "model": req_model,
                     "choices": [{"index": 0, "delta": {}, "finish_reason": None}]}
            handler.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            handler.wfile.flush()
            return
        chunk = {"id": stream_cid, "object": "chat.completion.chunk",
                 "created": stream_created, "model": req_model,
                 "choices": [{"index": 0,
                              "delta": ({"role": "assistant", "content": text}
                                        if stream_first else {"content": text}),
                              "finish_reason": None}]}
        handler.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        handler.wfile.flush()
        stream_first = False

    def _run_once(exec_sid, exec_prompt, exec_parts, model_override=None, wall_timeout=None):
        """One turn via event relay; blocking fallback only if async never started."""
        _eff_model = model_override if model_override is not None else req_model
        resolved = _resolve_model(_eff_model)
        body = {"model": {"providerID": "opencode", "modelID": resolved},
                "parts": [{"type": "text", "text": exec_prompt}] + list(exec_parts or [])}
        if agent:
            body["agent"] = agent
        if serve_format is not None:
            body["format"] = serve_format
        _wt = wall_timeout if wall_timeout is not None else TIMEOUT
        # When stop/max_tokens/response_format is active the final text is
        # post-processed (truncated / validated) — buffer instead of streaming
        # live so the client never sees the pre-truncation bytes (§A.1).
        _buffered = bool(stop or max_tokens or response_format)
        ok_r, text_r, err_r, meta_r = _relay_turn(
            exec_sid, body, _wt,
            on_delta=(_emit_live if (stream and not tools and not _buffered) else None))
        if ok_r:
            return True, text_r, "", meta_r, True
        # Fallback: blocking call only when prompt_async never started the turn
        # (anything later may already be running server-side — never double-run).
        if err_r.startswith("prompt_async"):
            ok_b, text_b, err_b = _run_session_turn(exec_sid, exec_prompt, exec_parts, agent, model_override=model_override,
                                                    serve_format=serve_format)
            return ok_b, text_b, err_b, meta_r, False
        return False, "", err_r, meta_r, True

    try:
        if sid is None:
            ok, sess = _serve_call("POST", "/session",
                                   {"agent": agent}, timeout=15)
            if not ok or "id" not in sess:
                fail(500, f"session create failed: {sess}", "500", "backend_error")
                return
            sid = sess["id"]
        slock = L1.session_lock(sid)
        if not slock.acquire(blocking=True, timeout=SHIM_SESSION_LOCK_TIMEOUT):
            fail(429, "busy: previous turn on this conversation still running", "429", "rate_limit")
            return
        # SSE headers go out now (stream path) so deltas + heartbeats flow live.
        # 429s above still return JSON; everything after this point is SSE.
        if stream:
            try:
                handler.send_response(200)
                handler.send_header("Content-Type", "text/event-stream")
                handler.send_header("Cache-Control", "no-cache")
                handler.send_header("Connection", "keep-alive")
                if warnings:
                    handler.send_header("Warning", _warnings_header(warnings))
                handler._cors()
                handler.end_headers()
                stream_live = True
                _emit_live("")  # zero-width role-delta, satisfies TTFB instantly
            except (BrokenPipeError, ConnectionResetError):
                slock.release()
                entry = {"id": req_id, "sess": sid[:14], "session_id": sid,
                         "hit": hit, "depth": n_msgs - 1,
                         "delta_msgs": len(delta), "delta_bytes": prompt_bytes,
                         "prompt_bytes": prompt_bytes,
                         "tools": n_tools, "tools_sent": tools_sent,
                         "sys_drift": sys_drift, "forked": forked, "fork_reason": fork_reason,
                         "profile": profile, "attach": len(normed), "stream": 1,
                         "fence": "-", "tool_calls": 0, "ttfb_ms": 0,
                         "total_ms": int((time.time() - t0) * 1000), "finish": "client_gone"}
                _log_req_line(entry)
                return
        try:
            # Fallback chain: all discovered models, most powerful first, requested first.
            # Cumulative deadline so 4×120s does not stall 8 min on total failure (fix design gap).
            _fallback_deadline = time.time() + TIMEOUT
            _chain = _fallback_chain(req_model) if SHIM_MODEL_FALLBACK else [req_model or MODEL_ID]
            _chain = _chain[: SHIM_MODEL_FALLBACK_RETRIES + 1]
            ok = False; out = ""; err = "no attempt"; meta_got = {}; used_relay = True
            _active_try_model = None
            for _try_idx, _try_model in enumerate(_chain):
                _active_try_model = _try_model
                if _try_idx > 0:
                    print(f"[fallback] model {req_model} -> {_try_model} after: {err[:120]}")
                _remaining = _fallback_deadline - time.time()
                if _remaining <= 5:
                    print(f"[fallback] overall budget exhausted at attempt {_try_idx+1}/{len(_chain)}")
                    err = f"fallback budget exhausted after {_try_idx} attempts"
                    break
                ok, out, err, meta_got, used_relay = _run_once(sid, prompt, file_parts, model_override=_try_model, wall_timeout=_remaining)
                # _relay_turn may return None meta on failure; guard it
                if isinstance(meta_got, dict):
                    relay_meta.update({k: v for k, v in meta_got.items() if v is not None})
                if not used_relay:
                    stream_live = False
                if ok:
                    if _try_idx > 0:
                        print(f"[fallback] succeeded with {_try_model}")
                    break
                if err.startswith("SESSION_GONE"):
                    break
                _low = (err or "").lower()
                _eligible = any(kw in _low for kw in ["401","unauthorized","api key","overloaded","429","503","rate limit","quota","capacity","model not found","not found","billing","insufficient","forbidden","overload"])
                if not _eligible:
                    break
                if _try_idx >= len(_chain)-1:
                    break
            if not ok and err.startswith("SESSION_GONE"):
                # Serve pruned the session; recreate once with a full replay.
                print(f"[sessions] {sid[:14]} gone on serve, recreating with full replay")
                ok2, sess2 = _serve_call("POST", "/session",
                                          {"agent": agent}, timeout=15)
                if not ok2 or "id" not in sess2:
                    fail(500, f"session recreate failed: {sess2}", "500", "backend_error")
                    return
                sid = sess2["id"]
                slock = L1.session_lock(sid)
                if not slock.acquire(blocking=True, timeout=SHIM_SESSION_LOCK_TIMEOUT):
                    fail(429, "busy: previous turn still running", "429", "rate_limit")
                    return
                try:
                    prompt2, atts2 = messages_to_prompt(messages, tools, tool_choice)
                    normed2 = normalize_attachments(atts2)
                    fp2, fn2 = prepare_file_inputs(normed2)
                    if fn2:
                        prompt2 += "\n\n" + "\n".join(fn2)
                    _remaining2 = _fallback_deadline - time.time()
                    if _remaining2 <= 5:
                        _remaining2 = 5
                    ok, out, err, meta_got2, used_relay2 = _run_once(sid, prompt2, fp2, model_override=_active_try_model, wall_timeout=_remaining2)
                    relay_meta.update(meta_got2)
                    if not used_relay2:
                        stream_live = False
                    prompt_bytes = len(prompt2.encode())
                    tools_sent = n_tools
                    first_dump = True
                    forked, fork_reason = True, "session_gone"
                finally:
                    slock.release()
        finally:
            # release only if still held by us (recreate path swapped locks)
            try:
                slock.release()
            except RuntimeError:
                pass
    finally:
        _sem.release()

    if not ok:
        msg = err or "opencode run failed"
        # Distinct Zen-side 429 (rate limit) — before generic 401/500 collapse
        _low = (err or "").lower()
        is_429 = ("429" in (err or "")) or any(kw in _low for kw in ["rate limit", "quota", "too many requests", "overloaded", "capacity"])
        # Also check structured error in msg if present
        if not is_429 and "429" in msg:
            is_429 = True
        if is_429:
            with _stats_lock:
                _stats["zen_429"] += 1
            msg += "\n\nZen backend rate-limited (429). Back off with jitter and retry; do not hammer."
            status = 429
        elif "auth" in msg.lower() or "401" in msg or "unauthorized" in msg or "api key" in msg.lower():
            msg += ("\n\nHint: one-time setup needed: run `opencode auth login` -> OpenCode Zen "
                    "(free, no card at opencode.ai/auth), then retry. Hermes side stays keyless.")
            status = 401
        elif "inactivity timeout" in err or "wall timeout" in err:
            status = 504
        else:
            status = 401 if "401" in err or "unauthorized" in err.lower() else 500
        entry = {"id": req_id, "sess": sid[:14], "session_id": sid,
                 "hit": hit, "depth": n_msgs - 1,
                 "delta_msgs": len(delta), "delta_bytes": prompt_bytes, "prompt_bytes": prompt_bytes,
                 "tools": n_tools, "tools_sent": tools_sent,
                 "sys_drift": sys_drift, "forked": forked, "fork_reason": fork_reason,
                 "profile": profile, "attach": len(normed), "stream": int(bool(stream)),
                 "fence": "-", "tool_calls": 0, "ttfb_ms": relay_meta.get("ttfb_ms", 0),
                 "total_ms": int((time.time() - t0) * 1000), "finish": str(status),
                 "err": (err or msg)[:200]}
        _log_req_line(entry)
        if stream_live:
            # Headers already sent: close the SSE stream with an error chunk.
            try:
                err_chunk = {"id": stream_cid, "object": "chat.completion.chunk",
                             "created": stream_created, "model": req_model,
                             "choices": [{"index": 0, "delta": {},
                                          "finish_reason": "stop",
                                          "error": msg[:500]}]}
                handler.wfile.write(f"data: {json.dumps(err_chunk)}\n\ndata: [DONE]\n\n".encode())
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        _etype = "rate_limit" if status == 429 else "backend_error"
        _ebody = _openai_error(msg, _etype)
        _ebody["error"]["output"] = out[:2000]
        handler._json(status, _ebody)
        return

    # Fence protocol v2 (§3.3): parse, validate, one repair turn on failure.
    raw_out = out  # full model text (stream clients echo this verbatim, fence included)
    calls, remaining = ([], out)
    fence = "text"
    tcs = None
    forced_name = None
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function", {})
        forced_name = fn.get("name") if isinstance(fn, dict) else None
    must_call = tool_choice == "required" or bool(forced_name)
    if tools and tool_choice != "none":
        try:
            parsed, remaining, ferr = parse_fence(out, tools, forced_name)
        except Exception as e:
            print(f"[tools] parse failed: {e}")
            parsed, remaining, ferr = None, out, None
        if parsed:
            calls = parsed
        elif ferr is None and must_call:
            ferr = (f"tool_choice requires a tool call "
                    f"{('(name: ' + forced_name + ')') if forced_name else ''} "
                    f"but the model answered in plain text")
        if ferr and SHIM_REPAIR_TURNS > 0:
            # One repair turn on the SAME session: the model still has its own
            # reasoning in context, so this is far cheaper than a Hermes retry.
            print(f"[tools] repair turn: {ferr[:200]}")
            repair_prompt = (
                f"[protocol error] {ferr}. Re-emit ONLY the corrected "
                f"```hermes-toolcalls block (a JSON array of "
                f'{{"name": "<tool>", "arguments": {{<args object>}}}}), '
                f"as the last thing in your reply, nothing after it.")
            if _sem.acquire(blocking=True, timeout=60):
                try:
                    slock2 = L1.session_lock(sid)
                    if slock2.acquire(blocking=True, timeout=60):
                        try:
                            ok_r, out_r, err_r, meta_r, _ = _run_once(sid, repair_prompt, [])
                            relay_meta.update({k: v for k, v in meta_r.items()
                                               if v and k != "ttfb_ms"})
                        finally:
                            slock2.release()
                    else:
                        ok_r, out_r, err_r = False, "", "repair lock busy"
                finally:
                    _sem.release()
            else:
                ok_r, out_r, err_r = False, "", "repair slot busy"
            if ok_r:
                try:
                    parsed2, remaining2, ferr2 = parse_fence(out_r, tools, forced_name)
                except Exception:
                    parsed2, remaining2, ferr2 = None, out_r, "re-parse failed"
                if parsed2:
                    calls, remaining, out = parsed2, remaining2, out_r
                    fence = "repaired"
                    ferr = None
                else:
                    out = out_r
                    fence = "failed"
                    ferr = ferr2
                    print(f"[tools] repair failed: {str(ferr2)[:200]}")
            else:
                fence = "failed"
                print(f"[tools] repair turn failed: {err_r[:200]}")
        # §A.4: client opted out of parallel calls — keep only the first.
        calls = _limit_parallel_calls(calls, parallel_tool_calls)
        if calls:
            if fence == "text":
                fence = "ok"
            tcs = _openai_tool_calls(calls)
            print(f"[tools] emitting {len(calls)} call(s): {[c['name'] for c in calls]}")
            store.note_tool_calls(sid, {tc["id"]: tc["function"]["name"] for tc in tcs})
            out = remaining
        elif fence == "text" and must_call:
            fence = "failed"

    if not calls and not (out or "").strip() and SHIM_REPAIR_TURNS > 0:
        with _stats_lock:
            _stats["empty_nudges"] += 1
        print(f"[repair] {sid[:14]} empty completion, no tool call - nudging once")
        nudge = ("[system] Your previous turn produced no visible output. "
                 "Respond now - either plain text, or a ```hermes-toolcalls "
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
                        calls2 = _limit_parallel_calls(calls2, parallel_tool_calls)
                        calls, out = calls2, remaining2
                        tcs = _openai_tool_calls(calls)
                        store.note_tool_calls(sid, {tc["id"]: tc["function"]["name"] for tc in tcs})
                except Exception:
                    pass
        else:
            fail(502, "opencode produced no output for this turn, even after a nudge", "502", "backend_error")
            return

    # response_format (§A.6): validate reconciled text; one repair turn on failure.
    # Skipped for tool-call turns (content is incidental when tools fire).
    if response_format and not calls and SHIM_REPAIR_TURNS > 0:
        _rf_ok, _rf_err = _validate_response_format(out, response_format)
        if not _rf_ok:
            print(f"[response_format] repair turn: {_rf_err[:200]}")
            _rf_prompt = ("[system] Your last response was not valid JSON"
                          + (f" ({_rf_err})" if _rf_err else "")
                          + ". Reply with only the corrected JSON, nothing else.")
            _rf_ok_r, _rf_out_r, _rf_err_r = False, "", ""
            if _sem.acquire(blocking=True, timeout=60):
                try:
                    _rf_lock = L1.session_lock(sid)
                    if _rf_lock.acquire(blocking=True, timeout=60):
                        try:
                            _rf_ok_r, _rf_out_r, _rf_err_r, _rf_meta_r, _ = _run_once(sid, _rf_prompt, [])
                            relay_meta.update({k: v for k, v in _rf_meta_r.items()
                                               if v and k != "ttfb_ms"})
                        finally:
                            _rf_lock.release()
                    else:
                        _rf_ok_r, _rf_out_r, _rf_err_r = False, "", "repair lock busy"
                finally:
                    _sem.release()
            else:
                _rf_ok_r, _rf_out_r, _rf_err_r = False, "", "repair slot busy"
            if _rf_ok_r and _rf_out_r.strip():
                _rf_ok2, _rf_err2 = _validate_response_format(_rf_out_r, response_format)
                if _rf_ok2:
                    out = raw_out = _rf_out_r
                    fence = "repaired-json"
                else:
                    _rf_err = _rf_err2
                    _rf_ok = False
            else:
                _rf_ok = False
                _rf_err = _rf_err_r or _rf_err
            if not _rf_ok:
                _msg502 = ("response_format could not be satisfied: model did not "
                           f"produce valid JSON ({(_rf_err or 'unknown')[:300]})")
                if stream_live:
                    try:
                        err_chunk = {"id": stream_cid, "object": "chat.completion.chunk",
                                     "created": stream_created, "model": req_model,
                                     "choices": [{"index": 0, "delta": {},
                                                  "finish_reason": "stop",
                                                  "error": _msg502[:500]}]}
                        handler.wfile.write(f"data: {json.dumps(err_chunk)}\n\ndata: [DONE]\n\n".encode())
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                fail(502, _msg502, "502", "backend_error")
                return

    # stop / max_tokens post-hoc shaping (§A.1/A.2) on the reconciled text.
    _hit_max = False
    if stop or max_tokens:
        if calls:
            _new_remaining, _ = _apply_stop(remaining or "", stop)
            _new_remaining, _hit_max = _apply_max_tokens(_new_remaining, max_tokens)
            remaining, out = _new_remaining, _new_remaining
        else:
            _new_out, _ = _apply_stop(out or "", stop)
            _new_out, _hit_max = _apply_max_tokens(_new_out, max_tokens)
            out = raw_out = _new_out
    # A valid tool call always wins: "length" beside an executable tool_calls
    # array is an ambiguous signal no client can act on. The cap still
    # bounds the narration text above; only the label yields.
    _finish = "tool_calls" if calls else ("length" if _hit_max else "stop")

    # Register: new session -> full chain; then pre-register chain+reply.
    if first_dump:
        store.register_new(chain, sid, _system_raw_of(messages),
                            tools_hash=new_tools_hash, profile=profile,
                            tool_names=new_tool_names)
    reply_msg = {"role": "assistant", "content": (out if out else None),
                 "tool_calls": tcs}
    store.register_reply(chain, reply_msg, sid)
    # Streaming clients (Hermes) build history from streamed deltas, whose
    # content is the FULL raw text (fence included) — not the stripped
    # remaining. Pre-register that variant too so tool-call turns hit.
    if raw_out and raw_out != out:
        store.register_reply(chain, {"role": "assistant", "content": raw_out,
                                      "tool_calls": tcs}, sid, bump=False)

    total_ms = int((time.time() - t0) * 1000)
    usage = relay_meta.get("usage")
    # Preserve fallback-served model for debugging (ignored by OpenAI clients)
    _served = locals().get("_active_try_model") or req_model
    if calls:
        resp = chat_completion_tool_response(req_model, remaining or None, calls, tcs=tcs,
                                             cid=stream_cid, created=stream_created,
                                             usage=usage, finish_reason=_finish)
    else:
        resp = chat_completion_response(req_model, out,
                                        cid=stream_cid, created=stream_created,
                                        usage=usage, finish_reason=_finish)
    # Harmless extra field so quality complaints can be traced to fallback model
    try:
        if _served and _served != req_model:
            resp["x_shim_served_model"] = _served
    except Exception:
        pass
    store.store_idempotent(full_hash, {"resp": resp, "tool": bool(calls)})
    entry = {"id": req_id, "sess": sid[:14], "session_id": sid,
             "hit": hit, "depth": n_msgs - 1,
             "delta_msgs": len(delta), "delta_bytes": prompt_bytes, "prompt_bytes": prompt_bytes,
             "tools": n_tools, "tools_sent": tools_sent,
             "sys_drift": sys_drift, "forked": forked, "fork_reason": fork_reason,
             "profile": profile, "attach": len(normed), "stream": int(bool(stream)),
              "fence": fence, "tool_calls": len(calls),
              "ttfb_ms": relay_meta.get("ttfb_ms", 0),
              "total_ms": total_ms, "finish": _finish}
    _log_req_line(entry)

    if stream_live and not calls:
        def _flat(s):
            return re.sub(r"\s+", "", s or "")
        streamed = relay_meta.get("streamed_text", "")
        displayed = relay_meta.get("displayed_text", "")
        final_text = out or ""
        if not streamed and final_text.strip():
            _emit_live(final_text)
            with _stats_lock:
                _stats["catchup_fires"] += 1
            print(f"[stream] catch-up: 0 live chunks, sent {len(final_text)} reconciled chars")
        elif streamed and _flat(final_text) != _flat(streamed):
            # Separator/whitespace differences make exact tail-diffing unreliable
            # across multiple parts. Occasional visible duplication is far safer
            # than a silently dropped tail. Compare against streamed_text (model-only),
            # not displayed_text (which includes narrated tool lines), so narrations
            # do not trigger spurious resends.
            _emit_live(("\n\n" if streamed.strip() else "") + final_text)
            with _stats_lock:
                _stats["catchup_fires"] += 1
            print(f"[stream] catch-up: flat mismatch, resent {len(final_text)} reconciled chars (streamed {len(streamed)} displayed {len(displayed)})")

    if stream_live:
        # Headers sent before the turn; text deltas already forwarded live.
        # Emit only the terminal chunk (no content duplication).
        cid, created = resp["id"], resp["created"]
        try:
            if calls:
                _sse_send_tool_calls(handler, cid, created, req_model, None, calls, tcs=tcs,
                                     finish_reason=_finish, usage=usage,
                                     include_usage=include_usage)
            else:
                done = {"id": cid, "object": "chat.completion.chunk", "created": created,
                        "model": req_model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": _finish}]}
                handler.wfile.write(f"data: {json.dumps(done)}\n\n".encode())
                if include_usage and usage is not None:
                    _sse_send_usage_chunk(handler, cid, created, req_model, usage)
                handler.wfile.write(b"data: [DONE]\n\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
        return
    if calls and stream:
        cid, created = resp["id"], resp["created"]
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "keep-alive")
        if warnings:
            handler.send_header("Warning", _warnings_header(warnings))
        handler._cors()
        handler.end_headers()
        _sse_send_tool_calls(handler, cid, created, req_model, remaining or None, calls, tcs=tcs,
                             finish_reason=_finish, usage=usage,
                             include_usage=include_usage)
        return
    if not calls and stream:
        cid, created = resp["id"], resp["created"]
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "keep-alive")
        if warnings:
            handler.send_header("Warning", _warnings_header(warnings))
        handler._cors()
        handler.end_headers()
        _sse_send_text(handler, cid, created, req_model, out, finish_reason=_finish,
                       usage=usage, include_usage=include_usage)
        return
    if warnings:
        resp["x_shim_warnings"] = list(warnings)
    handler._json(200, resp)


def _serve_cached(self, cached, stream, req_model, include_usage=False, warnings=None):
    """Re-emit an idempotent-cached response in the requested format."""
    resp = cached["resp"]
    if not stream:
        if warnings:
            resp = dict(resp)
            resp["x_shim_warnings"] = list(warnings)
        self._json(200, resp)
        return
    cid, created = resp["id"], resp["created"]
    self.send_response(200)
    self.send_header("Content-Type", "text/event-stream")
    self.send_header("Cache-Control", "no-cache")
    self.send_header("Connection", "keep-alive")
    if warnings:
        self.send_header("Warning", _warnings_header(warnings))
    self._cors()
    self.end_headers()
    _finish = ((resp.get("choices") or [{}])[0] or {}).get("finish_reason") or "stop"
    _usage = resp.get("usage")
    if cached.get("tool"):
        tcs = resp["choices"][0]["message"].get("tool_calls") or []
        content = resp["choices"][0]["message"].get("content")
        _sse_send_tool_calls(self, cid, created, req_model, content, None, tcs=tcs,
                             finish_reason=_finish, usage=_usage,
                             include_usage=include_usage)
    else:
        _sse_send_text(self, cid, created, req_model,
                       resp["choices"][0]["message"].get("content") or "",
                       finish_reason=_finish, usage=_usage,
                       include_usage=include_usage)


def _redact_payload(prompt, file_parts):
    """Copy of outbound payload with data: bytes replaced by size markers."""
    try:
        red_parts = []
        for p in file_parts or []:
            q = dict(p)
            url = q.get("url", "")
            if isinstance(url, str) and url.startswith("data:"):
                q["url"] = url[:64] + f"...<redacted {len(url)} chars>"
            red_parts.append(q)
        return {"prompt": prompt, "file_parts": red_parts}
    except Exception:
        return {"prompt": prompt[:2000], "file_parts": f"<{len(file_parts or [])} parts>"}


def _record_request(entry):
    with _stats_lock:
        _stats["requests"] += 1
        if entry.get("finish") in ("error", "400", "500", "429"):
            _stats["errors"] += 1
        if entry.get("fence") in ("ok", "text"):
            _stats["fence_ok"] += 1
        elif entry.get("fence") == "repaired":
            _stats["fence_repaired"] += 1
        elif entry.get("fence") == "failed":
            _stats["fence_failed"] += 1
        _stats["tool_calls_total"] += int(entry.get("tool_calls") or 0)
        try:
            _stats["total_ms"].append(int(entry.get("total_ms") or 0))
            _stats["prompt_bytes"].append(int(entry.get("prompt_bytes") or 0))
        except Exception:
            pass
        _req_log.append(entry)


def _percentile(vals, pct):
    if not vals:
        return 0
    s = sorted(vals)
    k = min(len(s) - 1, max(0, int(len(s) * pct / 100)))
    return s[k]


MIME_BY_EXT = {
    # images
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
    ".avif": "image/avif", ".svg": "image/svg+xml",
    # video (model input.video=true)
    ".mp4": "video/mp4", ".m4v": "video/mp4", ".webm": "video/webm",
    ".mov": "video/quicktime", ".mkv": "video/x-matroska",
    # pdf
    ".pdf": "application/pdf",
    # audio (model input.audio=true)
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg",
    ".oga": "audio/ogg", ".m4a": "audio/mp4", ".flac": "audio/flac",
    ".opus": "audio/opus",
}
MAX_IMAGE_BYTES = int(os.environ.get("SHIM_MAX_IMAGE_BYTES", str(12 * 1024 * 1024)))
MAX_VIDEO_BYTES = int(os.environ.get("SHIM_MAX_VIDEO_BYTES", str(25 * 1024 * 1024)))
MAX_PDF_BYTES = int(os.environ.get("SHIM_MAX_PDF_BYTES", str(12 * 1024 * 1024)))
MAX_AUDIO_BYTES = int(os.environ.get("SHIM_MAX_AUDIO_BYTES", str(12 * 1024 * 1024)))
# Generic binary (zip, apk, tar, ...): staged to inbox, never embedded.
MAX_FILE_BYTES = int(os.environ.get("SHIM_MAX_FILE_BYTES", str(50 * 1024 * 1024)))
# Non-forwardable mimes that UTF-8 decode within this size are re-sent as
# text/plain file parts (serve accepts text/*). Larger text -> inbox path ref.
MAX_TEXT_INLINE_BYTES = int(os.environ.get("SHIM_MAX_TEXT_INLINE_BYTES", str(256 * 1024)))
INBOX_NAME = os.environ.get("SHIM_INBOX_NAME", ".shim-inbox")
# Preferred search roots (colon-separated). Memory/history matches resolve here
# first; wide filesystem walks outside these roots are forbidden (see below).
SEARCH_ROOTS = [r for r in os.environ.get(
    "SHIM_SEARCH_ROOTS",
    "/home/mitansh/hermesworkspace:/home/mitansh/work",
).split(":") if r]
# opencode serve 1.18.31 rejects video/* + audio/* file parts
# ("'file part media type video/mp4/audio/wav' functionality not supported"),
# even though model caps list them. Gate them with a clear 400 until serve
# supports them; set SHIM_ENABLE_VIDEO/AUDIO=1 to forward anyway on newer serve.
ENABLE_VIDEO = os.environ.get("SHIM_ENABLE_VIDEO", "0") == "1"
ENABLE_AUDIO = os.environ.get("SHIM_ENABLE_AUDIO", "0") == "1"


def is_forwardable(mime):
    """Mimes serve accepts as file parts. Video/audio gated by enable flags."""
    m = (mime or "").lower()
    if m.startswith("image/") or m == "application/pdf" or m.startswith("text/"):
        return True
    if m.startswith("video/"):
        return ENABLE_VIDEO
    if m.startswith("audio/"):
        return ENABLE_AUDIO
    return False


def is_gated_media(mime):
    """Video/audio serve rejects outright -> explicit 400 instead of fallback."""
    m = (mime or "").lower()
    if m.startswith("video/"):
        return not ENABLE_VIDEO
    if m.startswith("audio/"):
        return not ENABLE_AUDIO
    return False


def inbox_dir():
    d = os.path.join(WORKDIR, INBOX_NAME)
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d


def stage_to_inbox(filename, data):
    """Write bytes into WORKDIR inbox so opencode tools can read them. Returns abs path."""
    d = inbox_dir()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(filename or "attachment"))[:100] or "attachment"
    name = f"{uuid.uuid4().hex[:8]}-{safe}"
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"file too large ({len(data)} bytes, max {MAX_FILE_BYTES}). [file]")
    path = os.path.join(d, name)
    with open(path, "wb") as f:
        f.write(data)
    return path


def data_url_to_bytes(url):
    header, _, b64 = url.partition(",")
    if ";base64" in header:
        return base64.b64decode("".join(b64.split()))
    return b64.encode("utf-8", "replace")


def limit_for_mime(mime):
    m = (mime or "").lower()
    if m.startswith("video/"):
        return MAX_VIDEO_BYTES
    if m == "application/pdf":
        return MAX_PDF_BYTES
    if m.startswith("audio/"):
        return MAX_AUDIO_BYTES
    if m.startswith("image/") or m.startswith("text/"):
        return MAX_IMAGE_BYTES
    return MAX_FILE_BYTES


def _allowed_mime(mime):
    m = (mime or "").lower()
    return m.startswith("image/") or m.startswith("video/") or m.startswith("audio/") or m == "application/pdf"


def _guess_mime_and_name(url, default="image/jpeg"):
    """Return (mime, filename_or_None) guessed from a data: URL, http(s) URL, file:// or bare path."""
    if url.startswith("data:"):
        mime = url[5:].split(";", 1)[0].strip() or default
        return mime, None
    clean = url.split("?", 1)[0].split("#", 1)[0].replace("file://", "", 1)
    ext = os.path.splitext(clean)[1].lower()
    mime = MIME_BY_EXT.get(ext, default)
    try:
        if "://" in url:
            base = os.path.basename(urlparse(url).path)
        else:
            base = os.path.basename(clean)
        fn = base or None
    except Exception:
        fn = None
    return mime, fn


def extract_text_and_images(content):
    """Split OpenAI-style message content into (text, [(mime, ref, filename)]).

    ref is a data: URL, http(s) URL, file:// URL or bare local path — normalized
    later to a data: URL by normalize_attachments(). Handles image_url,
    input_image (Responses API), generic image/video/file/document parts and
    Anthropic base64 source blocks.
    """
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return str(content), []
    texts, atts = [], []
    for p in content:
        if isinstance(p, str):
            texts.append(p)
        elif isinstance(p, dict):
            t = p.get("type")
            if t == "text":
                v = p.get("text", "")
                if isinstance(v, str):
                    texts.append(v)
            elif t == "image_url":
                ref = p.get("image_url")
                url = ref.get("url", "") if isinstance(ref, dict) else (ref if isinstance(ref, str) else "")
                if isinstance(url, str) and url:
                    mime, fn = _guess_mime_and_name(url)
                    atts.append((mime, url, fn))
            elif t == "input_image":
                ref = p.get("image_url", p.get("url", ""))
                url = ref.get("url", "") if isinstance(ref, dict) else (ref if isinstance(ref, str) else "")
                if isinstance(url, str) and url:
                    mime, fn = _guess_mime_and_name(url)
                    atts.append((mime, url, fn))
            elif t == "image":
                src = p.get("source")
                if isinstance(src, dict) and src.get("type") == "base64":
                    media = src.get("media_type") or p.get("mime_type") or "image/png"
                    b64 = src.get("data") or ""
                    if b64:
                        url = b64 if b64.startswith("data:") else f"data:{media};base64,{b64}"
                        atts.append((media, url, p.get("file_name") or p.get("filename")))
                        continue
                if isinstance(p.get("data"), str) and p["data"]:
                    mime = p.get("mime_type") or "image/png"
                    b64 = p["data"]
                    url = b64 if b64.startswith("data:") else f"data:{mime};base64,{b64}"
                    atts.append((mime, url, p.get("file_name") or p.get("filename")))
                    continue
                url = ""
                for key in ("image_url", "url", "image"):
                    v = p.get(key)
                    if isinstance(v, dict) and isinstance(v.get("url"), str):
                        url = v["url"]
                        break
                    if isinstance(v, str) and v:
                        url = v
                        break
                if url:
                    mime, fn2 = _guess_mime_and_name(url)
                    atts.append((mime, url, p.get("file_name") or p.get("filename") or fn2))
            elif t in ("video_url", "video"):
                ref = p.get("video_url", p.get("url", p.get("video", "")))
                url = ref.get("url", "") if isinstance(ref, dict) else (ref if isinstance(ref, str) else "")
                if not url and isinstance(p.get("url"), str):
                    url = p["url"]
                if url:
                    mime, fn = _guess_mime_and_name(url, default="video/mp4")
                    atts.append((mime, url, fn))
            elif t in ("audio_url", "audio"):
                ref = p.get("audio_url", p.get("url", p.get("audio", "")))
                url = ref.get("url", "") if isinstance(ref, dict) else (ref if isinstance(ref, str) else "")
                if not url and isinstance(p.get("url"), str):
                    url = p["url"]
                if url:
                    mime, fn = _guess_mime_and_name(url, default="audio/mpeg")
                    atts.append((mime, url, fn))
            elif t == "input_audio":
                # Genuine OpenAI shape: {"type":"input_audio","input_audio":{"data","format"}} (§B.1)
                ia = p.get("input_audio") or {}
                b64 = ia.get("data") if isinstance(ia, dict) else None
                fmt = (ia.get("format") if isinstance(ia, dict) else None) or "wav"
                if isinstance(b64, str) and b64:
                    mime = f"audio/{fmt}"
                    url = b64 if b64.startswith("data:") else f"data:{mime};base64,{b64}"
                    atts.append((mime, url, None))
            elif t == "refusal":
                # Assistant refusal echoed back into history (§B.3) — keep as text.
                v = p.get("refusal", "")
                if isinstance(v, str) and v:
                    texts.append(v)
            elif t in ("file", "document", "pdf"):
                fobj = p.get("file", p)
                url, fn = "", None
                if isinstance(fobj, dict):
                    url = fobj.get("url") or fobj.get("file_url") or p.get("url") or ""
                    fn = fobj.get("filename") or fobj.get("file_name") or p.get("filename")
                    if not url and isinstance(fobj.get("file_data"), str):
                        url = fobj["file_data"]
                    if not url and isinstance(fobj.get("data"), str):
                        b64 = fobj["data"]
                        mime0 = fobj.get("mime") or fobj.get("mime_type") or "application/pdf"
                        url = b64 if b64.startswith("data:") else f"data:{mime0};base64,{b64}"
                if not url and isinstance(fobj, dict) and fobj.get("file_id"):
                    # Files API reference with no inline data — unresolvable here (§B.4).
                    print(f"[attach] file_id reference not resolvable by this shim: {fobj['file_id']}")
                if url:
                    mime, fn2 = _guess_mime_and_name(url, default="application/pdf")
                    atts.append((mime, url, fn or fn2))
            elif "text" in p and isinstance(p["text"], str):
                texts.append(p["text"])
            elif "image_url" in p:
                # fallback: {"image_url": {...}} without type
                ref = p["image_url"]
                url = ref.get("url", "") if isinstance(ref, dict) else (ref if isinstance(ref, str) else "")
                if url:
                    mime, fn = _guess_mime_and_name(url)
                    atts.append((mime, url, fn))
    return "\n".join(texts), atts


def guess_mime(url):
    """Back-compat: return (mime, url) pair."""
    mime, _fn = _guess_mime_and_name(url)
    return mime, url


def validate_data_url(url):
    """Check a data: URL's mime + estimated size. Returns (mime, url, None)."""
    try:
        header, _, b64 = url.partition(",")
        if not header.startswith("data:") or not b64:
            raise ValueError("malformed data URL (missing data payload)")
        mime = header[5:].split(";")[0].strip() or "application/octet-stream"
        # Any mime allowed here; serve-side gating happens later (forwardable /
        # text-inline / inbox path). Only the byte cap is enforced.
        limit = limit_for_mime(mime)
        if ";base64" in header:
            est = len("".join(b64.split())) * 3 // 4
            if est > limit:
                raise ValueError(
                    f"attachment too large (~{est} bytes, max {limit} for {mime}). "
                    "Please shrink/compress and retry. [image_url]")
        elif len(url) > limit + 1024:
            raise ValueError(f"attachment too large (max {limit} for {mime}). [image_url]")
        return mime, url, None
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"bad data URL attachment: {e} [image_url]")


def file_to_data_url_with_path(ref):
    """Read file:// or bare local path -> (mime, data: URL, filename, abs_path)."""
    raw_ref = ref
    path = ref.replace("file://", "", 1)
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(WORKDIR, path)
    path = os.path.normpath(path)
    if not os.path.isfile(path):
        raise ValueError(f"local file not found: {raw_ref} (resolved {path}) [image_url]")
    ext = os.path.splitext(path)[1].lower()
    mime = MIME_BY_EXT.get(ext, "application/octet-stream")
    limit = limit_for_mime(mime)
    size = os.path.getsize(path)
    if size > limit:
        raise ValueError(f"file too large ({size} bytes, max {limit} for {mime}): {path}. Please shrink and retry. [image_url]")
    with open(path, "rb") as f:
        raw = f.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f"file too large (max {limit} for {mime}): {path} [image_url]")
    return mime, f"data:{mime};base64," + base64.b64encode(raw).decode(), os.path.basename(path), path


def file_to_data_url(ref):
    """Read file:// or bare local path -> (mime, data: URL, filename)."""
    mime, data_url, fn, _path = file_to_data_url_with_path(ref)
    return mime, data_url, fn


def download_http_to_data_url(url):
    """Download http(s) URL -> (mime, data: URL, filename). Serve can't fetch remote URLs."""
    req = urllib.request.Request(url, headers={"User-Agent": "opencode-shim/2.0"})
    try:
        with urllib.request.urlopen(req, timeout=IMAGE_TIMEOUT) as r:
            try:
                ctype = r.headers.get_content_type()
            except Exception:
                ctype = (r.headers.get("Content-Type", "") or "").split(";")[0].strip().lower()
            ctype = (ctype or "").lower().split(";")[0].strip()
            _mime_g, fn_guess = _guess_mime_and_name(url, default="application/octet-stream")
            hard_cap = max(MAX_IMAGE_BYTES, MAX_VIDEO_BYTES, MAX_PDF_BYTES, MAX_AUDIO_BYTES, MAX_FILE_BYTES) + 1
            raw = r.read(hard_cap)
            mime = None
            if ctype and "/" in ctype:
                mime = ctype
            if not mime:
                mime = _mime_g
            limit = limit_for_mime(mime)
            if len(raw) > limit:
                raise ValueError(
                    f"download too large ({len(raw)}+ bytes, max {limit} for {mime}): {url}. "
                    "Please shrink and retry. [image_url]")
            if not raw:
                raise ValueError(f"download empty: {url} [image_url]")
            return mime, f"data:{mime};base64," + base64.b64encode(raw).decode(), fn_guess
    except ValueError:
        raise
    except urllib.error.HTTPError as e:
        raise ValueError(f"download failed HTTP {e.code} for {url} [image_url]")
    except Exception as e:
        raise ValueError(f"download failed for {url}: {e} [image_url]")


def normalize_attachments(atts_raw):
    """Convert [(mime_guess, ref, filename)] -> [(mime, data_url, filename, src_path)].

    All payloads become data: URLs for serve. src_path is the absolute local
    path when the ref was a file:// or bare path (else None) — used for the
    inbox fallback for types serve rejects.
    """
    out = []
    for _mime_g, ref, fn_g in atts_raw or []:
        if not ref or not isinstance(ref, str):
            continue
        if ref.startswith("data:"):
            mime, data_url, _ = validate_data_url(ref)
            out.append((mime, data_url, fn_g, None))
        elif ref.startswith(("http://", "https://")):
            mime, data_url, fn = download_http_to_data_url(ref)
            out.append((mime, data_url, fn or fn_g, None))
        else:
            mime, data_url, fn, src = file_to_data_url_with_path(ref)
            out.append((mime, data_url, fn or fn_g, src))
    return out


def prepare_file_inputs(normed):
    """Split normalized attachments into serve file-parts + prompt notes.

    Returns (parts, notes) where parts = [{"type":"file",...}] for serve and
    notes = [str] describing inbox-staged binaries opencode should read via tools.
    Raises ValueError for gated video/audio (explicit 400 upstream).
    """
    parts, notes = [], []
    for mime, data_url, filename, src_path in normed:
        if is_gated_media(mime):
            raise ValueError(
                f"{mime.split('/')[0]} attachment(s) not supported by opencode serve "
                f"(model caps list it but serve rejects it). Supported now: image + pdf + text + "
                f"generic files via tool-readable path. Workaround: transcribe/describe the media "
                f"first and send text. Files: {filename or 'unknown'}. [audio/video]")
        if is_forwardable(mime):
            fp = {"type": "file", "mime": mime, "url": data_url}
            if filename:
                fp["filename"] = os.path.basename(str(filename))[:120]
            parts.append(fp)
            continue
        # Non-forwardable (zip, apk, tar, ...): small UTF-8 text -> text/plain part.
        raw = data_url_to_bytes(data_url)
        if len(raw) <= MAX_TEXT_INLINE_BYTES:
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = None
            if text is not None and ("\x00" not in text):
                fp = {"type": "file", "mime": "text/plain",
                      "url": "data:text/plain;base64," + base64.b64encode(raw).decode()}
                if filename:
                    fp["filename"] = os.path.basename(str(filename))[:120]
                parts.append(fp)
                continue
        # Binary fallback: tool-readable path in WORKDIR inbox.
        if src_path and os.path.isfile(src_path) and src_path.startswith(os.path.abspath(WORKDIR) + os.sep):
            usable = src_path
        else:
            usable = stage_to_inbox(filename or "attachment", raw)
        notes.append(
            f"[attached file: {filename or 'attachment'} ({mime}, {len(raw)} bytes) "
            f"saved at: {usable} — serve file-parts reject this type, so use your "
            f"tools (read/bash) to inspect it. Do not ask the user for the path.]")
    return parts, notes


def load_url_or_path(url):
    """Back-compat single-item normalizer (data: passthrough, http download, file read)."""
    mime_g, fn_g = _guess_mime_and_name(url)
    out = normalize_attachments([(mime_g, url, fn_g)])
    return out[0][1] if out else url


def extract_text(content):
    text, _ = extract_text_and_images(content)
    return text


TOOLCALL_FENCE = "hermes-toolcalls"
TOOLS_JSON_BUDGET = int(os.environ.get("SHIM_TOOLS_JSON_BUDGET", "49152"))


def tool_names(tools):
    names = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            names.append(str(fn["name"]))
        elif isinstance(t.get("name"), str):
            names.append(t["name"])
    return names


def build_tools_instruction(tools, tool_choice):
    """Prompt block teaching opencode how to emit Hermes tool calls (or not)."""
    if not tools:
        return ""
    if tool_choice == "none":
        return ("\n\n[Tools are defined but tool_choice='none': answer directly in plain "
                "text. Do NOT emit tool calls.]")
    try:
        full = json.dumps(tools)
    except Exception:
        full = str(tools)[:TOOLS_JSON_BUDGET]
    if len(full) > TOOLS_JSON_BUDGET:
        brief = []
        for t in tools:
            try:
                s = json.dumps(t)
            except Exception:
                s = str(t)
            brief.append(s[:1500])
        full = "[" + ", ".join(brief) + "]"[:TOOLS_JSON_BUDGET] + "...(truncated)"
    names = tool_names(tools)
    must = ""
    if tool_choice == "required":
        must = " You MUST call a tool (do not answer directly)."
    elif isinstance(tool_choice, dict):
        fn = tool_choice.get("function", {})
        want = fn.get("name") if isinstance(fn, dict) else None
        if want:
            must = f" You MUST call the tool named '{want}' (do not answer directly)."
    return (
        "\n\n[You have Hermes tools available." + must +
        " To call one or more tools, output ONLY a fenced block:\n"
        "```" + TOOLCALL_FENCE + "\n"
        '[{"name": "<tool-name>", "arguments": {<json-object-args>}}]\n'
        "```\n"
        "Rules: use only these tool names: " + ", ".join(names[:100]) + ". "
        "Arguments must be a single JSON object matching that tool's parameters schema "
        "(use {} if it takes none). No prose outside the fence when calling tools. "
        "To answer directly instead, output plain text with no fence.]\n"
        "Tool definitions: " + full
    )


def _tool_spec(tools):
    """{name: (required_list, {prop: type})} from OpenAI tools[]."""
    spec = {}
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            name, params = str(fn["name"]), fn.get("parameters") or {}
        elif isinstance(t.get("name"), str):
            name, params = t["name"], t.get("parameters") or {}
        else:
            continue
        req = (params.get("required") if isinstance(params, dict) else []) or []
        props = params.get("properties") if isinstance(params, dict) else {}
        types = {k: (v.get("type") if isinstance(v, dict) else None)
                 for k, v in (props.items() if isinstance(props, dict) else [])}
        spec[name] = ([str(r) for r in req if isinstance(r, str)], types)
    return spec


def _check_args(name, args, spec):
    """Validate arguments object. Returns error string or None. §3.3."""
    if not isinstance(args, dict):
        return f"tool '{name}': arguments must be a JSON object"
    required, types = spec.get(name, ([], {}))
    missing = [k for k in required if k not in args]
    if missing:
        return f"tool '{name}': missing required argument(s): {', '.join(missing)}"
    for k, v in args.items():
        want = types.get(k)
        if want == "string" and not isinstance(v, str):
            return f"tool '{name}': argument '{k}' must be a string"
        if want == "boolean" and not isinstance(v, bool):
            return f"tool '{name}': argument '{k}' must be a boolean"
        if want == "integer" and not (isinstance(v, int) and not isinstance(v, bool)):
            return f"tool '{name}': argument '{k}' must be an integer"
        if want == "number" and not (isinstance(v, (int, float)) and not isinstance(v, bool)):
            return f"tool '{name}': argument '{k}' must be a number"
    return None


def parse_fence(text, tools, forced_name=None):
    """Fence protocol v2 (§3.3).

    Accepts: fenced block (first valid wins), bare top-level array, single
    bare object. Validates names, arguments object, required keys, scalars.
    Returns (calls|None, remaining_text, error|None):
      calls=list -> valid fence; calls=None+error=None -> plain text answer;
      calls=None+error=str -> invalid (repair once).
    """
    valid = set(tool_names(tools))
    spec = _tool_spec(tools)
    if not valid or not text:
        return None, text or "", None
    cands = []  # (kind, raw, drop_pattern_or_None)
    if "```" in text:
        for m in re.finditer(r"```(?:[\w-]+)?\s*\n?(.*?)```", text, re.S):
            b = (m.group(1) or "").strip()
            if b:
                cands.append(("fence", b, m.group(0)))
    s = text.strip()
    if s.startswith("[") or s.startswith("{"):
        cands.append(("bare", s, None))
    if not cands:
        return None, text, None
    first_err = None
    json_attempt = False
    for kind, raw, _ in cands:
        try:
            obj = json.loads(raw)
            json_attempt = True
        except Exception as e:
            if first_err is None:
                first_err = f"fenced block is not valid JSON ({e})"
            continue
        items = obj if isinstance(obj, list) else [obj]
        if not items or not all(isinstance(i, dict) for i in items):
            first_err = first_err or "fence must be a JSON array of {name, arguments} objects"
            continue
        calls, bad = [], None
        for it in items:
            name = it.get("name")
            if not isinstance(name, str) or name not in valid:
                bad = (f"unknown tool '{name}'. Offered: {', '.join(sorted(valid)[:20])}")
                break
            if forced_name and name != forced_name:
                bad = f"tool_choice forces '{forced_name}' but got '{name}'"
                break
            args = it.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except Exception:
                    bad = f"tool '{name}': arguments string is not valid JSON"
                    break
            err = _check_args(name, args, spec)
            if err:
                bad = err
                break
            calls.append({"name": name, "arguments": args})
        if bad:
            first_err = first_err or bad
            continue
        if not calls:
            continue
        if kind == "fence":
            # drop the first fenced block; keep any surrounding prose as content
            remaining = re.sub(r"```(?:[\w-]+)?\s*\n?.*?```", "", text, count=1, flags=re.S).strip()
        else:
            remaining = ""
        return calls, remaining, None
    # Candidates existed but none valid. Repair only real JSON attempts
    # (semantic errors); non-JSON fences are likely code samples -> text.
    if json_attempt and first_err:
        return None, text, first_err
    return None, text, None


def extract_tool_calls(text, tools):
    """Back-compat wrapper: (calls, remaining), invalid -> ([], text)."""
    calls, remaining, _err = parse_fence(text, tools)
    return (calls or [], remaining)


def build_search_discipline():
    """Standing working agreement: memory-first resolution, narrow search.

    Prevents repeats of the 2026-09-21 incident where a 'send me the paper PDF'
    request triggered a full-home `**/*.pdf` glob + 329MB directory read and
    stalled for minutes, when Hermes memory already knew the exact file.
    """
    roots = ", ".join(SEARCH_ROOTS) if SEARCH_ROOTS else "(none configured)"
    return (
        "\n\n[Working agreement — file lookup discipline: "
        "1) MEMORY FIRST: when the user refers to prior work ('the paper', 'it', "
        "'earlier', 'that file', 'send me X'), resolve WHAT/WHERE from this "
        "conversation's history first, then via Hermes memory/session tools if "
        "offered (fence a call), before touching the filesystem. "
        "2) SEARCH NARROW: scope globs to these roots only: " + roots + ". "
        "List the exact directory before any glob. "
        "3) NEVER run recursive `**` globs or directory reads from `/`, "
        "`/home/mitansh`, `~`, or other top-level trees — they stall for minutes. "
        "If the file isn't under the roots, say which roots you checked and ask "
        "the user for the exact path instead of widening the search.]"
    )


def messages_to_prompt(messages, tools=None, tool_choice=None):
    lines, atts = [], []
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    for m in messages or []:
        if isinstance(m, str):
            m = {"role": "user", "content": m}
        if not isinstance(m, dict):
            lines.append(f"USER: {m}")
            continue
        role = (m.get("role") or "user").upper()
        text, imgs = extract_text_and_images(m.get("content"))
        atts.extend(imgs)
        if m.get("tool_calls"):
            try:
                tc_summary = []
                for tc in m["tool_calls"]:
                    fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                    tc_summary.append(f"{tc.get('id', '?')}={fn.get('name', '?')}({str(fn.get('arguments', ''))[:800]})")
                text += "\n[assistant tool_calls in history: " + "; ".join(tc_summary) + "]"
            except Exception:
                text += f"\n[tool_calls in history: {json.dumps(m.get('tool_calls'))[:2000]}]"
        if m.get("tool_call_id"):
            role = f"TOOL_RESULT({m.get('name') or m.get('tool_call_id')})"
            if len(text) > 3000:
                text = text[:3000] + "...(truncated)"
        lines.append(f"{role}: {text}")
    prompt = "\n\n".join(lines).strip()
    if atts:
        kinds = sorted({("image" if m.startswith("image/") else "video" if m.startswith("video/")
                         else "audio" if m.startswith("audio/") else "pdf") for m, _r, _f in atts})
        prompt += f"\n\n[{len(atts)} attached file(s) follow as vision/file input ({', '.join(kinds)}). Inspect them via the attached files.]"
    if not prompt.strip() and atts:
        prompt = ("Describe the attached file(s) in detail. If video, summarize key moments. "
                  "If PDF, summarize contents. If audio, transcribe/summarize. " + prompt).strip()
    if tools:
        instr = build_tools_instruction(tools, tool_choice)
        prompt += instr if instr else (
            "\n\n[Context: no usable Hermes tool names found; answer in plain text.]"
        )
    prompt += build_search_discipline()
    return (prompt or "Say hi"), atts


def _serve_call(method, path, payload=None, timeout=30):
    """Low-level JSON call to the warm opencode serve daemon."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        SERVE_URL + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return True, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode()[:2000]
        except Exception:
            detail = ""
        return False, f"serve HTTP {e.code}: {detail}"
    except Exception as e:
        return False, f"serve unreachable: {e}"


def run_opencode(prompt, images=None, file_parts=None, serve_format=None):
    """Fast path: warm serve daemon (no subprocess cold boot).

    Preferred: pass ready serve file-parts via file_parts (see
    prepare_file_inputs()). Legacy: images as normalized
    [(mime, data_url, filename[, src_path])] tuples are converted here.
    serve_format optionally carries opencode's native {"type":"json_schema",...}
    OutputFormat for response_format (§A.6).
    """
    ok, health = _serve_call("GET", "/global/health", timeout=15)
    if not ok:
        return False, "", f"opencode serve down ({health}). Check `systemctl --user status opencode-serve`."
    if file_parts is None:
        norm = []
        for item in images or []:
            if len(item) >= 3:
                norm.append(item[:3])
            elif len(item) == 2:
                mime, ref = item
                try:
                    norm.extend([t[:3] for t in normalize_attachments([(mime, ref, None)])])
                except Exception as e:
                    return False, "", f"bad attachment: {e}"
            else:
                return False, "", f"bad attachment entry: {item!r}"[:500]
        file_parts = []
        for mime, data_url, filename in norm:
            fp = {"type": "file", "mime": mime, "url": data_url}
            if filename:
                fp["filename"] = os.path.basename(str(filename))[:120]
            file_parts.append(fp)
    parts = [{"type": "text", "text": prompt}] + list(file_parts or [])
    ok, sess = _serve_call("POST", "/session", {}, timeout=15)
    if not ok or "id" not in sess:
        return False, "", f"session create failed: {sess}"
    sid = sess["id"]
    try:
        resolved_model = _resolve_model(MODEL_ID)
        _msg_body = {"model": {"providerID": "opencode", "modelID": resolved_model},
                     "parts": parts}
        if serve_format is not None:
            _msg_body["format"] = serve_format
        ok, resp = _serve_call(
            "POST", f"/session/{sid}/message",
            _msg_body,
            timeout=TIMEOUT,
        )
        if not ok:
            return False, "", str(resp)
        info = resp.get("info", {}) if isinstance(resp, dict) else {}
        info_err = info.get("error") if isinstance(info, dict) else None
        if info_err:
            try:
                detail = json.dumps(info_err)[:2000]
            except Exception:
                detail = str(info_err)[:2000]
            return False, "", f"serve model error: {detail}"
        texts = [p.get("text", "") for p in resp.get("parts", []) if p.get("type") == "text"]
        out = "\n".join(t for t in texts if t).strip()
        return True, out, ""
    finally:
        try:
            _serve_call("DELETE", f"/session/{sid}", timeout=10)
        except Exception:
            pass


def run_opencode_subprocess(prompt):
    """Legacy cold path (kept for reference, unused)."""
    cmd = [OPENCODE_BIN, "run", "-m", OPENCODE_MODEL, prompt]
    # opencode needs HOME etc; inherit env
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=TIMEOUT, cwd=WORKDIR
        )
    except subprocess.TimeoutExpired:
        return False, "", f"opencode timed out after {TIMEOUT}s"
    except FileNotFoundError:
        return False, "", f"opencode binary not found: {OPENCODE_BIN}"
    except Exception as e:
        return False, "", f"opencode launch failed: {e}"
    if p.returncode != 0:
        err = (p.stderr or p.stdout or f"exit {p.returncode}").strip()
        return False, p.stdout.strip(), err
    return True, p.stdout.strip(), p.stderr.strip()


def chat_completion_tool_response(model, content, calls, tcs=None, cid=None,
                                    created=None, usage=None, finish_reason="tool_calls"):
    if tcs is None:
        tcs = _openai_tool_calls(calls)
    return {
        "id": cid or f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": created or int(time.time()),
        "model": model or MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, "tool_calls": tcs},
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _sse_send_usage_chunk(self, cid, created, req_model, usage):
    """Terminal usage-only chunk for stream_options.include_usage (§A.9)."""
    usage_chunk = {"id": cid, "object": "chat.completion.chunk", "created": created,
                   "model": req_model, "choices": [], "usage": usage}
    self.wfile.write(f"data: {json.dumps(usage_chunk)}\n\n".encode())


def _sse_send_tool_calls(self, cid, created, req_model, content, calls, tcs=None,
                         finish_reason="tool_calls", usage=None, include_usage=False):
    """SSE variant for tool_calls responses (Hermes streams with stream:true)."""
    try:
        if tcs is None:
            tcs = _openai_tool_calls(calls or [])
        delta_calls = []
        for i, tc in enumerate(tcs):
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            delta_calls.append({
                "index": i,
                "id": tc.get("id", f"call_{uuid.uuid4().hex[:12]}"),
                "type": "function",
                "function": {"name": fn.get("name"), "arguments": fn.get("arguments")},
            })
        # keep ids stable between delta and final not required by most clients
        chunk = {"id": cid, "object": "chat.completion.chunk", "created": created,
                 "model": req_model,
                 "choices": [{"index": 0,
                              "delta": {"role": "assistant", "content": content,
                                        "tool_calls": delta_calls},
                              "finish_reason": None}]}
        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        done = {"id": cid, "object": "chat.completion.chunk", "created": created,
                "model": req_model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]}
        self.wfile.write(f"data: {json.dumps(done)}\n\n".encode())
        if include_usage and usage is not None:
            _sse_send_usage_chunk(self, cid, created, req_model, usage)
        self.wfile.write(b"data: [DONE]\n\n")
    except (BrokenPipeError, ConnectionResetError):
        pass


def _sse_send_text(self, cid, created, req_model, out, finish_reason="stop",
                   usage=None, include_usage=False):
    try:
        chunk = {"id": cid, "object": "chat.completion.chunk", "created": created,
                 "model": req_model,
                 "choices": [{"index": 0, "delta": {"role": "assistant", "content": out}, "finish_reason": None}]}
        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        done = {"id": cid, "object": "chat.completion.chunk", "created": created,
                "model": req_model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]}
        self.wfile.write(f"data: {json.dumps(done)}\n\n".encode())
        if include_usage and usage is not None:
            _sse_send_usage_chunk(self, cid, created, req_model, usage)
        self.wfile.write(b"data: [DONE]\n\n")
    except (BrokenPipeError, ConnectionResetError):
        pass


def chat_completion_response(model, content, cid=None, created=None, usage=None,
                               finish_reason="stop"):
    return {
        "id": cid or f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": created or int(time.time()),
        "model": model or MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "OpencodeShim/2.6-phase4"

    def log_message(self, fmt, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}")

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def _model_object(self):
        return {
            "id": MODEL_ID,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "opencode-shim",
        }

    def do_GET(self):
        from urllib.parse import unquote
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path in ("/v1/models", "/models"):
            models = _ranked_models()
            self._json(200, {
                "object": "list",
                "data": models,
            })
        elif path in ("/v1/models/" + MODEL_ID, "/models/" + MODEL_ID):
            self._json(200, self._model_object())
        elif path.startswith(("/v1/models/", "/models/")):
            # OpenAI-compatible retrieve: check discovered models (full or short name), 404 if unknown.
            mid = unquote(path.rsplit("/", 1)[-1])
            models = _ranked_models()
            # build lookup for both full and short
            model_ids = set()
            short_map = {}
            for m in models:
                fid = m["id"]
                model_ids.add(fid)
                if "/" in fid:
                    short_map[fid.split("/")[-1]] = fid
            if mid in model_ids or mid in short_map:
                lookup = short_map.get(mid, mid)
                self._json(200, {"id": lookup, "object": "model",
                                  "created": int(time.time()), "owned_by": "opencode-shim"})
            else:
                self._json(404, _openai_error(f"model not found: {mid}", "invalid_request",
                                              param="model", code="model_not_found"))
        elif path in ("/health", "/v1/health", "/"):
            self._json(200, {"ok": True, "model": OPENCODE_MODEL, "mode": "opencode-serve", "serve": SERVE_URL,
                             "models": _ranked_models(),
                             "fallback_models": _fallback_chain(MODEL_ID)[:8],
                             "fallback_enabled": SHIM_MODEL_FALLBACK,
                             "attachments_supported": ["image", "pdf", "text/*"],
                             "attachments_converted": ["any UTF-8 text (json, csv, code, ...) -> text/plain part"],
                             "attachments_staged": ["other binary (zip, apk, tar, ...) -> tool-readable path in WORKDIR inbox"],
                             "attachments_pending": ["video", "audio"],
                             "function_calling": True,
                             "attachments_note": "opencode serve 1.18.31 rejects video/audio file parts; set SHIM_ENABLE_VIDEO/AUDIO=1 to forward anyway on newer serve."})
        elif path in ("/debug/sessions", "/v1/debug/sessions"):
            if SHIM_SESSIONS:
                try:
                    info = get_store().debug_sessions()
                except Exception as e:
                    info = {"error": str(e)[:200]}
                info["mode"] = "sessions"
                self._json(200, info)
            else:
                # Kill switch active: v2.2 flatten path, no index.
                with _stats_lock:
                    n = _stats["requests"]
                self._json(200, {"mode": "flatten", "sessions": "flatten-mode (SHIM_SESSIONS=0)",
                                 "index_size": 0, "requests": n,
                                 "note": "prefix hit rate N/A in v2.2 flatten path"})
        elif path in ("/debug/last", "/v1/debug/last"):
            with _stats_lock:
                recent = list(_req_log)[-10:]
                last = dict(_last_payload)
            self._json(200, {"recent": recent, "last_payload": last})
        elif path in ("/debug/stats", "/v1/debug/stats"):
            with _stats_lock:
                totals = list(_stats["total_ms"])
                pbytes = list(_stats["prompt_bytes"])
                obj = {"requests": _stats["requests"], "errors": _stats["errors"],
                       "fence_ok": _stats["fence_ok"], "fence_repaired": _stats["fence_repaired"],
                       "fence_failed": _stats["fence_failed"],
                       "tool_calls_total": _stats["tool_calls_total"],
                       "p50_total_ms": _percentile(totals, 50), "p95_total_ms": _percentile(totals, 95),
                       "mean_prompt_bytes": (sum(pbytes) // len(pbytes)) if pbytes else 0,
                       "uptime_s": int(time.time() - _SHIM_START),
                       "catchup_fires": _stats.get("catchup_fires", 0),
                       "empty_nudges": _stats.get("empty_nudges", 0),
                       "usage_estimated": _stats.get("usage_estimated", 0),
                       "forks_max_turns": _stats.get("forks_max_turns", 0),
                       "zen_429": _stats.get("zen_429", 0)}
                ftot = _stats["fence_ok"] + _stats["fence_repaired"] + _stats["fence_failed"]
                obj["fence_ok_rate"] = round(
                    (_stats["fence_ok"] + _stats["fence_repaired"]) / ftot, 4) if ftot else 1.0
                # percentages for observability
                if obj["requests"]:
                    obj["catchup_pct"] = round(obj["catchup_fires"] / obj["requests"], 4)
                    obj["empty_nudge_pct"] = round(obj["empty_nudges"] / obj["requests"], 4)
                    _est = obj["usage_estimated"]
                    obj["usage_estimated_pct"] = round(_est / obj["requests"], 4)
            self._json(200, obj)
        else:
            self._json(404, _openai_error(f"not found: {path}", "not_found"))

    def do_POST(self):
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path not in ("/v1/chat/completions", "/chat/completions"):
            self._json(404, _openai_error(f"not found: {path}", "not_found"))
            return
        t0 = time.time()
        req_id = f"r_{uuid.uuid4().hex[:6]}"
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            body = json.loads(raw.decode() or "{}")
        except Exception:
            self._json(400, _openai_error("invalid JSON body", "invalid_request",
                                               code="invalid_json"))
            return

        # --- OpenAI compat guards (PLAN v5 Phase 1): fail fast, before any
        # opencode call, so silent-wrongness becomes a loud 400. ---
        _guard = _compat_guard(body)
        if _guard is not None:
            _gcode, _gebody = _guard
            self._json(_gcode, _gebody)
            return

        messages = body.get("messages", [])
        if not isinstance(messages, list) or len(messages) == 0:
            self._json(400, _openai_error(
                "messages array must not be empty", "invalid_request",
                param="messages", code="empty_messages"))
            return
        tools = body.get("tools")
        tool_choice = body.get("tool_choice")
        # Legacy function-calling alias (§A.8): only when modern tools absent.
        if not tools and body.get("functions"):
            try:
                tools = [{"type": "function", "function": f} for f in body["functions"]]
                tool_choice = tool_choice or body.get("function_call")
            except Exception:
                pass
        stream = bool(body.get("stream"))
        req_model = body.get("model") or MODEL_ID
        n_msgs = len(messages) if isinstance(messages, list) else 1
        n_tools = len(tools) if isinstance(tools, list) else 0
        # Compat params (PLAN v5 §A): stop / token cap / parallel calls /
        # response_format / streaming usage flag.
        stop = body.get("stop")
        max_tokens = body.get("max_completion_tokens")
        if max_tokens is None:
            max_tokens = body.get("max_tokens")
        parallel_tool_calls = body.get("parallel_tool_calls")
        response_format = body.get("response_format")
        # Single kill-switch (currently always None — see function docstring).
        serve_format = _serve_format_for_response_format(response_format)
        include_usage = bool((body.get("stream_options") or {}).get("include_usage")) \
            if isinstance(body.get("stream_options"), dict) else False
        # Sampling params (§A.5): opencode serve's message body
        # (additionalProperties:false) exposes no temperature/top_p/penalty
        # fields, so these cannot be forwarded. Deliberately accepted as a
        # no-op — rejecting would break every standard OpenAI client (SDKs
        # send temperature by default) — with a one-line log for visibility.
        _sampling = {k: body.get(k) for k in
                     ("temperature", "top_p", "presence_penalty", "frequency_penalty")
                     if body.get(k) is not None}
        # Surfaced to the client (body field on JSON responses, Warning
        # header on SSE) so the no-op is visible, not silent (§A.5).
        _warnings = [f"{k} ignored: no upstream field in opencode serve message body"
                     for k in sorted(_sampling)] if _sampling else []
        if _sampling:
            print(f"[compat] {req_id} sampling params accepted as no-op "
                  f"(no upstream field): {_sampling}")
        if SHIM_SESSIONS and isinstance(messages, list) and messages:
            try:
                profile_override = self.headers.get("X-Shim-Profile")
            except Exception:
                profile_override = None
            _do_session_turn(self, t0, req_id, messages, tools, tool_choice,
                             stream, req_model, n_msgs, n_tools,
                             profile_override=profile_override,
                             stop=stop, max_tokens=max_tokens,
                             parallel_tool_calls=parallel_tool_calls,
                             response_format=response_format,
                             include_usage=include_usage,
                             serve_format=serve_format,
                             warnings=_warnings)
            return
        prompt, atts_raw = messages_to_prompt(messages, tools, tool_choice)
        # Sole enforcement path (native serve format kill-switched); see note
        # on the session path above.
        if response_format:
            prompt += _response_format_instruction(response_format)
        if tools:
            try:
                print(f"[tools] n={len(tools)} names={tool_names(tools)[:15]} choice={str(tool_choice)[:120]}")
            except Exception:
                pass

        # Normalize attachments BEFORE acquiring the LLM slot: downloads/reads
        # must not hold semaphore. Bad files -> fast 400, no retry burn.
        def _fail(code, msg, finish, param=None, code_slug=None):
            total_ms = int((time.time() - t0) * 1000)
            entry = {"id": req_id, "sess": "-", "hit": 0, "depth": n_msgs,
                     "delta_msgs": n_msgs, "delta_bytes": len(prompt.encode()),
                     "prompt_bytes": len(prompt.encode()),
                     "tools": n_tools, "tools_sent": n_tools,
                     "sys_drift": 0, "forked": 0, "profile": "-",
                     "attach": len(atts_raw), "stream": int(bool(stream)),
                     "fence": "-", "tool_calls": 0,
                     "total_ms": total_ms, "finish": finish}
            _record_request(entry)
            print(f"[req] id={req_id} sess=- hit=0 depth={n_msgs} "
                  f"delta_msgs={n_msgs} delta_bytes={len(prompt.encode())} "
                  f"tools={n_tools} tools_sent={n_tools} sys_drift=0 forked=0 profile=- "
                  f"attach={len(atts_raw)} stream={int(bool(stream))} fence=- tool_calls=0 "
                  f"total_ms={total_ms} finish={finish}")
            _ft = "invalid_request" if code == 400 else ("rate_limit" if code == 429 else "backend_error")
            self._json(code, _openai_error(msg, _ft, param=param, code=code_slug))

        try:
            normed = normalize_attachments(atts_raw)
        except ValueError as e:
            _fail(400, e, "400", param="messages", code_slug="invalid_attachment")
            return
        except Exception as e:
            _fail(400, f"bad attachment: {e}", "400", param="messages", code_slug="invalid_attachment")
            return

        # Split into serve file-parts vs tool-readable inbox notes. Gated
        # video/audio raise ValueError -> explicit 400 with workaround.
        try:
            file_parts, file_notes = prepare_file_inputs(normed)
        except ValueError as e:
            _fail(400, e, "400", param="messages", code_slug="unsupported_media")
            return
        except Exception as e:
            _fail(400, f"bad attachment: {e}", "400", param="messages", code_slug="invalid_attachment")
            return
        if file_notes:
            prompt += "\n\n" + "\n".join(file_notes)

        if file_parts or file_notes:
            try:
                total_b64 = sum(len(p.get("url", "")) for p in file_parts)
                mimes = sorted({p.get("mime", "?") for p in file_parts})
                print(f"[attachments] parts={len(file_parts)} notes={len(file_notes)} mimes={mimes} b64chars={total_b64}")
            except Exception:
                pass

        # Phase 0 payload capture: redacted copy always kept; full dump opt-in.
        prompt_bytes = len(prompt.encode())
        try:
            with _stats_lock:
                _last_payload.update({"redacted": _redact_payload(prompt[:8000], file_parts),
                                      "prompt_bytes": prompt_bytes, "time": int(time.time())})
            if SHIM_DEBUG_DUMP:
                os.makedirs("/tmp/shim-dumps", exist_ok=True)
                with open(f"/tmp/shim-dumps/{req_id}.json", "w") as f:
                    json.dump({"prompt": prompt, "file_parts": file_parts,
                               "n_msgs": n_msgs, "n_tools": n_tools}, f)
        except Exception as e:
            print(f"[debug] payload capture failed: {e}")

        # Queue instead of instant 429: Hermes retries overlap while opencode runs ~20s
        if not _sem.acquire(blocking=True, timeout=280):
            _fail(429, "busy: another opencode run in progress, retry shortly", "429")
            return
        try:
            ok, out, err = run_opencode(prompt, file_parts=file_parts,
                                        serve_format=serve_format)
        finally:
            _sem.release()

        if not ok:
            msg = err or "opencode run failed"
            _low2 = (err or "").lower()
            if "429" in (err or "") or any(kw in _low2 for kw in ["rate limit", "quota", "too many requests", "overloaded", "capacity"]):
                with _stats_lock:
                    _stats["zen_429"] += 1
                msg += "\n\nZen backend rate-limited (429). Back off with jitter and retry."
                status = 429
            elif "auth" in msg.lower() or "401" in msg or "unauthorized" in msg or "api key" in msg.lower():
                msg += ("\n\nHint: one-time setup needed: run `opencode auth login` -> OpenCode Zen "
                        "(free, no card at opencode.ai/auth), then retry. Hermes side stays keyless.")
                status = 401
            else:
                status = 401 if "401" in err or "unauthorized" in err.lower() else 500
            total_ms = int((time.time() - t0) * 1000)
            entry = {"id": req_id, "sess": "-", "hit": 0, "depth": n_msgs,
                     "delta_msgs": n_msgs, "delta_bytes": prompt_bytes,
                     "prompt_bytes": prompt_bytes,
                     "tools": n_tools, "tools_sent": n_tools,
                     "sys_drift": 0, "forked": 0, "profile": "-",
                     "attach": len(normed), "stream": int(bool(stream)),
                     "fence": "-", "tool_calls": 0,
                     "total_ms": total_ms, "finish": str(status)}
            _record_request(entry)
            print(f"[req] id={req_id} sess=- hit=0 depth={n_msgs} "
                  f"delta_msgs={n_msgs} delta_bytes={prompt_bytes} "
                  f"tools={n_tools} tools_sent={n_tools} sys_drift=0 forked=0 profile=- "
                  f"attach={len(normed)} stream={int(bool(stream))} fence=- tool_calls=0 "
                  f"total_ms={total_ms} finish={status}")
            _ebody = _openai_error(msg, "rate_limit" if status == 429 else "backend_error")
            _ebody["error"]["output"] = out[:2000]
            self._json(status, _ebody)
            return

        # Tool-call bridge: if Hermes offered tools, let opencode's output decide.
        # Fenced ```hermes-toolcalls JSON -> OpenAI tool_calls response (Hermes
        # executes tools and loops back with role:tool results). Anything else ->
        # plain text (opencode may still have used its own tools internally).
        calls, remaining = ([], out)
        fence = "-"
        if tools and tool_choice != "none":
            try:
                calls, remaining = extract_tool_calls(out, tools)
            except Exception as e:
                print(f"[tools] parse failed: {e}")
                calls, remaining = [], out
            calls = _limit_parallel_calls(calls, parallel_tool_calls)
            if calls:
                fence = "ok"
                print(f"[tools] emitting {len(calls)} call(s): {[c['name'] for c in calls]}")
                _tool_text, _ = _apply_stop(remaining or "", stop)
                _tool_text, _ = _apply_max_tokens(_tool_text, max_tokens)
                remaining, out = _tool_text, _tool_text
                # Never "length" alongside a valid tool call (see session path).
                _tool_finish = "tool_calls"
                total_ms = int((time.time() - t0) * 1000)
                entry = {"id": req_id, "sess": "-", "hit": 0, "depth": n_msgs,
                         "delta_msgs": n_msgs, "delta_bytes": prompt_bytes,
                         "prompt_bytes": prompt_bytes,
                         "tools": n_tools, "tools_sent": n_tools,
                         "sys_drift": 0, "forked": 0, "profile": "-",
                         "attach": len(normed), "stream": int(bool(stream)),
                         "fence": fence, "tool_calls": len(calls),
                         "total_ms": total_ms, "finish": _tool_finish}
                _record_request(entry)
                print(f"[req] id={req_id} sess=- hit=0 depth={n_msgs} "
                      f"delta_msgs={n_msgs} delta_bytes={prompt_bytes} "
                      f"tools={n_tools} tools_sent={n_tools} sys_drift=0 forked=0 profile=- "
                      f"attach={len(normed)} stream={int(bool(stream))} fence={fence} tool_calls={len(calls)} "
                      f"total_ms={total_ms} finish={_tool_finish}")
                resp = chat_completion_tool_response(req_model, remaining or None, calls,
                                                     finish_reason=_tool_finish)
                if _warnings:
                    resp["x_shim_warnings"] = list(_warnings)
                if not stream:
                    self._json(200, resp)
                    return
                cid, created = resp["id"], resp["created"]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                if _warnings:
                    self.send_header("Warning", _warnings_header(_warnings))
                self._cors()
                self.end_headers()
                _sse_send_tool_calls(self, cid, created, req_model, remaining or None, calls,
                                     finish_reason=_tool_finish, usage=resp.get("usage"),
                                     include_usage=include_usage)
                return
            out = remaining
            fence = "text"

        # response_format in flatten mode (no session for a repair turn):
        # prompt instruction was already injected. If the output still does
        # not validate, fail loudly (502) rather than shipping malformed
        # JSON as a "successful" 200 (§A.6).
        if response_format and not calls:
            _rf_ok, _rf_err = _validate_response_format(out, response_format)
            if not _rf_ok:
                _fail(502, ("response_format could not be satisfied: model did not "
                            f"produce valid JSON ({(_rf_err or 'unknown')[:300]})"),
                      "502")
        _flat_out, _ = _apply_stop(out or "", stop)
        _flat_out, _flat_hit_max = _apply_max_tokens(_flat_out, max_tokens)
        out = _flat_out
        _flat_finish = "length" if _flat_hit_max else "stop"
        total_ms = int((time.time() - t0) * 1000)
        entry = {"id": req_id, "sess": "-", "hit": 0, "depth": n_msgs,
                 "delta_msgs": n_msgs, "delta_bytes": prompt_bytes,
                 "prompt_bytes": prompt_bytes,
                 "tools": n_tools, "tools_sent": n_tools,
                 "sys_drift": 0, "forked": 0, "profile": "-",
                 "attach": len(normed), "stream": int(bool(stream)),
                 "fence": fence, "tool_calls": 0,
                 "total_ms": total_ms, "finish": _flat_finish}
        _record_request(entry)
        print(f"[req] id={req_id} sess=- hit=0 depth={n_msgs} "
              f"delta_msgs={n_msgs} delta_bytes={prompt_bytes} "
              f"tools={n_tools} tools_sent={n_tools} sys_drift=0 forked=0 profile=- "
              f"attach={len(normed)} stream={int(bool(stream))} fence={fence} tool_calls=0 "
              f"total_ms={total_ms} finish={_flat_finish}")
        resp = chat_completion_response(req_model, out, finish_reason=_flat_finish)
        if _warnings:
            resp["x_shim_warnings"] = list(_warnings)
        if not stream:
            self._json(200, resp)
            return

        # Minimal SSE streaming: one delta + DONE (satisfies Hermes stream:true)
        cid = resp["id"]
        created = resp["created"]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        if _warnings:
            self.send_header("Warning", _warnings_header(_warnings))
        self._cors()
        self.end_headers()
        _sse_send_text(self, cid, created, req_model, out, finish_reason=_flat_finish,
                       usage=resp.get("usage"), include_usage=include_usage)


def main():
    global WORKDIR
    if HOST not in ("127.0.0.1", "localhost", "::1"):
        print(f"WARN: SHIM_HOST={HOST} is not loopback; shim is designed keyless localhost-only.")
    if not os.path.isdir(WORKDIR):
        print(f"WARN: WORKDIR {WORKDIR} missing, using /home/mitansh")
        WORKDIR = "/home/mitansh"
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"opencode-shim listening on http://{HOST}:{PORT}/v1 (keyless, localhost-only)")
    print(f"  backend: warm serve {SERVE_URL} model={OPENCODE_MODEL}")
    print(f"  caps: image<={MAX_IMAGE_BYTES} video<={MAX_VIDEO_BYTES} pdf<={MAX_PDF_BYTES} audio<={MAX_AUDIO_BYTES}")
    print(f"  forward: image+pdf always; video={'on' if ENABLE_VIDEO else 'gated-400'} audio={'on' if ENABLE_AUDIO else 'gated-400'}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

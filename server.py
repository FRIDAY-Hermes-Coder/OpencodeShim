#!/usr/bin/env python3
"""Keyless local OpenAI-compatible shim for Hermes -> on-device opencode.

Hermes (keyless, dummy Bearer) -> http://127.0.0.1:8000/v1 -> warm `opencode serve`
with opencode/muse-spark-1.3-contributor-free (Zen free, auth stored in opencode).

Stdlib only. Binds 127.0.0.1 (localhost-only, no auth needed).
Media (image/video/pdf/audio) is normalized to data: URLs because serve accepts
data: and file:// but rejects bare paths (500) and http(s) URLs (400).
"""
import base64
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

HOST = os.environ.get("SHIM_HOST", "127.0.0.1")
PORT = int(os.environ.get("SHIM_PORT", "8000"))
MODEL_ID = os.environ.get("SHIM_MODEL_ID", "muse-spark-1.3-contributor-free")
OPENCODE_MODEL = os.environ.get("SHIM_OPENCODE_MODEL", "opencode/muse-spark-1.3-contributor-free")
OPENCODE_BIN = os.environ.get("SHIM_OPENCODE_BIN", "/home/mitansh/.opencode/bin/opencode")
SERVE_URL = os.environ.get("SHIM_SERVE_URL", "http://127.0.0.1:4096").rstrip("/")
WORKDIR = os.environ.get("SHIM_WORKDIR", "/home/mitansh/hermesworkspace")
TIMEOUT = int(os.environ.get("SHIM_TIMEOUT", "300"))
MAX_CONCURRENT = int(os.environ.get("SHIM_MAX_CONCURRENT", "4"))
IMAGE_TIMEOUT = int(os.environ.get("SHIM_IMAGE_TIMEOUT", "20"))

_sem = threading.Semaphore(MAX_CONCURRENT)


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
# opencode serve 1.18.31 rejects video/* + audio/* file parts
# ("'file part media type video/mp4/audio/wav' functionality not supported"),
# even though model caps list them. Gate them with a clear 400 until serve
# supports them; set SHIM_ENABLE_VIDEO/AUDIO=1 to forward anyway on newer serve.
ENABLE_VIDEO = os.environ.get("SHIM_ENABLE_VIDEO", "0") == "1"
ENABLE_AUDIO = os.environ.get("SHIM_ENABLE_AUDIO", "0") == "1"


def is_forwardable(mime):
    m = (mime or "").lower()
    if m.startswith("image/") or m == "application/pdf":
        return True
    if m.startswith("video/"):
        return ENABLE_VIDEO
    if m.startswith("audio/"):
        return ENABLE_AUDIO
    return False


def limit_for_mime(mime):
    m = (mime or "").lower()
    if m.startswith("video/"):
        return MAX_VIDEO_BYTES
    if m == "application/pdf":
        return MAX_PDF_BYTES
    if m.startswith("audio/"):
        return MAX_AUDIO_BYTES
    return MAX_IMAGE_BYTES


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
        mime = header[5:].split(";")[0].strip() or "image/jpeg"
        if not _allowed_mime(mime):
            raise ValueError(f"unsupported data URL mime {mime}. Send image/video/pdf/audio only. [image_url]")
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


def file_to_data_url(ref):
    """Read file:// or bare local path -> (mime, data: URL, filename)."""
    raw_ref = ref
    path = ref.replace("file://", "", 1)
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(WORKDIR, path)
    path = os.path.normpath(path)
    if not os.path.isfile(path):
        raise ValueError(f"local file not found: {raw_ref} (resolved {path}) [image_url]")
    ext = os.path.splitext(path)[1].lower()
    mime = MIME_BY_EXT.get(ext, "image/jpeg")
    if not _allowed_mime(mime):
        raise ValueError(f"unsupported file type {ext or '?'} ({mime}) for {path}. Send image/video/pdf/audio only. [image_url]")
    limit = limit_for_mime(mime)
    size = os.path.getsize(path)
    if size > limit:
        raise ValueError(f"file too large ({size} bytes, max {limit} for {mime}): {path}. Please shrink and retry. [image_url]")
    with open(path, "rb") as f:
        raw = f.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f"file too large (max {limit} for {mime}): {path} [image_url]")
    return mime, f"data:{mime};base64," + base64.b64encode(raw).decode(), os.path.basename(path)


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
            _mime_g, fn_guess = _guess_mime_and_name(url)
            hard_cap = max(MAX_IMAGE_BYTES, MAX_VIDEO_BYTES, MAX_PDF_BYTES, MAX_AUDIO_BYTES) + 1
            raw = r.read(hard_cap)
            mime = None
            if ctype and "/" in ctype and _allowed_mime(ctype):
                mime = ctype
            if not mime:
                mime = _mime_g
            if not _allowed_mime(mime):
                raise ValueError(
                    f"unsupported Content-Type {ctype or mime} for {url}. Send image/video/pdf/audio only. [image_url]")
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
    """Convert [(mime_guess, ref, filename)] -> [(mime, data_url, filename)]. All data: URLs for serve."""
    out = []
    for _mime_g, ref, fn_g in atts_raw or []:
        if not ref or not isinstance(ref, str):
            continue
        if ref.startswith("data:"):
            mime, data_url, _ = validate_data_url(ref)
            out.append((mime, data_url, fn_g))
        elif ref.startswith(("http://", "https://")):
            mime, data_url, fn = download_http_to_data_url(ref)
            out.append((mime, data_url, fn or fn_g))
        else:
            mime, data_url, fn = file_to_data_url(ref)
            out.append((mime, data_url, fn or fn_g))
    return out


def load_url_or_path(url):
    """Back-compat single-item normalizer (data: passthrough, http download, file read)."""
    mime_g, fn_g = _guess_mime_and_name(url)
    out = normalize_attachments([(mime_g, url, fn_g)])
    return out[0][1] if out else url


def extract_text(content):
    text, _ = extract_text_and_images(content)
    return text


TOOLCALL_FENCE = "hermes-toolcalls"
TOOLS_JSON_BUDGET = int(os.environ.get("SHIM_TOOLS_JSON_BUDGET", "24000"))


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


def extract_tool_calls(text, tools):
    """Parse opencode output for a Hermes tool-call fence.

    Returns (calls, remaining_text) where calls = [{"name","arguments":dict}].
    Falls back to ([], text) when no valid fence is found.
    """
    valid = set(tool_names(tools))
    if not valid or not text or "```" not in text:
        # also accept a bare JSON array as the whole message
        if text:
            s = text.strip()
            if s.startswith("[") and valid:
                try:
                    obj = json.loads(s)
                except Exception:
                    return [], text
                calls, ok = _coerce_calls(obj, valid)
                if ok:
                    return calls, ""
        return [], text
    blocks = re.findall(r"```(?:[\w-]+)?\s*\n?(.*?)```", text, re.S)
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        try:
            obj = json.loads(b)
        except Exception:
            continue
        calls, ok = _coerce_calls(obj if isinstance(obj, list) else [obj], valid)
        if ok:
            # drop the first fenced block; keep any surrounding prose as content
            remaining = re.sub(r"```(?:[\w-]+)?\s*\n?.*?```", "", text, count=1, flags=re.S).strip()
            return calls, remaining
    return [], text


def _coerce_calls(items, valid):
    calls = []
    for it in items:
        if not isinstance(it, dict):
            return [], False
        name = it.get("name")
        if not isinstance(name, str) or name not in valid:
            return [], False
        args = it.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except Exception:
                return [], False
        if not isinstance(args, dict):
            return [], False
        calls.append({"name": name, "arguments": args})
    return (calls, True) if calls else ([], False)


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


def run_opencode(prompt, images=None):
    """Fast path: warm serve daemon (no subprocess cold boot).

    images must already be normalized [(mime, data_url, filename)] — see
    normalize_attachments(). Kept tolerant: raw (mime, url) pairs are normalized
    here as a fallback (holds the semaphore; prefer normalizing before acquire).
    """
    ok, health = _serve_call("GET", "/global/health", timeout=15)
    if not ok:
        return False, "", f"opencode serve down ({health}). Check `systemctl --user status opencode-serve`."
    norm = []
    for item in images or []:
        if len(item) == 3:
            norm.append(item)
        elif len(item) == 2:
            mime, ref = item
            try:
                norm.extend(normalize_attachments([(mime, ref, None)]))
            except Exception as e:
                return False, "", f"bad attachment: {e}"
        else:
            return False, "", f"bad attachment entry: {item!r}"[:500]
    parts = [{"type": "text", "text": prompt}]
    for mime, data_url, filename in norm:
        fp = {"type": "file", "mime": mime, "url": data_url}
        if filename:
            fp["filename"] = os.path.basename(str(filename))[:120]
        parts.append(fp)
    ok, sess = _serve_call("POST", "/session", {}, timeout=15)
    if not ok or "id" not in sess:
        return False, "", f"session create failed: {sess}"
    sid = sess["id"]
    try:
        ok, resp = _serve_call(
            "POST", f"/session/{sid}/message",
            {"model": {"providerID": "opencode", "modelID": MODEL_ID},
             "parts": parts},
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


def chat_completion_tool_response(model, content, calls):
    tcs = []
    for c in calls:
        tcs.append({
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])},
        })
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model or MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, "tool_calls": tcs},
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _sse_send_tool_calls(self, cid, created, req_model, content, calls):
    """SSE variant for tool_calls responses (Hermes streams with stream:true)."""
    try:
        delta_calls = []
        for i, c in enumerate(calls):
            delta_calls.append({
                "index": i,
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])},
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
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
        self.wfile.write(f"data: {json.dumps(done)}\n\ndata: [DONE]\n\n".encode())
    except (BrokenPipeError, ConnectionResetError):
        pass


def _sse_send_text(self, cid, created, req_model, out):
    try:
        chunk = {"id": cid, "object": "chat.completion.chunk", "created": created,
                 "model": req_model,
                 "choices": [{"index": 0, "delta": {"role": "assistant", "content": out}, "finish_reason": None}]}
        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        done = {"id": cid, "object": "chat.completion.chunk", "created": created,
                "model": req_model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        self.wfile.write(f"data: {json.dumps(done)}\n\ndata: [DONE]\n\n".encode())
    except (BrokenPipeError, ConnectionResetError):
        pass


def chat_completion_response(model, content):
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model or MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "OpencodeShim/2.1"

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

    def do_GET(self):
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path in ("/v1/models", "/models"):
            self._json(200, {
                "object": "list",
                "data": [{
                    "id": MODEL_ID,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "opencode-shim",
                }],
            })
        elif path in ("/health", "/v1/health", "/"):
            self._json(200, {"ok": True, "model": OPENCODE_MODEL, "mode": "opencode-serve", "serve": SERVE_URL,
                             "attachments_supported": ["image", "pdf"],
                             "attachments_pending": ["video", "audio"],
                             "function_calling": True,
                             "attachments_note": "opencode serve 1.18.31 rejects video/audio file parts; set SHIM_ENABLE_VIDEO/AUDIO=1 to forward anyway on newer serve."})
        else:
            self._json(404, {"error": {"message": f"not found: {path}", "type": "not_found"}})

    def do_POST(self):
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path not in ("/v1/chat/completions", "/chat/completions"):
            self._json(404, {"error": {"message": f"not found: {path}", "type": "not_found"}})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            body = json.loads(raw.decode() or "{}")
        except Exception:
            self._json(400, {"error": {"message": "invalid JSON body", "type": "invalid_request"}})
            return

        messages = body.get("messages", [])
        tools = body.get("tools")
        tool_choice = body.get("tool_choice")
        stream = bool(body.get("stream"))
        req_model = body.get("model") or MODEL_ID
        prompt, atts_raw = messages_to_prompt(messages, tools, tool_choice)
        if tools:
            try:
                print(f"[tools] n={len(tools)} names={tool_names(tools)[:15]} choice={str(tool_choice)[:120]}")
            except Exception:
                pass

        # Normalize attachments BEFORE acquiring the LLM slot: downloads/reads
        # must not hold semaphore. Bad files -> fast 400, no retry burn.
        try:
            images = normalize_attachments(atts_raw)
        except ValueError as e:
            self._json(400, {"error": {"message": str(e)[:2000], "type": "invalid_request"}})
            return
        except Exception as e:
            self._json(400, {"error": {"message": f"bad attachment: {e}"[:2000], "type": "invalid_request"}})
            return

        # Gate media types serve can't handle yet (video/audio) with a clear 400
        # instead of forwarding to an empty/error reply.
        blocked = [(m, f) for m, _u, f in images if not is_forwardable(m)]
        if blocked:
            kinds = sorted({m.split("/")[0] for m, _f in blocked})
            names = ", ".join([f for _m, f in blocked if f][:5])
            self._json(400, {"error": {
                "message": (f"{'/'.join(kinds)} attachment(s) not supported by opencode serve 1.18.31 "
                            f"('file part media type' not supported; model caps list it but serve rejects it). "
                            f"Supported now: image + pdf. Pending: video/audio. "
                            f"Workaround: transcribe/describe the media first and send text. "
                            f"Files: {names or 'unknown'}. [audio/video]"),
                "type": "invalid_request"}})
            return

        if images:
            try:
                total_b64 = sum(len(u) for _m, u, _f in images)
                mimes = sorted({m for m, _u, _f in images})
                print(f"[attachments] n={len(images)} mimes={mimes} b64chars={total_b64}")
            except Exception:
                pass

        # Queue instead of instant 429: Hermes retries overlap while opencode runs ~20s
        if not _sem.acquire(blocking=True, timeout=280):
            self._json(429, {"error": {"message": "busy: another opencode run in progress, retry shortly", "type": "rate_limit"}})
            return
        try:
            ok, out, err = run_opencode(prompt, images)
        finally:
            _sem.release()

        if not ok:
            msg = err or "opencode run failed"
            # Friendly hint for missing Zen auth (opencode stores it separately; shim stays keyless)
            if "auth" in msg.lower() or "401" in msg or "unauthorized" in msg or "api key" in msg.lower():
                msg += ("\n\nHint: one-time setup needed: run `opencode auth login` -> OpenCode Zen "
                        "(free, no card at opencode.ai/auth), then retry. Hermes side stays keyless.")
            status = 401 if "401" in err or "unauthorized" in err.lower() else 500
            self._json(status, {"error": {"message": msg[:4000], "type": "backend_error", "output": out[:2000]}})
            return

        # Tool-call bridge: if Hermes offered tools, let opencode's output decide.
        # Fenced ```hermes-toolcalls JSON -> OpenAI tool_calls response (Hermes
        # executes tools and loops back with role:tool results). Anything else ->
        # plain text (opencode may still have used its own tools internally).
        calls, remaining = ([], out)
        if tools and tool_choice != "none":
            try:
                calls, remaining = extract_tool_calls(out, tools)
            except Exception as e:
                print(f"[tools] parse failed: {e}")
                calls, remaining = [], out
            if calls:
                print(f"[tools] emitting {len(calls)} call(s): {[c['name'] for c in calls]}")
                resp = chat_completion_tool_response(req_model, remaining or None, calls)
                if not stream:
                    self._json(200, resp)
                    return
                cid, created = resp["id"], resp["created"]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self._cors()
                self.end_headers()
                _sse_send_tool_calls(self, cid, created, req_model, remaining or None, calls)
                return
            out = remaining

        resp = chat_completion_response(req_model, out)
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
        self._cors()
        self.end_headers()
        _sse_send_text(self, cid, created, req_model, out)


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

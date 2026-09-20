# opencode-local shim — plan

Date: 2026-09-20. Owner: mitansh. Device: surface (this device only).

## 1. Goal

Keyless, on-device call to opencode. No API keys on Hermes side, no external
API from user perspective:

```
Hermes (provider=custom, base_url=http://127.0.0.1:8000/v1, dummy api_key)
  -> opencode-hermes-shim (127.0.0.1:8000, stdlib only, no auth)
  -> warm `opencode serve` (127.0.0.1:4096)
  -> opencode/muse-spark-1.3-contributor-free (Zen free, auth stored in opencode)
```

User constraint: keep sharing small files only (no large-image handling needed
beyond a sane cap + friendly error). Video support is optional/nice-to-have,
PDF support required. Audio added on request (2026-09-20).

## 2a. Backend media support (measured 2026-09-20, serve 1.18.31)

- `image/*` + `application/pdf` file parts: forwarded, `200` proven
  (png data:/file://, http->data: conversion, tiny pdf).
- `video/*` + `audio/*` file parts: serve returns
  `UnknownError: 'file part media type video/mp4|audio/wav|...' functionality
  not supported`, empty parts — despite model caps `input.audio/video=true`.
- Shim behavior: normalize video/audio (download/read/validate) but gate with
  fast `400 invalid_request` explaining the serve limitation + STT/describe
  workaround. Override with `SHIM_ENABLE_VIDEO=1` / `SHIM_ENABLE_AUDIO=1` on
  newer serve. `run_opencode` also treats `info.error` as failure (no empty-OK).

## 2. Verified current state

- `server.py` (339 lines): `ThreadingHTTPServer` on `127.0.0.1:8000`,
  `/v1/chat/completions` + `/v1/models` + `/health`, SSE single-delta,
  semaphore queue (`MAX_CONCURRENT`, 280s acquire timeout).
- `opencode serve --hostname 127.0.0.1 --port 4096`, cwd
  `/home/mitansh/hermesworkspace`. Model `muse-spark-1.3-contributor-free`:
  `attachment:true`, `input: text/audio/image/video/pdf=true`.
- Systemd user units `opencode-serve.service` + `opencode-shim.service`
  both `enabled` + `active`.
- Hermes `~/.hermes/config.yaml`: `model.default=muse-spark-1.3-contributor-free`,
  `provider=custom`, `base_url=http://127.0.0.1:8000/v1`.
- Live probes (2026-09-20):
  - `data:image/png;base64,...` -> serve `200 IMG_OK`. Works.
  - `file:///tmp/tiny.png` -> `200`. Works (shim converts to `data:`).
  - bare `/tmp/tiny.png` -> serve `500 UnknownError`. Crashes serve.
  - `https://httpbin.org/image/png` as FilePart `url` -> serve `400 BadRequest`.
    Serve does NOT fetch remote URLs. Shim passed `http(s)` through unchanged:
    **this is the bug being fixed**.
- Hermes sends OpenAI `image_url` parts (`agent/plugin_llm.py:_image_part`),
  `data:` URLs for bytes; `http(s)` URLs possible (web/attachments).

## 3. Design: normalize everything to `data:` URLs in the shim

`opencode serve` `FilePartInput = {type:"file", mime, url, filename?}` accepts
`data:` and `file://` but not bare paths or `http(s)`. Shim normalizes all
attachment refs to `data:` URLs before calling serve:

| Incoming ref | Shim action |
|---|---|
| `data:<mime>;base64,...` | validate mime + decoded size, pass through |
| `http://` / `https://` | download (timeout, UA, Content-Type sniff, byte cap), convert to `data:` |
| `file://...` | strip scheme, read file, convert to `data:` + `filename` |
| bare path (`/x`, `~/x`, `./x`, `../x`) | resolve relative against `WORKDIR`, read, convert to `data:` + `filename` |

Limits (env-overridable, user keeps files small):

- `SHIM_MAX_IMAGE_BYTES` default `12MB` (images).
- `SHIM_MAX_VIDEO_BYTES` default `25MB` (video, optional).
- `SHIM_MAX_PDF_BYTES` default `12MB` (pdf).
- `SHIM_IMAGE_TIMEOUT` default `20s` per download.

Allowed mimes: `image/*`, `video/mp4|webm|quicktime (+x-matroska best-effort)`,
`application/pdf`. Others -> friendly `400`.

## 4. Parsing extensions (`extract_text_and_images`)

Handle what Hermes/OpenAI/Responses/Anthropic clients may send:

- `{"type":"text","text":...}` (as before).
- `{"type":"image_url","image_url":{"url":...,"detail":...}}` and
  `{"type":"image_url","image_url":"<str>"}`.
- `{"type":"input_image","image_url": str|{url}}` (Responses API).
- `{"type":"image", ...}` with `url` / `image_url` / `source` dict.
- Anthropic `{"type":"image","source":{"type":"base64","media_type":...,"data":...}}`
  -> `data:` URL.
- `{"type":"video_url","video_url":{"url":...}}` / `{"type":"video",...}` -> video.
- `{"type":"file",...}` with `url`/`filename`/`file_data` -> pdf/video/image.
- Return `(text, [(mime_guess, url, filename_guess)])`; mime guessed from
  `data:` prefix or extension, refined after download/read.

Prompt tweaks in `messages_to_prompt`:

- Empty text + attachments -> `"Describe the attached file(s) in detail. ..."`.
- Append `"[<N> attached file(s) follow as vision input: mime list]"` so the
  agent looks at file parts.
- Keep existing Hermes `tools` hint (agentic-delegation, answer as plain text).

## 5. Concurrency / errors / logging

- Normalize attachments **before** `_sem.acquire()`; downloads must not hold
  LLM slots. Only the serve call holds the semaphore.
- Bad attachment -> `400 {invalid_request}` immediately, message names
  `image_url`/attachment so Hermes can surface it (no 3x LLM retry burn).
- Serve errors keep existing mapping (`401` hint for `opencode auth login`,
  else `500 backend_error`).
- Log one line per request: `n_attachments, mimes, bytes, sources`
  (`data/http/file/path`), never base64 content. Journald via `print()`.

## 6. Keyless guarantees (do not regress)

- Default `SHIM_HOST=127.0.0.1`; warn if bound elsewhere.
- No `Authorization` / API-key checks on `:8000` (localhost-only is the auth).
- No keys in env, repo, logs. Zen auth lives only in opencode
  (`opencode auth login` -> Zen).
- `curl http://127.0.0.1:8000/health` + `/v1/models` stay public-on-loopback.

## 7. Tests (acceptance)

1. Unit (no LLM): parse all shapes above; normalizer for `data/file/bare/http`;
   oversize + missing file -> `ValueError`.
2. Live serve `:4096`: `data:` + `file://` succeed; raw `https:` `400`
   (justifies shim conversion).
3. E2E shim `:8000/v1/chat/completions`:
   - text-only -> `200`.
   - `data:` image -> `200`.
   - `http` image URL (e.g. `https://httpbin.org/image/png`) -> `200` via shim
     download (was `400` direct).
   - local path image -> `200` (was `500` direct).
   - tiny pdf -> `200` / sensible answer.
   - `stream:true` SSE `data: ...` + `data: [DONE]`.
   - bad image (missing file, oversize, bad host) -> `400`, no semaphore hold.

## 8. Rollout

1. Backup `server.py`, apply new version (stdlib only).
2. `python3 -m py_compile server.py`.
3. `systemctl --user restart opencode-shim && systemctl --user status opencode-shim`.
4. Run §7 tests; check `journalctl --user -u opencode-shim -n 50`.
5. Send one real Hermes message (text + small image) via linked channel.

## 9. Non-goals

- No public bind, no auth layer, no TLS (loopback only).
- No downscaling/compression of media (user keeps files small).
- No audio input support in this round (model allows it; add later if needed).
- No Hermes config change needed (already pointed at shim).

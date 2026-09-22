#!/usr/bin/env python3
"""L1 Session Manager (plan_enhanced.md §2) — stdlib only.

Maps one Hermes OpenAI conversation (full history every turn) to one opencode
session, so each turn sends only the NEW messages (delta) instead of a flatten
of the whole thread.

Core pieces:
  - canon() / chain_hashes(): prefix-chain hashing over canonical messages.
  - norm_system(): volatile-insensitive system hash (timestamps, memory blocks).
  - SessionStore: persisted index chain_hash -> {session_id, idx} + session
    records (atomic write, LRU cap, TTL), idempotency cache.
  - resolve(): longest-prefix match -> (session_id | None, delta, meta).
  - projected_hash(): pre-register the chain including our own reply so the
    next request hits.
  - per-session locks: one in-flight turn per conversation.
"""
import hashlib
import json
import os
import re
import threading
import time

VERSION = 3
CHAIN_SEED = "v3|"

SYSTEM_VOLATILE = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:?\d{2})?\b"),
    re.compile(r"(?s)<memory>.*?</memory>"),
    re.compile(r"(?s)<context>.*?</context>"),
    re.compile(r"(?s)<memories>.*?</memories>"),
]
VOLATILE_TOKEN = "<volatile>"

# Blocks worth forwarding as [system update] when they change (timestamp-only
# drift sends nothing).
SYSTEM_BLOCK_RES = [
    re.compile(r"(?s)<memory>.*?</memory>"),
    re.compile(r"(?s)<context>.*?</context>"),
    re.compile(r"(?s)<memories>.*?</memories>"),
]

MAX_SYSTEM_STORE = 8192


def _ws(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def _short_ref(s, limit=256):
    s = str(s)
    if len(s) > limit:
        return "sha256:" + hashlib.sha256(s.encode()).hexdigest()
    return s


def norm_content(content):
    """Canonical content: text normalized; attachment bytes replaced by refs.

    data: URLs longer than 256 chars become sha256 markers (stable without
    dragging megabytes through every hash). http(s)/file/bare refs kept raw.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        if content.startswith("data:") and len(content) > 256:
            return _short_ref(content)
        return _ws(content)
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(_ws(p))
            elif isinstance(p, dict):
                t = p.get("type", "")
                if t in ("text",):
                    v = p.get("text", "")
                    parts.append({"type": "text", "text": _ws(v) if isinstance(v, str) else str(v)})
                elif t in ("image_url", "input_image"):
                    ref = p.get("image_url", p.get("url", ""))
                    url = ref.get("url", "") if isinstance(ref, dict) else (ref if isinstance(ref, str) else "")
                    parts.append({"type": t, "url": _short_ref(url)})
                else:
                    # generic: keep type + short scalar fields, hash long blobs
                    keep = {"type": str(t)}
                    for k in ("text", "url", "image_url", "data", "file_data", "source"):
                        v = p.get(k)
                        if isinstance(v, str) and v:
                            keep[k] = _short_ref(_ws(v) if k == "text" else v)
                        elif isinstance(v, dict):
                            keep[k] = _short_ref(json.dumps(v, sort_keys=True))
                    parts.append(keep)
            else:
                parts.append(str(p))
        return parts
    return _ws(str(content))


def _norm_args(args):
    if isinstance(args, dict):
        try:
            return json.dumps(args, sort_keys=True, separators=(",", ":"))
        except Exception:
            return str(args)
    if isinstance(args, str):
        s = args.strip()
        if s.startswith("{"):
            try:
                return json.dumps(json.loads(s), sort_keys=True, separators=(",", ":"))
            except Exception:
                return s
        return s
    return str(args)


def canon(m):
    """Stable canonical form of one OpenAI message (order-independent)."""
    if isinstance(m, str):
        m = {"role": "user", "content": m}
    if not isinstance(m, dict):
        m = {"role": "user", "content": str(m)}
    role = m.get("role") or "user"
    content = m.get("content")
    # o1-and-newer clients may send role "developer" instead of "system"
    # (§B.2) — normalize it the same way so volatile content hashes stable.
    if role in ("system", "developer") and isinstance(content, str):
        content = norm_system(content)
    else:
        content = norm_content(content)
    tcs = []
    for c in m.get("tool_calls") or []:
        if not isinstance(c, dict):
            continue
        fn = c.get("function") or {}
        tcs.append([str(fn.get("name", "")), _norm_args(fn.get("arguments", ""))])
    tcs.sort()
    return json.dumps({
        "role": role,
        "content": content,
        "name": m.get("name"),
        "tool_call_id": m.get("tool_call_id"),
        "tool_calls": tcs,
    }, sort_keys=True, separators=(",", ":"))


def norm_system(text):
    for rx in SYSTEM_VOLATILE:
        text = rx.sub(VOLATILE_TOKEN, text)
    return _ws(text)


def chain_hashes(messages):
    """chain[i] identifies a conversation whose first i+1 messages are exactly these."""
    out = []
    prev = None
    for m in messages or []:
        h = hashlib.sha256(((prev + "|" if prev else CHAIN_SEED) + canon(m)).encode()).hexdigest()
        out.append(h)
        prev = h
    return out


def projected_hash(chain, reply_msg):
    """Chain hash extended with our own reply (pre-register for next turn)."""
    base = chain[-1] if chain else None
    h = hashlib.sha256((((base + "|") if base else CHAIN_SEED) + canon(reply_msg)).encode()).hexdigest()
    return h


def system_blocks(raw):
    """Extract volatile content blocks (memory/context) from raw system text."""
    found = []
    for rx in SYSTEM_BLOCK_RES:
        found.extend(rx.findall(raw or ""))
    return found


def system_update_block(old_raw, new_raw):
    """Compact block to send when normalized system matches but raw drifted.

    Returns "" when only noise (timestamps) changed.
    """
    old_blocks = system_blocks(old_raw)
    new_blocks = system_blocks(new_raw)
    if new_blocks == old_blocks:
        return ""
    if not new_blocks:
        return ""
    return "[system update]\n" + "\n".join(new_blocks) + "\n[/system update]"


def default_store_path():
    base = os.environ.get("XDG_STATE_HOME") or os.path.join(os.path.expanduser("~"), ".local", "state")
    return os.path.join(base, "opencode-shim", "sessions.json")


class SessionStore:
    def __init__(self, path=None, max_sessions=200, ttl=7 * 24 * 3600):
        self.path = path or default_store_path()
        self.max_sessions = max_sessions
        self.ttl = ttl
        self.lock = threading.Lock()
        self.index = {}     # chain_hash -> {"session_id": str, "idx": int}
        self.sessions = {}  # session_id -> record dict
        self.idempotent = {}  # chain_hash -> {"resp": dict, "ts": float}
        self.hits = 0
        self.misses = 0
        self.load()

    def load(self):
        try:
            with open(self.path) as f:
                data = json.load(f)
            if data.get("version") != VERSION:
                return
            self.index = data.get("index", {})
            self.sessions = data.get("sessions", {})
            self._sweep_locked()
        except FileNotFoundError:
            pass
        except Exception:
            self.index, self.sessions = {}, {}

    def save_locked(self):
        tmp = self.path + ".tmp"
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(tmp, "w") as f:
                json.dump({"version": VERSION, "index": self.index,
                           "sessions": self.sessions}, f)
            os.replace(tmp, self.path)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

    def _sweep_locked(self):
        now = time.time()
        dead = [sid for sid, r in self.sessions.items()
                if now - r.get("last_used", 0) > self.ttl]
        for sid in dead:
            del self.sessions[sid]
        if dead:
            gone = set(dead)
            self.index = {h: e for h, e in self.index.items()
                          if e.get("session_id") not in gone}
            # Prune locks for TTL-expired sessions
            try:
                with _session_locks_guard:
                    for sid in list(dead):
                        _session_locks.pop(sid, None)
            except Exception:
                pass
        # LRU cap
        if len(self.sessions) > self.max_sessions:
            ordered = sorted(self.sessions.items(), key=lambda kv: kv[1].get("last_used", 0))
            for sid, _ in ordered[:len(self.sessions) - self.max_sessions]:
                del self.sessions[sid]
            alive = set(self.sessions)
            self.index = {h: e for h, e in self.index.items()
                          if e.get("session_id") in alive}
        # Prune orphaned per-session locks for evicted sessions (fix #3)
        try:
            with _session_locks_guard:
                for sid in list(_session_locks.keys()):
                    if sid not in self.sessions:
                        try:
                            del _session_locks[sid]
                        except KeyError:
                            pass
        except Exception:
            pass

    def resolve(self, messages):
        """Longest-prefix match. Returns dict with session_id|None, delta, meta.

        Must hold no locks across I/O — caller holds store.lock only for this call.
        """
        n = len(messages)
        chain = chain_hashes(messages)
        with self.lock:
            hit_idx, hit = -1, None
            for i in range(n - 1, -1, -1):
                e = self.index.get(chain[i])
                if e and e.get("session_id") in self.sessions:
                    hit_idx, hit = i, e
                    break
            if not hit:
                self.misses += 1
                return {"session_id": None, "delta": list(messages), "chain": chain,
                        "hit": False, "depth": n - 1, "hit_idx": -1, "forked": False,
                        "fork_reason": "miss", "sys_drift": False, "system_update": ""}
            self.hits += 1
            rec = self.sessions[hit["session_id"]]
            # Divergence: Hermes history shorter than what the session already holds
            # (regeneration / edit / compaction) -> caller forks a fresh session.
            if hit_idx < rec.get("idx", -1):
                self.misses += 1
                return {"session_id": None, "delta": list(messages), "chain": chain,
                        "hit": False, "depth": n - 1, "hit_idx": hit_idx,
                        "forked": True,
                        "fork_reason": f"divergence depth={hit_idx} session_idx={rec.get('idx')}",
                        "sys_drift": False, "system_update": ""}
            # System drift check (messages[0] is system by convention;
            # o1-and-newer clients may send role "developer" instead — §B.2)
            sys_drift, sys_update = False, ""
            if messages and isinstance(messages[0], dict) and messages[0].get("role") in ("system", "developer"):
                raw = messages[0].get("content")
                raw = raw if isinstance(raw, str) else json.dumps(raw or "")
                live = hashlib.sha256(raw.encode()).hexdigest()
                if live != rec.get("system_live_hash"):
                    sys_drift = True
                    sys_update = system_update_block(rec.get("system_raw", ""), raw)
                    rec["system_live_hash"] = live
                    rec["system_raw"] = raw[:MAX_SYSTEM_STORE]
            delta = messages[hit_idx + 1:]
            rec["last_used"] = time.time()
            self.save_locked()
            return {"session_id": hit["session_id"], "delta": delta, "chain": chain,
                    "hit": True, "depth": n - 1, "hit_idx": hit_idx, "forked": False,
                    "fork_reason": "", "sys_drift": sys_drift, "system_update": sys_update}

    def register_new(self, chain, session_id, system_raw, tools_hash="", profile="-",
                     tool_names=None):
        """Index a freshly created session (full dump sent once)."""
        n = len(chain)
        with self.lock:
            raw = system_raw if isinstance(system_raw, str) else json.dumps(system_raw or "")
            live = ""
            if chain:
                live = hashlib.sha256(raw.encode()).hexdigest()
            self.sessions[session_id] = {
                "created": time.time(), "last_used": time.time(), "idx": n - 1,
                "system_norm_hash": "", "system_live_hash": live,
                "system_raw": (system_raw or "")[:MAX_SYSTEM_STORE],
                "tools_hash": tools_hash, "profile": profile,
                "tool_names": list(tool_names or []),
                "tool_calls": {}, "turns": 1, "forked_from": None,
            }
            for i, h in enumerate(chain):
                self.index[h] = {"session_id": session_id, "idx": i}
            self._sweep_locked()
            self.save_locked()

    def register_reply(self, chain, reply_msg, session_id, bump=True):
        """Pre-register chain+reply so the next request (which echoes our reply) hits."""
        h = projected_hash(chain, reply_msg)
        with self.lock:
            rec = self.sessions.get(session_id)
            if not rec:
                return h
            idx = len(chain)  # reply is message index n (chain had n entries 0..n-1)
            self.index[h] = {"session_id": session_id, "idx": idx}
            rec["idx"] = idx
            if bump:
                rec["turns"] = rec.get("turns", 0) + 1
            rec["last_used"] = time.time()
            self.save_locked()
            return h

    def session_info(self, session_id):
        with self.lock:
            rec = self.sessions.get(session_id)
            return dict(rec) if rec else None

    def update_tools(self, session_id, tools_hash, tool_names, profile):
        with self.lock:
            rec = self.sessions.get(session_id)
            if rec:
                rec["tools_hash"] = tools_hash
                rec["tool_names"] = list(tool_names or [])
                rec["profile"] = profile
                rec["last_used"] = time.time()
                self.save_locked()

    def note_tool_calls(self, session_id, mapping):
        with self.lock:
            rec = self.sessions.get(session_id)
            if rec:
                rec.setdefault("tool_calls", {}).update(mapping)
                self.save_locked()

    def tool_name(self, session_id, call_id):
        with self.lock:
            rec = self.sessions.get(session_id)
            if rec:
                return (rec.get("tool_calls") or {}).get(call_id)
        return None

    def check_idempotent(self, chain_hash, ttl):
        with self.lock:
            e = self.idempotent.get(chain_hash)
            if e and time.time() - e["ts"] < ttl:
                return e["resp"]
            return None

    def store_idempotent(self, chain_hash, resp):
        with self.lock:
            self.idempotent[chain_hash] = {"resp": resp, "ts": time.time()}
            if len(self.idempotent) > 500:
                old = sorted(self.idempotent.items(), key=lambda kv: kv[1]["ts"])
                for k, _ in old[:len(old) - 500]:
                    del self.idempotent[k]

    def debug_sessions(self):
        with self.lock:
            total = self.hits + self.misses
            return {
                "index_size": len(self.index),
                "sessions": len(self.sessions),
                "hit_rate": round(self.hits / total, 4) if total else 0.0,
                "hits": self.hits, "misses": self.misses,
                "oldest_last_used": min([r.get("last_used", 0) for r in self.sessions.values()],
                                        default=0),
            }


# --- per-session in-flight locks (one turn per conversation) ---
_session_locks = {}
_session_locks_guard = threading.Lock()


def session_lock(session_id):
    with _session_locks_guard:
        lk = _session_locks.get(session_id)
        if lk is None:
            lk = threading.Lock()
            _session_locks[session_id] = lk
        return lk

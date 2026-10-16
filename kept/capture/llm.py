"""The AI layer: NVIDIA Nemotron models served on Nebius Token Factory.

Kept picks the right model for each job:
  - fast:   Nemotron 3 Nano 30B-A3B   short texts, voice notes, the month plan
  - vision: Nemotron 3.5 Super VL     photos, letters, frames of an app-history video
  - plan:   Nemotron 3 Super 120B     planning the rest of the month around her routine
  - deep:   Nemotron 3 Ultra 550B     explaining a state letter in plain English and Spanish

The app never depends on it. Every call has a time limit, must return strict JSON, and its
answer is checked by Kept's rules afterwards. If the key is missing, the spending cap is
reached, or a call fails, the caller falls back to rules or asks the person.

Without a Nebius key, a local Ollama model (free, on your own computer) can handle the text
jobs instead; photos then need to be typed.
"""

from __future__ import annotations

import base64
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

NEBIUS_URL = os.environ.get("KEPT_NEBIUS_URL", "https://api.tokenfactory.nebius.com/v1").rstrip("/")
MODELS = {
    "fast": os.environ.get("KEPT_MODEL_FAST", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"),
    "vision": os.environ.get("KEPT_MODEL_VISION", "nvidia/Nemotron-3.5-Super-VL"),
    "deep": os.environ.get("KEPT_MODEL_DEEP", "nvidia/Nemotron-3-Ultra-550b-a55b"),
    "plan": os.environ.get("KEPT_MODEL_PLAN", "nvidia/nemotron-3-super-120b-a12b"),
}
LABELS = {
    "fast": "NVIDIA Nemotron 3 Nano",
    "vision": "NVIDIA Nemotron 3.5 Super VL",
    "deep": "NVIDIA Nemotron 3 Ultra",
    "plan": "NVIDIA Nemotron 3 Super",
}
# US dollars per million tokens (input, output), from the Token Factory price list
PRICES = {
    MODELS["fast"]: (0.06, 0.24),
    MODELS["vision"]: (0.30, 0.90),
    MODELS["deep"]: (1.00, 3.00),
    MODELS["plan"]: (0.30, 0.90),
}
TIMEOUT = float(os.environ.get("KEPT_AI_TIMEOUT", "45"))
# Hard spending limits, far below the free credits: when one is reached, Kept stops calling
# the AI and falls back to its rules until the next day.
DAILY_USD = float(os.environ.get("KEPT_AI_DAILY_USD", "1.00"))
TOTAL_USD = float(os.environ.get("KEPT_AI_TOTAL_USD", "35"))

OLLAMA_URL = os.environ.get("KEPT_OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("KEPT_MODEL", "qwen2.5:3b")
MODEL = MODELS["fast"]  # kept for older callers

_enabled = {"on": os.environ.get("KEPT_USE_AI", "1") != "0"}
_lock = threading.Lock()
_ollama_cache = {"t": 0.0, "ok": False, "models": []}
_last: dict = {}
_guard = {"fn": None}   # per-visitor limit, set by the web app for each request
_local = threading.local()  # per-request switch: a visitor who turns the AI off affects only themselves


def _key() -> str:
    return os.environ.get("NEBIUS_API_KEY", "").strip()


def set_enabled(on: bool):
    _enabled["on"] = bool(on)


def is_enabled() -> bool:
    off = getattr(_local, "off", None)
    return _enabled["on"] and not off


def set_request(off: bool = False, guard=None):
    """Called by the web app at the start of each request: this visitor's AI switch, and
    guard() -> bool, which says whether they may make one more AI call (and counts it)."""
    _local.off = bool(off)
    _local.guard = guard


class paused:
    """with llm.paused(): ... — no AI calls inside (used to replay the demo's past days quickly)."""

    def __enter__(self):
        self.prev = getattr(_local, "off", None)
        _local.off = True

    def __exit__(self, *exc):
        _local.off = self.prev


def set_request_guard(fn):
    _guard["fn"] = fn


def provider() -> str:
    if _key():
        return "nvidia"
    if _ollama_status()["ok"]:
        return "ollama"
    return "none"


def label(task: str = "fast") -> str:
    if provider() == "ollama":
        return f"local AI ({OLLAMA_MODEL})"
    return LABELS.get(task, "NVIDIA Nemotron")


def last_call() -> dict:
    return dict(_last)


# ------------------------------------------------------------------ spending

def _usage_path() -> Path:
    from .. import db
    return Path(os.environ.get("KEPT_AI_USAGE_FILE", db.DATA_DIR / "ai_usage.json"))


def _load_usage() -> dict:
    try:
        return json.loads(_usage_path().read_text())
    except (OSError, ValueError):
        return {}


def usage() -> dict:
    u = _load_usage()
    today = date.today().isoformat()
    day = u.get("days", {}).get(today, {})
    return {"today_usd": day.get("usd", 0.0), "today_calls": day.get("calls", 0),
            "total_usd": u.get("total_usd", 0.0), "total_calls": u.get("total_calls", 0),
            "by_model": u.get("by_model", {}), "daily_cap": DAILY_USD, "total_cap": TOTAL_USD}


def _record(model: str, tin: int, tout: int, seconds: float):
    pin, pout = PRICES.get(model, (1.0, 3.0))
    usd = tin / 1e6 * pin + tout / 1e6 * pout
    with _lock:
        u = _load_usage()
        today = date.today().isoformat()
        days = u.setdefault("days", {})
        d = days.setdefault(today, {"usd": 0.0, "calls": 0})
        d["usd"] += usd
        d["calls"] += 1
        u["total_usd"] = u.get("total_usd", 0.0) + usd
        u["total_calls"] = u.get("total_calls", 0) + 1
        m = u.setdefault("by_model", {}).setdefault(model, {"calls": 0, "seconds": 0.0})
        m["calls"] += 1
        m["seconds"] += seconds
        for k in sorted(days)[:-30]:   # keep a month of daily history
            days.pop(k, None)
        try:
            p = _usage_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(u))
        except OSError:
            pass


def within_budget() -> bool:
    u = usage()
    return u["today_usd"] < DAILY_USD and u["total_usd"] < TOTAL_USD


# ------------------------------------------------------------------ availability

def _ollama_status(force: bool = False) -> dict:
    if not force and time.time() - _ollama_cache["t"] < 30:
        return dict(_ollama_cache)
    ok, models = False, []
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=1.0) as r:
            models = [m.get("name", "") for m in json.loads(r.read().decode()).get("models", [])]
            ok = any(m == OLLAMA_MODEL or m.startswith(OLLAMA_MODEL.split(":")[0]) for m in models)
    except (urllib.error.URLError, OSError, ValueError):
        ok = False
    _ollama_cache.update(t=time.time(), ok=ok, models=models)
    return dict(_ollama_cache)


def status(force: bool = False) -> dict:
    p = provider() if not force else ("nvidia" if _key() else ("ollama" if _ollama_status(True)["ok"] else "none"))
    return {"ok": p != "none", "provider": p, "models": MODELS, "labels": LABELS, "usage": usage(),
            "ollama_model": OLLAMA_MODEL, "enabled": is_enabled()}


def available(task: str = "fast") -> bool:
    if not is_enabled():
        return False
    p = provider()
    if p == "nvidia":
        return within_budget()
    if p == "ollama":
        return task != "vision"
    return False


def vision_available() -> bool:
    return available("vision")


# ------------------------------------------------------------------ calls

def _parse_json(content: str):
    if not content:
        return None
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", content, re.S)
    if m:
        content = m.group(1).strip()
    a = content.find("{")
    if a < 0:
        return None
    body = content[a:]
    # models sometimes stop one bracket short of the end: try closing it
    for tail in ("", "}", "]}", "}]}", '"}]}', '"}'):
        try:
            out = json.loads(body + tail)
            return out if isinstance(out, dict) else None
        except ValueError:
            continue
    b = body.rfind("}")
    try:
        out = json.loads(body[:b + 1]) if b > 0 else None
    except ValueError:
        return None
    return out if isinstance(out, dict) else None


def _nebius(task: str, messages: list, max_tokens: int, timeout: float | None):
    g = getattr(_local, "guard", None) or _guard["fn"]
    if g is not None and not g():
        return None
    model = MODELS[task]
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        # answer directly: reasoning traces cost time and credits, and the rules check the result
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if task != "vision":
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(f"{NEBIUS_URL}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout or TIMEOUT) as r:
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        _last.update(task=task, model=model, ok=False, error=f"HTTP {e.code}", seconds=round(time.time() - t0, 1))
        return None
    except (urllib.error.URLError, OSError, ValueError, TimeoutError) as e:
        _last.update(task=task, model=model, ok=False, error=type(e).__name__, seconds=round(time.time() - t0, 1))
        return None
    secs = time.time() - t0
    u = data.get("usage") or {}
    _record(model, int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0), secs)
    try:
        content = data["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        content = ""
    out = _parse_json(content)
    _last.update(task=task, model=model, ok=out is not None, seconds=round(secs, 1),
                 tokens=int(u.get("total_tokens") or 0))
    return out


def _ollama(system: str, user: str, timeout: float | None):
    body = json.dumps({"model": OLLAMA_MODEL, "format": "json", "stream": False, "options": {"temperature": 0},
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout or TIMEOUT) as r:
            data = json.loads(r.read().decode())
        return _parse_json(data.get("message", {}).get("content", ""))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None


def chat_json(system: str, user: str, timeout: float | None = None, task: str = "fast",
              max_tokens: int = 700) -> dict | None:
    """Ask a text model for a JSON object. Returns None on any problem."""
    if not available(task):
        return None
    if provider() == "ollama":
        return _ollama(system, user, timeout)
    return _nebius(task, [{"role": "system", "content": system}, {"role": "user", "content": user}],
                   max_tokens, timeout)


def chat_tools(messages: list, tools: list, task: str = "plan", max_tokens: int = 900,
               timeout: float | None = None) -> dict | None:
    """One turn of a tool-using conversation. Returns the assistant message (with tool_calls,
    or content), or None on any problem."""
    if not available(task) or provider() != "nvidia":
        return None
    g = getattr(_local, "guard", None) or _guard["fn"]
    if g is not None and not g():
        return None
    model = MODELS[task]
    body = {"model": model, "messages": messages, "tools": tools, "tool_choice": "auto", "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{NEBIUS_URL}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout or TIMEOUT) as r:
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        _last.update(task=task, model=model, ok=False, error=f"HTTP {e.code}", seconds=round(time.time() - t0, 1))
        return None
    except (urllib.error.URLError, OSError, ValueError, TimeoutError) as e:
        _last.update(task=task, model=model, ok=False, error=type(e).__name__, seconds=round(time.time() - t0, 1))
        return None
    secs = time.time() - t0
    u = data.get("usage") or {}
    _record(model, int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0), secs)
    _last.update(task=task, model=model, ok=True, seconds=round(secs, 1), tokens=int(u.get("total_tokens") or 0))
    try:
        return data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None


def _image_part(path: Path) -> dict:
    p = Path(path)
    mime = {".png": "image/png", ".webp": "image/webp"}.get(p.suffix.lower(), "image/jpeg")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"}}


def vision_json(prompt: str, images: list, timeout: float | None = None, max_tokens: int = 1500) -> dict | None:
    """Show one or more images to Nemotron Super VL and get a JSON object back."""
    if not images or not vision_available() or provider() != "nvidia":
        return None
    content = [{"type": "text", "text": prompt}] + [_image_part(p) for p in images]
    return _nebius("vision", [{"role": "user", "content": content}], max_tokens, timeout)

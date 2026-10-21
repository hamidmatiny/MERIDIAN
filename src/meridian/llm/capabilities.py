"""Probe whether a provider can call tools or emit a schema, and cache the answer."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, ValidationError

from meridian.config import Settings
from meridian.llm.router import resolve_chain

_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"severity": {"type": "string", "enum": ["SEV1", "SEV2", "SEV3", "SEV4"]}},
    "required": ["severity"],
}
_TOOL: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "ping",
        "description": "Reply with a ping.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


class Capability(BaseModel):
    """What one provider and model accepted. A 501 is a miss, not a guess."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    model: str
    ollama_version: str = ""
    tool_calling: bool = False
    structured_json_schema: bool = False
    structured_json_mode: bool = False
    reachable: bool = True
    auth_failed: bool = False
    probed_at: str = ""


def method_for(capability: Capability) -> str | None:
    """Prefer a submit tool, then JSON schema, then JSON mode."""
    if capability.auth_failed or not capability.reachable:
        return None
    if capability.tool_calling:
        return "submit_tool"
    if capability.structured_json_schema:
        return "json_schema"
    if capability.structured_json_mode:
        return "json_mode"
    return None


def cache_key(provider: str, model: str, ollama_version: str) -> str:
    """Identity of one probe row."""
    return f"{provider}|{model}|{ollama_version}"


def supported(status: int, body: str) -> bool:
    """A capability is present only when the server accepts the request."""
    text = body.lower()
    if status == 501:
        return False
    if "unavailable" in text and ("structured" in text or "capability" in text):
        return False
    return status == 200


def arm_node(settings: Settings, node: str, chat: Any) -> None:
    """Probe the chain and record the method on one live chat."""
    _rows, methods = plan_nodes(settings)
    chosen = methods.get(node, [])
    if not any(chosen):
        raise RuntimeError(f"no structured method for {node}")
    apply_methods(chat, chosen)


def format_table(rows: list[Capability], methods: dict[str, list[str | None]]) -> str:
    """The table printed at the start of a live run."""
    lines = ["capabilities"]
    seen: set[str] = set()
    for row in rows:
        key = cache_key(row.provider, row.model, row.ollama_version)
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"  {row.provider} {row.model} version={row.ollama_version or '-'} "
            f"tool_calling={row.tool_calling} "
            f"json_schema={row.structured_json_schema} "
            f"json_mode={row.structured_json_mode} "
            f"method={_method_label(row)}"
        )
    for node, chosen in methods.items():
        rendered = ",".join(item or "none" for item in chosen) or "none"
        lines.append(f"node {node} {rendered}")
    return "\n".join(lines)


def apply_methods(chat: Any, methods: list[str | None]) -> None:
    """Remember which output method each hop in this chain can use."""
    chat.hop_methods = list(methods)
    chat.output_method = next((item for item in methods if item), None)


def plan_nodes(
    settings: Settings,
    *,
    cache_path: Path | None = None,
    probe: Any | None = None,
    ollama_version: str | None = None,
) -> tuple[list[Capability], dict[str, list[str | None]]]:
    """Probe every hop. ``probe`` replaces the network call in tests."""
    path = cache_path if cache_path is not None else Path(settings.capabilities_path)
    if ollama_version is not None:
        version = ollama_version
    elif probe is None:
        version = _ollama_version(settings.ollama_base_url)
    else:
        version = ""
    rows: list[Capability] = []
    methods: dict[str, list[str | None]] = {}
    caller = _http_probe if probe is None else probe
    for node in ("triage", "rootcause", "fix", "critic"):
        provider = str(getattr(settings, f"{node}_provider"))
        model = getattr(settings, f"{node}_model")
        fallback = str(getattr(settings, f"{node}_fallback"))
        if not isinstance(model, str) or model.strip() == "":
            methods[node] = []
            continue
        hops = resolve_chain(
            provider,
            model,
            fallback,
            allow_paid=settings.allow_paid,
            omniroute_model=_omniroute_model(settings, node),
        )
        chosen: list[str | None] = []
        for hop_provider, hop_model in hops:
            row = _cached(
                caller,
                path,
                hop_provider,
                hop_model,
                version if hop_provider == "ollama" else "",
                settings,
            )
            rows.append(row)
            chosen.append(method_for(row))
        methods[node] = chosen
    return rows, methods


def _cached(
    probe: Any,
    path: Path,
    provider: str,
    model: str,
    version: str,
    settings: Settings,
) -> Capability:
    key = cache_key(provider, model, version)
    stored = _read(path)
    raw = stored.get(key)
    if isinstance(raw, dict):
        try:
            cached = Capability.model_validate(raw)
        except ValidationError:
            cached = None
        if cached is not None and cached.reachable and _fresh(cached):
            return cached
    row = probe(provider, model, version, settings)
    if not isinstance(row, Capability):
        row = Capability.model_validate(row)
    if not row.reachable or row.auth_failed:
        return row
    stamped = row.model_copy(update={"probed_at": datetime.now(UTC).isoformat()})
    stored[key] = stamped.model_dump()
    _write(path, stored)
    return stamped


def _read(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(loaded, dict):
        return {}
    return loaded


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _omniroute_model(settings: Settings, node: str) -> str | None:
    value = getattr(settings, f"{node}_omniroute_model", None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _fresh(row: Capability) -> bool:
    """A cached probe lasts 24 hours. A blank timestamp is not a probe answer."""
    if not row.probed_at:
        return False
    try:
        probed = datetime.fromisoformat(row.probed_at)
    except ValueError:
        return False
    if probed.tzinfo is None:
        probed = probed.replace(tzinfo=UTC)
    return datetime.now(UTC) - probed < timedelta(hours=24)


def _method_label(row: Capability) -> str:
    if not row.reachable:
        return "UNREACHABLE"
    if row.auth_failed:
        return "AUTH_FAILED"
    found = method_for(row)
    return "none" if found is None else found


def _answered(results: list[tuple[int, str]]) -> bool:
    """A connection error or a timeout is status 0 and is not a probe answer."""
    return any(status != 0 for status, _body in results)


def _ollama_version(base_url: str) -> str:
    status, body = _request("GET", base_url.rstrip("/") + "/api/version", None)
    if status != 200:
        return ""
    try:
        loaded = json.loads(body)
    except json.JSONDecodeError:
        return ""
    version = loaded.get("version") if isinstance(loaded, dict) else None
    return version if isinstance(version, str) else ""


def _http_probe(provider: str, model: str, version: str, settings: Settings) -> Capability:
    if provider == "ollama":
        return _probe_ollama(settings.ollama_base_url, model, version)
    if provider == "omniroute":
        return _probe_openai(settings.omniroute_base_url, "omniroute", model, version, settings)
    if provider in {"anthropic", "xai"}:
        return Capability(
            provider=provider,
            model=model,
            ollama_version=version,
            tool_calling=True,
            structured_json_schema=True,
            structured_json_mode=False,
        )
    return Capability(provider=provider, model=model, ollama_version=version)


def _probe_ollama(base_url: str, model: str, version: str) -> Capability:
    root = base_url.rstrip("/")
    tool = _request("POST", root + "/api/chat", _ollama_tool(model))
    schema = _request("POST", root + "/api/chat", _ollama_format(model, _SCHEMA))
    mode = _request("POST", root + "/api/chat", _ollama_format(model, "json"))
    answers = [tool, schema, mode]
    return Capability(
        provider="ollama",
        model=model,
        ollama_version=version,
        tool_calling=supported(*tool),
        structured_json_schema=supported(*schema),
        structured_json_mode=supported(*mode),
        reachable=_answered(answers),
    )


def _probe_openai(
    base_url: str,
    provider: str,
    model: str,
    version: str,
    settings: Settings | None = None,
) -> Capability:
    root = base_url.rstrip("/")
    url = root + "/chat/completions"
    headers = {} if settings is None else _bearer(settings)
    tool = _request("POST", url, _openai_tool(model), headers)
    schema = _request("POST", url, _openai_schema(model), headers)
    mode = _request("POST", url, _openai_json(model), headers)
    answers = [tool, schema, mode]
    denied = _auth_failed(answers)
    return Capability(
        provider=provider,
        model=model,
        ollama_version=version,
        tool_calling=False if denied else supported(*tool),
        structured_json_schema=False if denied else supported(*schema),
        structured_json_mode=False if denied else supported(*mode),
        reachable=_answered(answers),
        auth_failed=denied,
    )


def _bearer(settings: Settings) -> dict[str, str]:
    secret = settings.omniroute_api_key
    if secret is None:
        return {}
    value = secret.get_secret_value().strip()
    if not value:
        return {}
    return {"Authorization": "Bearer " + value}


def _auth_failed(results: list[tuple[int, str]]) -> bool:
    codes = [status for status, _body in results if status != 0]
    return bool(codes) and all(status == 401 for status in codes)


def _ollama_tool(model: str) -> dict[str, object]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Call the ping tool."}],
        "tools": [_TOOL],
        "stream": False,
        "options": {"num_predict": 8},
    }


def _ollama_format(model: str, fmt: object) -> dict[str, object]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "severity is SEV2"}],
        "format": fmt,
        "stream": False,
        "options": {"num_predict": 8},
    }


def _openai_tool(model: str) -> dict[str, object]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Call the ping tool."}],
        "tools": [_TOOL],
        "max_tokens": 8,
    }


def _openai_schema(model: str) -> dict[str, object]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "severity is SEV2"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "severity", "schema": _SCHEMA},
        },
        "max_tokens": 8,
    }


def _openai_json(model: str) -> dict[str, object]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with JSON."}],
        "response_format": {"type": "json_object"},
        "max_tokens": 8,
    }


def _request(
    method: str,
    url: str,
    payload: dict[str, object] | None,
    headers: dict[str, str] | None = None,
) -> tuple[int, str]:
    data = None if payload is None else json.dumps(payload).encode()
    sent = {"Content-Type": "application/json"}
    if headers:
        sent.update(headers)
    request = Request(url, data=data, headers=sent, method=method)
    try:
        with urlopen(request, timeout=30) as response:
            return response.status, response.read().decode()
    except HTTPError as exc:
        return exc.code, exc.read().decode()
    except (URLError, TimeoutError, OSError) as exc:
        return 0, str(exc)

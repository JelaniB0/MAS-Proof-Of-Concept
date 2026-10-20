"""Small REST adapter for Groq, Azure Foundry, and Ollama. No implicit retries."""

from dataclasses import dataclass
import json
import os
import re
from string import Template
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .config import Model
from .graph import token_reserve


class ProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    usage_source: str
    finish_reason: str


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def expand(value):
    try:
        return Template(value).substitute(os.environ)
    except KeyError as error:
        raise ProviderError(f"Set {error.args[0]} in .env or the environment before running live requests") from None


def connection(model):
    endpoint, name = expand(model.endpoint), expand(model.model)
    url = urlsplit(endpoint)
    local = url.hostname in {"localhost", "127.0.0.1", "::1"}
    if (url.scheme != "https" and not (url.scheme == "http" and local)) or not url.hostname:
        raise ProviderError(f"{model.id}: endpoint must use HTTPS (HTTP is allowed for localhost)")
    if url.username or url.password or url.fragment:
        raise ProviderError(f"{model.id}: endpoint must not contain credentials or a fragment")
    if not name.strip():
        raise ProviderError(f"{model.id}: model/deployment name is empty")
    headers = {"Content-Type": "application/json", "Accept": "application/json",
               "User-Agent": "AgentBalance-PoC/0.1.0"}
    if model.api_key_env:
        key = os.environ.get(model.api_key_env, "")
        if not key:
            raise ProviderError(f"Set {model.api_key_env} in .env or the environment before running live requests")
        headers[model.auth_header] = f"Bearer {key}" if model.auth_header == "Authorization" else key
    elif model.provider != "ollama":
        raise ProviderError(f"{model.id}: api_key_env is required for a hosted provider")
    return endpoint, name, headers


def preflight(models):
    for model in models:
        if model.provider != "mock":
            connection(model)


def request_body(model, name, prompt, max_output_tokens):
    body = {"model": name, **model.extra_body}
    if model.protocol == "responses":
        body.update(input=prompt, max_output_tokens=max_output_tokens, store=False, stream=False)
    else:
        body.update(messages=prompt, stream=False)
        body[model.token_limit_field] = max_output_tokens
    return body


def http_error_hint(status, body):
    """Classify known errors without exposing response text or credential values."""
    if status == 403 and re.search(rb"error code:\s*1010\b", body, re.IGNORECASE):
        return "Gateway rejected the HTTP client signature (error 1010); check client headers or contact provider support."
    try:
        data = json.loads(body)
        error = data.get("error", {}) if isinstance(data, dict) else {}
        code = error.get("code") if isinstance(error, dict) else None
    except (ValueError, UnicodeError):
        code = None
    if status == 403 and code == "model_permission_blocked_org":
        return "Model blocked by organization permissions; ask the organization admin to enable it in Groq Limits."
    if status == 403 and code == "model_permission_blocked_project":
        return "Model blocked by project permissions; ask the project admin to enable it in Groq project Limits."
    return {
        400: "Provider rejected the request; check the model and supported request parameters.",
        401: "Authentication failed; check the API key and its provider.",
        403: "Access forbidden; check provider permissions and network access restrictions.",
        404: "Endpoint or model not found; check the request URL and model/deployment ID.",
        429: "Provider rate limit or quota reached; wait for the limit to reset or check account limits.",
    }.get(status, "Provider request failed; check service status and configuration.")


def parse_response(data, model, prompt, max_output_tokens):
    if not isinstance(data, dict) or data.get("error"):
        raise ProviderError(f"{model.id}: provider returned an error or invalid response")
    usage = data.get("usage") or {}
    if model.protocol == "responses":
        text = "\n".join(part.get("text", "") for item in data.get("output", [])
                         if item.get("type") == "message" for part in item.get("content", [])
                         if part.get("type") == "output_text")
        input_tokens, output_tokens = usage.get("input_tokens"), usage.get("output_tokens")
        reason = data.get("status", "unknown")
        if reason == "incomplete":
            reason = (data.get("incomplete_details") or {}).get("reason", "incomplete")
    else:
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError(f"{model.id}: provider returned no choices")
        text = (choices[0].get("message") or {}).get("content") or ""
        input_tokens, output_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
        reason = choices[0].get("finish_reason") or "unknown"
    if not isinstance(text, str):
        raise ProviderError(f"{model.id}: expected a text response")
    valid = lambda n: isinstance(n, int) and not isinstance(n, bool) and n >= 0
    reported = valid(input_tokens) and valid(output_tokens)
    # Missing usage must not make a paid request appear free.
    return Completion(text, input_tokens if valid(input_tokens) else token_reserve(prompt),
                      output_tokens if valid(output_tokens) else max_output_tokens,
                      "provider" if reported else "estimated_reservation", str(reason))


def complete(model: Model, prompt, max_output_tokens, timeout):
    if model.provider == "mock":
        time.sleep(min(model.latency_seconds, timeout))
        if model.latency_seconds > timeout:
            raise ProviderError(f"{model.id}: simulated timeout")
        # Deliberately a labeled fixture, never presented as LLM reasoning or accuracy.
        text = (f"[OFFLINE FIXTURE: {model.id}] The graph passed the task to this agent. "
                f"Received {prompt[-1]['content'].count('Contribution from ')} direct contributions. "
                "Connect a real model to generate a substantive answer.")[:max_output_tokens * 3]
        return Completion(text, max(1, token_reserve(prompt) // 4), max(1, (len(text) + 3) // 4),
                          "simulated", "stop")
    endpoint, name, headers = connection(model)
    body = request_body(model, name, prompt, max_output_tokens)
    request = Request(endpoint, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    try:
        # No redirect following: credentials stay on the configured endpoint.
        with build_opener(NoRedirect()).open(request, timeout=timeout) as response:
            raw = response.read(4_000_001)
        if len(raw) > 4_000_000:
            raise ProviderError(f"{model.id}: response exceeds the 4 MB PoC limit")
        return parse_response(json.loads(raw), model, prompt, max_output_tokens)
    except HTTPError as error:
        # Do not echo provider bodies, which can contain keys or the original task.
        status_code = error.code
        try:
            error_body = error.read(8192)
        except (OSError, TimeoutError):
            error_body = b""
        finally:
            error.close()
        raise ProviderError(f"{model.id}: HTTP {status_code}; {http_error_hint(status_code, error_body)} "
                            "Request was not retried.") from None
    except (URLError, TimeoutError, OSError):
        raise ProviderError(f"{model.id}: network error or timeout; request was not retried") from None
    except (ValueError, KeyError, TypeError, AttributeError):
        raise ProviderError(f"{model.id}: malformed provider response; request was not retried") from None

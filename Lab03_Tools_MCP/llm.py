"""One small client for every lab: chat, raw completion and embeddings.

The models run on the GWDG Academic Cloud (Chat AI), which speaks the OpenAI API.
You need your API key from the Onboarding notebook (saved in ~/.academic_cloud_key).

The models:

    MODEL         qwen3-30b-a3b-instruct-2507, answers directly. Used by chat().
    THINK_MODEL   qwen3.6-35b-a3b, a reasoning model: thinks first, then answers.
                  Used by chat(..., think=True); the thinking is in message["reasoning"].
    EMBED_MODEL   qwen3-embedding-4b, turns text into vectors for search (RAG).

Responses are the plain OpenAI JSON, so what you learn here works with any
OpenAI-compatible provider:

    response = chat([{"role": "user", "content": "Hi"}])
    response["choices"][0]["message"]["content"]   # the reply
    response["usage"]["prompt_tokens"]             # tokens the model read
"""

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://chat-ai.academiccloud.de/v1"
MODEL = "qwen3-30b-a3b-instruct-2507"
THINK_MODEL = "qwen3.6-35b-a3b"
EMBED_MODEL = "qwen3-embedding-4b"
CONTEXT_LENGTH = 262_144   # tokens MODEL and THINK_MODEL can read at once
KEY_FILE = Path.home() / ".academic_cloud_key"
MAX_WAIT = 60              # longest rate-limit wait (seconds) before giving up with a message

NO_KEY = ("No Academic Cloud API key saved. Run the Onboarding notebook "
          "(Onboarding/onboarding.ipynb), Steps 1 to 3, then restart this notebook.")


def _post(path, body=None, retries=5):
    """POST body to the server (GET when body is None), retrying on rate limits
    and on short server hiccups."""
    if not KEY_FILE.exists():
        raise RuntimeError(NO_KEY)
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {KEY_FILE.read_text().strip()}"}
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(BASE_URL + path, data=data, headers=headers)
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            # 429 = rate limit (requests per minute): wait, then try again.
            if error.code == 429:
                wait = int(error.headers.get("Retry-After") or 20)
                if wait > MAX_WAIT:
                    # The hourly or daily quota is used up: waiting would freeze the cell.
                    raise RuntimeError(
                        f"Request limit of your API key reached (30 per minute, 200 per hour, "
                        f"1000 per day). Try again in about {wait // 60 + 1} minutes.") from None
                if attempt < retries:
                    print(f"(rate limit reached, waiting {wait} s)")
                    time.sleep(wait)
                    continue
            # 500-504 = a server-side hiccup; the same request usually works a moment later.
            if 500 <= error.code <= 504 and attempt < retries:
                time.sleep(2 ** attempt)   # 1, 2, 4, 8, 16 s
                continue
            detail = error.read().decode(errors="replace")[:300]
            raise RuntimeError(f"Academic Cloud answered HTTP {error.code}: {detail}") from None


def chat(messages, model=None, think=False, **options):
    """Send a conversation, get the model's reply as OpenAI JSON.

    think=True uses THINK_MODEL, which reasons before answering (slower, often better
    on hard tasks). options are passed through, e.g. temperature=0, max_tokens=200, tools=[...].
    """
    model = model or (THINK_MODEL if think else MODEL)
    return _post("/chat/completions", {"model": model, "messages": messages, **options})


def complete(prompt, model=None, **options):
    """Continue plain text: no chat template, the model reads exactly `prompt`."""
    return _post("/completions", {"model": model or MODEL, "prompt": prompt, **options})


def embed(texts, model=None):
    """Turn each text into a vector (a list of numbers)."""
    response = _post("/embeddings", {"model": model or EMBED_MODEL, "input": texts})
    return [item["embedding"] for item in response["data"]]


def models():
    """Names of the models this server offers."""
    return sorted(m["id"] for m in _post("/models")["data"])


print(f"llm: using the Academic Cloud, model {MODEL}")
if not KEY_FILE.exists():
    print(f"llm: WARNING: {NO_KEY}")

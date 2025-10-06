# src/llm.py
import os, time, json, requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")

def warm_up(model: str = "llama3"):
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    except Exception:
        return
    try:
        _ = chat(
            model=model,
            messages=[{"role": "user", "content": "ok"}],
            options={"num_predict": 1, "temperature": 0.0},
            timeout=20,
            max_retries=0,
            stream=True,
        )
    except Exception:
        pass

def chat(model: str,
         messages: list[dict],
         options: dict | None = None,
         timeout: int = 90,
         max_retries: int = 2,
         stream: bool = True) -> str:
    """
    Robust chat wrapper for Ollama.
    - Streams + accumulates chunks.
    - Retries on timeouts/conn errors.
    - **If streaming yields empty content, auto-fallback to a non-stream call with larger decode budget.**
    """
    url = f"{OLLAMA_URL}/api/chat"

    opts = dict(options or {})
    import os
    max_np = int(os.getenv("AGENT_NUM_PREDICT", "384"))  # default a bit smaller
    opts.setdefault("num_predict", max_np)
    opts.setdefault("temperature", 0.3)
    opts.setdefault("top_p", 0.9)
    opts.setdefault("repeat_penalty", 1.05)

    payload = {"model": model, "messages": messages, "stream": bool(stream), "options": opts}
    last_err = None

    for attempt in range(max_retries + 1):
        try:
            if stream:
                content = ""
                with requests.post(url, json=payload, timeout=timeout, stream=True) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if not line:
                            continue
                        try:
                            j = json.loads(line.decode("utf-8"))
                        except json.JSONDecodeError:
                            continue
                        chunk = (j.get("message") or {}).get("content") or ""
                        if chunk:
                            content += chunk
                # empty-stream fallback (no exception; just no tokens produced)
                if content.strip():
                    return content.strip()
                # one-shot non-stream fallback with a bit more budget
                fallback_payload = dict(payload)
                fallback_payload["stream"] = False
                fallback_opts = dict(opts)
                fallback_opts["num_predict"] = max(int(opts.get("num_predict", 512)) + 256, 640)
                fallback_payload["options"] = fallback_opts
                r2 = requests.post(url, json=fallback_payload, timeout=timeout)
                r2.raise_for_status()
                data = r2.json()
                return ((data.get("message") or {}).get("content") or "").strip()

            else:
                r = requests.post(url, json=payload, timeout=timeout)
                r.raise_for_status()
                data = r.json()
                return ((data.get("message") or {}).get("content") or "").strip()

        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            time.sleep(0.6 * (attempt + 1))
            continue
        except Exception as e:
            last_err = e
            break

    # final non-stream fallback if retries were exhausted
    try:
        payload["stream"] = False
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return ((data.get("message") or {}).get("content") or "").strip()
    except Exception as e:
        raise last_err or e
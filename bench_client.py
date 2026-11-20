#!/usr/bin/env python3
"""Benchmark one running vLLM server.

For each target prompt length it measures:
  ttft         time to first token on a cold prompt (prefill-dominated)
  ttft_cached  the same prompt sent again (prefix-cache hit, like an agent loop resending context)
  decode_tps   decode tokens/s at that context depth
  needle_ok    whether a passphrase buried mid-prompt was retrieved (cheap long-context sanity check)
Also measures decode speed on a short prompt.

  bench_client.py --port 8000 --key-file ~/.config/qwen-local/api_key --out r.json --label g8_fp8_pp1
  bench_client.py --summarize results/
"""
import argparse, glob, json, os, random, sys, time
import requests

WORDS = ("alpha river stone window garden signal market paper engine forest copper lantern harbor meadow "
         "thunder candle bridge anchor violet ribbon canyon marble orchard pepper saddle timber velvet "
         "whistle yellow zephyr basket cobalt dragon ember falcon glacier hammer island jungle kettle "
         "ladder mirror needle oyster pillow quartz rocket sunset tunnel umbrella voyage willow").split()


def chat_stream(base, headers, model, messages, max_tokens, timeout=3600):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter()
    t_first, text, usage = None, [], None
    with requests.post(f"{base}/v1/chat/completions", json=body, headers=headers,
                       stream=True, timeout=timeout) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            payload = line[6:]
            if payload == b"[DONE]":
                break
            obj = json.loads(payload)
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                d = ch.get("delta", {})
                piece = d.get("content") or d.get("reasoning_content") or d.get("reasoning")
                if piece:
                    if t_first is None:
                        t_first = time.perf_counter()
                    text.append(d.get("content") or "")
    t1 = time.perf_counter()
    usage = usage or {}
    ct = usage.get("completion_tokens", 0)
    ttft = (t_first - t0) if t_first else None
    dec = (ct - 1) / (t1 - t_first) if (t_first and ct > 1 and t1 > t_first) else None
    return {"ttft": ttft, "total": t1 - t0, "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": ct, "decode_tps": dec, "text": "".join(text)}


def tokens_per_word(base, headers, model):
    sample = " ".join(random.choice(WORDS) for _ in range(2000))
    try:
        r = requests.post(f"{base}/tokenize", json={"model": model, "prompt": sample},
                          headers=headers, timeout=120)
        r.raise_for_status()
        return r.json()["count"] / 2000
    except Exception:
        return 1.4


def build_prompt(n_words, code):
    words = [random.choice(WORDS) for _ in range(n_words)]
    words.insert(n_words // 2, f". The secret passphrase is {code}. ")
    return " ".join(words)


def run(args):
    base = f"http://127.0.0.1:{args.port}"
    key = open(os.path.expanduser(args.key_file)).read().strip()
    headers = {"Authorization": f"Bearer {key}"}
    random.seed(time.time())
    out = {"label": args.label, "short_decode": None, "lengths": []}

    q = "What is the secret passphrase? Reply with the passphrase, then write about 150 words describing what this text looks like."
    chat_stream(base, headers, args.model, [{"role": "user", "content": "Say hi."}], 8)  # warm-up

    try:
        r = chat_stream(base, headers, args.model,
                        [{"role": "user", "content": "Explain in detail how TCP congestion control works."}], 400)
        out["short_decode"] = {k: r[k] for k in ("ttft", "decode_tps", "completion_tokens")}
        print(f"[short] decode {r['decode_tps']:.1f} tok/s" if r["decode_tps"] else "[short] n/a", flush=True)
    except Exception as e:
        out["short_decode"] = {"error": str(e)[:300]}

    tpw = tokens_per_word(base, headers, args.model)
    for target in args.lengths:
        entry = {"target": target}
        try:
            code = "".join(random.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(10))
            prompt = build_prompt(int(target / tpw), code)
            msgs = [{"role": "user", "content": prompt + "\n\n" + q}]
            cold = chat_stream(base, headers, args.model, msgs, args.decode_tokens)
            entry.update(prompt_tokens=cold["prompt_tokens"], ttft=cold["ttft"],
                         decode_tps=cold["decode_tps"], needle_ok=code in cold["text"])
            warm = chat_stream(base, headers, args.model, msgs, 8)
            entry["ttft_cached"] = warm["ttft"]
            print(f"[{target}] prompt={entry['prompt_tokens']} ttft={entry['ttft']:.1f}s "
                  f"cached={entry['ttft_cached']:.2f}s decode={entry['decode_tps'] or 0:.1f} tok/s "
                  f"needle={entry['needle_ok']}", flush=True)
        except Exception as e:
            entry["error"] = str(e)[:300]
            print(f"[{target}] FAILED: {entry['error']}", flush=True)
            out["lengths"].append(entry)
            break  # longer prompts will not fit either
        out["lengths"].append(entry)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)


def summarize(d):
    f = lambda v, p=1: "-" if v is None else f"{v:.{p}f}"
    for path in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(path))
        sd = r.get("short_decode") or {}
        print(f"\n{r['label']}   short-prompt decode: {f(sd.get('decode_tps'))} tok/s")
        print(f"  {'prompt_tok':>10} {'ttft_s':>8} {'cached_s':>9} {'decode_tps':>11} {'needle':>7}")
        for e in r["lengths"]:
            if "error" in e:
                print(f"  {e['target']:>10} FAILED: {e['error'][:80]}")
            else:
                print(f"  {e['prompt_tokens']:>10} {f(e['ttft']):>8} {f(e['ttft_cached'], 2):>9} "
                      f"{f(e['decode_tps']):>11} {str(e['needle_ok']):>7}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--key-file", default="~/.config/qwen-local/api_key")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--lengths", type=int, nargs="+", default=[8000, 32000, 100000, 200000])
    ap.add_argument("--decode-tokens", type=int, default=200)
    ap.add_argument("--out", default="bench.json")
    ap.add_argument("--label", default="run")
    ap.add_argument("--summarize", metavar="DIR")
    a = ap.parse_args()
    summarize(a.summarize) if a.summarize else run(a)

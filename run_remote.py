"""Autonomous Kaggle notebook runner using only the Python standard library."""

import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

USER_NAME = "madiyarzhumazhan"
KERNEL_SLUG = "notebook0b34895d24"
BASE_URL = "https://www.kaggle.com/api/v1"
POLL_INTERVAL_SECONDS = 20

REPO_ROOT = Path(__file__).resolve().parent


def load_kaggle_credentials() -> tuple[str, str]:
    user_profile = os.environ.get("USERPROFILE") or os.path.expanduser("~")
    kaggle_json_path = Path(user_profile) / ".kaggle" / "kaggle.json"
    if not kaggle_json_path.is_file():
        raise FileNotFoundError(f"Kaggle credentials file not found: {kaggle_json_path}")

    with open(kaggle_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    username = data.get("username")
    key = data.get("key")
    if not username or not key:
        raise ValueError(f"Invalid credentials in {kaggle_json_path}: expected 'username' and 'key'")
    return username, key


def build_auth_headers(username: str, key: str) -> dict[str, str]:
    raw = f"{username}:{key}".encode("utf-8")
    b64 = base64.b64encode(raw).decode("ascii")
    return {
        "Authorization": f"Basic {b64}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Soup-Remote-Runner/1.0",
    }


def api_request(url: str, headers: dict[str, str], method: str = "GET", payload: dict | None = None) -> dict:
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {method} {url}: {err_body}") from exc


def sync_git_solution() -> None:
    print("[1/5] Checking git status for solution/...")
    status_proc = subprocess.run(
        ["git", "status", "--porcelain", "solution/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if status_proc.returncode != 0:
        print(f"Warning: git status exited with {status_proc.returncode}: {status_proc.stderr.strip()}")
        return

    changes = status_proc.stdout.strip()
    if changes:
        print("Detected changes in solution/:\n" + changes)
        subprocess.run(["git", "add", "solution/"], cwd=REPO_ROOT, check=True)
        commit_msg = f"Update solution Triton kernels ({time.strftime('%Y-%m-%d %H:%M:%S')})"
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=REPO_ROOT, check=True)
        print("Pushing changes to origin main...")
        subprocess.run(["git", "push", "origin", "main"], cwd=REPO_ROOT, check=True)
        print("Git push completed.")
    else:
        print("No uncommitted changes in solution/.")


def pull_kernel_blob(headers: dict[str, str]) -> dict:
    query = urllib.parse.urlencode({"userName": USER_NAME, "kernelSlug": KERNEL_SLUG})
    url = f"{BASE_URL}/kernels/pull?{query}"
    print(f"[2/5] Pulling notebook from {url} ...")
    resp = api_request(url, headers=headers, method="GET")
    blob = resp.get("blob", resp)
    if isinstance(blob, dict) and "text" not in blob and "source" in blob:
        blob["text"] = blob["source"]
    if not isinstance(blob, dict) or not blob.get("text"):
        fallback = REPO_ROOT / "templates" / "kaggle_notebook.ipynb"
        if fallback.is_file():
            blob = {"text": fallback.read_text(encoding="utf-8")}
        else:
            raise RuntimeError(f"Could not extract blob['text'] from pull response: {resp}")
    return blob


def push_kernel(headers: dict[str, str], blob: dict) -> dict:
    url = f"{BASE_URL}/kernels/push"
    payload = {
        "slug": f"{USER_NAME}/{KERNEL_SLUG}",
        "newTitle": KERNEL_SLUG,
        "text": blob["text"],
        "language": "python",
        "kernelType": "notebook",
        "isPrivate": True,
        "enableGpu": True,
        "enableInternet": True,
    }
    print(f"[3/5] Pushing notebook to {url} (triggering Run All on Kaggle T4)...")
    resp = api_request(url, headers=headers, method="POST", payload=payload)
    print(f"Push response: {json.dumps(resp, ensure_ascii=False)}")
    return resp


def poll_kernel_status(headers: dict[str, str]) -> tuple[str, dict]:
    query = urllib.parse.urlencode({"userName": USER_NAME, "kernelSlug": KERNEL_SLUG})
    url = f"{BASE_URL}/kernels/status?{query}"
    print(f"[4/5] Polling status every {POLL_INTERVAL_SECONDS}s at {url} ...")

    while True:
        resp = api_request(url, headers=headers, method="GET")
        raw_status = str(resp.get("status", "")).strip()
        norm_status = raw_status.lower().split(".")[-1]
        failure_msg = resp.get("failureMessage") or resp.get("failure_message") or ""
        ts = time.strftime("%H:%M:%S")
        print(f"  [{ts}] status = {raw_status}" + (f" ({failure_msg})" if failure_msg else ""))

        if norm_status in ("complete", "error", "cancelled", "cancelacknowledged"):
            return norm_status, resp

        time.sleep(POLL_INTERVAL_SECONDS)


def extract_log_text(output_resp: dict) -> str:
    raw_log = output_resp.get("log", "")
    if isinstance(raw_log, str):
        stripped = raw_log.strip()
        if stripped.startswith("["):
            try:
                events = json.loads(stripped)
                if isinstance(events, list):
                    return "".join(
                        str(item.get("data", "")) for item in events if isinstance(item, dict)
                    )
            except json.JSONDecodeError:
                pass
        return raw_log
    if isinstance(raw_log, list):
        parts = []
        for item in raw_log:
            if isinstance(item, dict):
                parts.append(str(item.get("data", item.get("message", ""))))
            else:
                parts.append(str(item))
        return "".join(parts)
    return json.dumps(output_resp, indent=2, ensure_ascii=False)


def fetch_and_print_output(headers: dict[str, str]) -> None:
    query = urllib.parse.urlencode({"userName": USER_NAME, "kernelSlug": KERNEL_SLUG})
    url = f"{BASE_URL}/kernels/output?{query}"
    print(f"[5/5] Fetching execution output from {url} ...")
    resp = api_request(url, headers=headers, method="GET")
    log_text = extract_log_text(resp)

    print("\n" + "=" * 80)
    print("FULL KAGGLE EXECUTION LOG")
    print("=" * 80)
    print(log_text)

    summary_lines = []
    for line in log_text.splitlines():
        if any(
            marker in line
            for marker in (
                "PASS  ",
                "FAIL  ",
                "NOTE  ",
                "ALL CHECKS PASSED",
                "SOME CHECKS FAILED",
                "speedup",
                "memory_ratio",
                "own_kernel_ms",
                "Score",
                "score",
            )
        ):
            summary_lines.append(line)

    if summary_lines:
        print("\n" + "=" * 80)
        print("HIGHLIGHTED CHECK_RULES & SCORE SUMMARY")
        print("=" * 80)
        print("\n".join(summary_lines))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    username, key = load_kaggle_credentials()
    headers = build_auth_headers(username, key)

    sync_git_solution()
    blob = pull_kernel_blob(headers)
    push_kernel(headers, blob)
    final_status, status_resp = poll_kernel_status(headers)

    fetch_and_print_output(headers)
    if final_status != "complete":
        print(f"\nKernel finished with status '{final_status}': {status_resp}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

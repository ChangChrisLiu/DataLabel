"""Download the annotated HumanSignal / Label Studio projects to D:\\DataSet\\raw_logs\\labelstudio\\.

Usage (run it yourself in a terminal; the token never leaves your machine):

    set LS_TOKEN=<your personal access token>        (PowerShell: $env:LS_TOKEN="...")
    D:\\Anaconda\\envs\\tda\\python.exe D:\\DataSet\\scripts\\ls_export.py

Create a token at https://app.humansignal.com/user/account (Access Token).
The script only reads (GET) and writes one JSON file per project plus a combined file.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

BASE = "https://app.humansignal.com"
OUT_DIR = r"D:\DataSet\raw_logs\labelstudio"
PROJECT_IDS = [195316, 195305, 195222, 195164, 195150, 195088, 195079, 195067, 194954, 194953, 194952,
               194907, 194897, 176047, 176046, 176044, 176043, 176041, 176039, 170872, 163678, 163344,
               163341, 163281, 163276, 162649]


def get(path: str, token: str):
    req = urllib.request.Request(BASE + path, headers={"Authorization": f"Token {token}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    token = os.environ.get("LS_TOKEN", "").strip()
    if not token:
        print("LS_TOKEN is not set; see the docstring.")
        return 2
    os.makedirs(OUT_DIR, exist_ok=True)
    projects = []
    page = 1
    while True:
        j = get(f"/api/projects?page_size=100&page={page}", token)
        projects.extend(j.get("results", []))
        if not j.get("next"):
            break
        page += 1
    meta = {p["id"]: p for p in projects}
    combined = {"exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": BASE, "projects": []}
    for pid in PROJECT_IDS:
        tasks = []
        page = 1
        while True:
            j = get(f"/api/tasks?project={pid}&page_size=100&page={page}&fields=all", token)
            t = j.get("tasks") or j.get("results") or []
            tasks.extend(t)
            if len(t) < 100:
                break
            page += 1
        m = meta.get(pid, {})
        rec = {"id": pid, "title": m.get("title"), "workspace": m.get("workspace_title"),
               "label_config": m.get("label_config"), "tasks": tasks}
        combined["projects"].append(rec)
        with open(os.path.join(OUT_DIR, f"ls_project_{pid}.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)
        n_ann = sum(1 for t in tasks if t.get("annotations"))
        print(f"{pid} {m.get('title')} {m.get('workspace_title')}: {len(tasks)} tasks, {n_ann} annotated")
    with open(os.path.join(OUT_DIR, "humansignal_annotated_projects_export.json"), "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False)
    print("done ->", OUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())

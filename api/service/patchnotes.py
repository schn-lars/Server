from pathlib import Path
import time
import requests
from datetime import datetime
from typing import Optional
import threading
import os
from dotenv import load_dotenv

load_dotenv("/app/.env")

refresh_repo = True
LOG_FILE = Path("../resources/commits.txt")
GITHUB_API_KEY = os.getenv("GITHUB_KEY", "DUMMY_KEY")
lock = threading.Lock()

def load_existing_logs():
    if LOG_FILE.exists():
        return set(LOG_FILE.read_text().splitlines())
    return set()

def append_new_logs(new_logs, existing_logs):
    existing_shas = set(log.strip().split("|", 2)[1] for log in existing_logs)
    new_ordered_logs = list(reversed(new_logs))
    logs_to_append = [log for log in new_ordered_logs if log['hash'] not in existing_shas]
    if not logs_to_append:
        print("No new logs to append")
        return
    with LOG_FILE.open("a") as f:
        for entry in logs_to_append:
            line = f"{entry['date']}|{entry['hash']}|{entry['message']}"
            f.write(line + "\n")

def puller():
    global refresh_repo
    while True:
        with lock:
            refresh_repo = True
        time.sleep(43200)

def try_pull():
    global refresh_repo
    try:
        with lock:
            if refresh_repo:
                print("Starting auto-pulling...")
                new_logs = fetch_commits_from_github()
                existing_logs = load_existing_logs()
                append_new_logs(new_logs, existing_logs)
                refresh_repo = False
                #print("Finish auto-pull")
    except Exception as e:
        print(f"Error doing auto-pull: {str(e)}")

def fetch_commits_from_github():
    #print("Fetching commits")
    url = f"https://api.github.com/repos/schn-lars/MrIntenso/commits?sha=main&per_page=100"
    headers = {"Accept": "application/vnd.github.v3+json"}
    headers['Authorization'] = f"token {GITHUB_API_KEY}"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    commit_logs = []
    for item in response.json():
        date = item['commit']['author']['date']
        dt = datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ")
        formatted_date = dt.strftime("%d.%m.%y")
        sha = item['sha']
        message = item['commit']['message'].replace('\n', ' ')
        commit_logs.append({
            "date": formatted_date,
            "hash": sha,
            "message": message
        })
    return commit_logs

def get_patch_notes(commit_hash: Optional[str] = ""):
    global refresh_repo # good for debugging
    try:
        refresh_repo = True # good for debugging
        try_pull()
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
        commits = []
        last_hash = ""
        found = commit_hash == ""
        for _, line in enumerate(lines):
            if not found:
                if f"|{commit_hash}|" in line:
                    found = True
                contents = line.strip().split("|", 2)
                last_hash = contents[1]
                continue
            date, hash, message = line.strip().split("|", 2)
            last_hash = hash
            commits.append({
                "date": date,
                "hash": hash,
                "message": message
            })
        return last_hash, commits
    except:
        raise Exception()
"""Refresh the local GUI server — stop the old process, start a new one.

Usage:
    python refresh_server.py             # restart the server (reloads code + config.json)
    python refresh_server.py --fresh     # restart with a wiped companion.db*
    python refresh_server.py --stop      # only stop the running server
    python refresh_server.py --logs      # show the last 30 lines of server.log
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOST, PORT = "127.0.0.1", 8765
LOG = ROOT / "server.log"
URL = f"http://{HOST}:{PORT}/api/state"
DB_FILES = ("companion.db", "companion.db-wal", "companion.db-shm")


def _listeners() -> list[int]:
    try:
        out = subprocess.run(["netstat", "-ano"],
                             capture_output=True, text=True).stdout
    except OSError:
        return []
    pids: list[int] = []
    for line in out.splitlines():
        if f":{PORT}" in line and "LISTENING" in line.upper():
            pid = line.rsplit(None, 1)[-1]
            if pid.isdigit():
                pids.append(int(pid))
    return pids


def stop() -> None:
    for pid in _listeners():
        print(f"stopping server (pid {pid})")
        try:
            subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                           capture_output=True)
        except OSError:
            os.kill(pid, signal.SIGKILL)


def start() -> int:
    log = open(LOG, "ab")
    # Detach so the server survives the refresh script exiting and no console
    # window pops up; stdout/stderr go to server.log.
    flags = 0
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen(
        # -u: stdout to a redirected file is otherwise block-buffered and the
        # startup lines would not reach server.log until the server exits.
        [sys.executable, "-u", "server.py"], cwd=ROOT,
        stdin=subprocess.DEVNULL, stdout=log, stderr=log, creationflags=flags)
    log.close()
    return proc.pid


def _up() -> bool:
    try:
        with urllib.request.urlopen(URL, timeout=1) as r:
            return r.status == 200
    except OSError:
        return False


def wait_up(pid: int, timeout: float = 12.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _up():
            return True
        if os.name == "nt" and not _alive(pid):
            return False
        time.sleep(0.3)
    return _up()


def _alive(pid: int) -> bool:
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                             capture_output=True, text=True).stdout
    except OSError:
        return True
    return str(pid) in out


def log_tail(lines: int = 30) -> None:
    if not LOG.exists():
        print("no server.log yet")
        return
    text = LOG.read_text("utf-8", errors="replace").splitlines()
    print("\n".join(text[-lines:]))


def main() -> None:
    args = sys.argv[1:]
    if "--logs" in args:
        log_tail()
        return
    if "--stop" in args:
        stop()
        print("stopped.")
        return
    if "--fresh" in args:
        for name in DB_FILES:
            (ROOT / name).unlink(missing_ok=True)
        print("wiped companion.db*")
    stop()
    time.sleep(0.4)          # let the old process release the port / DB
    pid = start()
    if wait_up(pid):
        print(f"server up (pid {pid}) — {URL}")
    else:
        print(f"server did not come up (pid {pid}); last log lines:")
        log_tail(15)
        sys.exit(1)


if __name__ == "__main__":
    main()

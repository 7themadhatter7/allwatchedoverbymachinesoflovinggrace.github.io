#!/usr/bin/env python3
"""
RM Script 03 - Add /api/status to bridge_server.py
Targeted line-based insertion. Safe: backup created first.
"""
import shutil, re
from pathlib import Path

BRIDGE = Path("/home/joe/sparky/bridge_server.py")
BACKUP = Path("/home/joe/sparky/bridge_server.py.pre_status_patch")

src = BRIDGE.read_text()
if "/api/status" in src:
    print("[OK] /api/status already present")
    exit(0)

shutil.copy(BRIDGE, BACKUP)
print(f"Backup: {BACKUP}")

lines = src.splitlines()

# Step 1: inject _rm_status dict after last import
last_import = max(
    (i for i, ln in enumerate(lines)
     if ln.startswith("import ") or ln.startswith("from ")),
    default=0
)
status_init = [
    "",
    "# RM status tracking",
    "import time as _rm_time",
    "_rm_status = {'start_time': _rm_time.time(), 'requests_total': 0,",
    "              'requests_error': 0, 'last_request_ts': None}",
    "",
]
for offset, line in enumerate(status_init):
    lines.insert(last_import + 1 + offset, line)
src = "\n".join(lines)
lines = src.splitlines()

# Step 2: find do_GET, locate first path-routing if statement
in_do_get = False
insert_at = None
indent = "        "
for i, line in enumerate(lines):
    if re.match(r"\s+def do_GET", line):
        in_do_get = True
        continue
    if in_do_get:
        if re.search(r"if (?:path|self\.path).*==", line):
            insert_at = i
            m = re.match(r"(\s+)", line)
            indent = m.group(1) if m else "        "
            break
        if re.match(r"\s+def \w", line) and "do_GET" not in line:
            break

if insert_at is None:
    print("[FAIL] Cannot locate path routing in do_GET - manual patch needed")
    import subprocess
    subprocess.run(["grep", "-n", "def do_GET", str(BRIDGE)])
    exit(1)

handler = [
    f'{indent}if self.path == "/api/status":',
    f'{indent}    import json as _sj, time as _st',
    f'{indent}    _rm_status["requests_total"] += 1',
    f'{indent}    body = _sj.dumps({{"service":"bridge_server","status":"running",',
    f'{indent}        "uptime_seconds":int(_st.time()-_rm_status["start_time"]),',
    f'{indent}        "requests_total":_rm_status["requests_total"],',
    f'{indent}        "requests_error":_rm_status["requests_error"]}}).encode()',
    f'{indent}    self.send_response(200)',
    f'{indent}    self.send_header("Content-Type","application/json")',
    f'{indent}    self.send_header("Content-Length",len(body))',
    f'{indent}    self.end_headers()',
    f'{indent}    self.wfile.write(body)',
    f'{indent}    return',
    "",
]
for offset, line in enumerate(handler):
    lines.insert(insert_at + offset, line)

BRIDGE.write_text("\n".join(lines))

import subprocess
r = subprocess.run(["python3", "-m", "py_compile", str(BRIDGE)],
                   capture_output=True, text=True)
if r.returncode == 0:
    print("[OK] bridge_server.py patched - syntax clean")
    print("     Restart: sudo systemctl restart rm-bridge.service")
else:
    print("[FAIL] Syntax error - restoring backup")
    shutil.copy(BACKUP, BRIDGE)
    print(r.stderr[:400])

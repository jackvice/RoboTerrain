#!/usr/bin/env python3
import subprocess, time, sys, math, shutil

topic = "/world/inspect/pose/info"
if len(sys.argv) > 1:
    topic = sys.argv[1]

# quick sanity check
if not shutil.which("ign"):
    print("Error: 'ign' not found in PATH")
    sys.exit(1)

# Spawn echo
p = subprocess.Popen(
    ["ign", "topic", "-e", "-t", topic],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1,
)

msg_count = 0
last_print = time.time()
ema_hz = None
window_msgs = 0
window_start = time.time()

def update(rate):
    global ema_hz
    alpha = 0.3
    ema_hz = rate if ema_hz is None else (alpha * rate + (1 - alpha) * ema_hz)

try:
    for line in p.stdout:
        # A blank line marks the end of one full protobuf message as printed by `ign topic -e`
        if line.strip() == "":
            msg_count += 1
            window_msgs += 1

        now = time.time()
        if now - last_print >= 1.0:
            elapsed = now - window_start
            hz = window_msgs / elapsed if elapsed > 0 else 0.0
            update(hz)
            print(f"{topic}: {hz:.2f} Hz (EMA {ema_hz:.2f})")
            window_msgs = 0
            window_start = now
            last_print = now
except KeyboardInterrupt:
    pass
finally:
    p.terminate()

"""Trains the simple and the unicycle network at the same time, each in its own process.

Both train_simple.py and train_unicycle.py are started in parallel. To keep the terminal readable, each child process
sends its progress bar and printed text to this parent process (via progress.py's ProgressHub), which draws one bar per
training run. Prints SUCCESS / FAILED for each script at the end.

Run:  python train_all.py        (from inside the Neural Networks folder)
"""
import os
import sys
import runpy
import multiprocessing

import progress

# Runs in a child process: redirect progress bars and print output to the parent, then execute the given training script
# exactly as if it had been started from the command line (run_name='__main__').
def run_script(script_name, slot, queue):
    # The parent owns the terminal: this process reports its progress bars and printed text to it,
    # so the two training runs never draw over each other.
    progress.attach_remote(queue, slot)
    writer = progress.QueueWriter(queue, slot)
    sys.stdout = sys.stderr = writer
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, script_name)
    os.chdir(script_dir)
    sys.path.insert(0, script_dir)
    try:
        runpy.run_path(script_path, run_name="__main__")
    finally:
        writer.flush()

# Main process: start the display hub, launch one process per script, wait for both and report the exit status.
if __name__ == "__main__":
    scripts = ["train_simple.py", "train_unicycle.py"]
    queue = multiprocessing.Queue()
    hub = progress.ProgressHub(queue).start()
    processes = []
    for slot, script in enumerate(scripts):
        p = multiprocessing.Process(target=run_script, args=(script, slot, queue), name=script)
        p.start()
        processes.append(p)
        hub.write(f"[Core Launcher] Initialized concurrent training process: {script} (PID {p.pid})")
    for p in processes:
        p.join()
    hub.stop()
    for p in processes:
        status = "SUCCESS" if p.exitcode == 0 else f"FAILED (exit code {p.exitcode})"
        print(f"[Core Launcher] Execution finished for {p.name} -> Status: {status}")

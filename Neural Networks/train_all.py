import os
import sys
import runpy
import multiprocessing

import progress

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

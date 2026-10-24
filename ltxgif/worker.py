"""Keep LTX loaded between notebook cell runs.

    w = get_worker(cfg, lora)                       # starts (or reuses) a background LTX process for this checkpoint
    w.generate(image, prompt, seed, "out.mp4")      # first call loads the model; later calls only sample
    stop_worker()                                   # frees the GPU (e.g. before resuming training)

A worker serves one LoRA (LTX builds the pipeline with its LoRAs fused in); asking for a different checkpoint
restarts it. The worker holds ~20+ GB of GPU memory until stopped, so stop it before training."""
import atexit
import json
import os
import shutil
import signal
import subprocess
import time
import uuid
from pathlib import Path

from . import ltx

_current = None


class Worker:
    def __init__(self, cfg, lora=None, steps=30, quantization=None):
        self.args = ltx.generation_args(cfg, lora, steps=steps, quantization=quantization)
        self.lora = str(Path(lora).resolve()) if lora else None
        self.dir = Path(cfg.work_dir).resolve() / "worker"
        _kill_stale(self.dir)
        shutil.rmtree(self.dir, ignore_errors=True)
        (self.dir / "in").mkdir(parents=True)
        (self.dir / "common.json").write_text(json.dumps(self.args))
        self.log_path = self.dir / "log.txt"
        self._log = open(self.log_path, "w")
        self._pos = 0
        env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
        self.proc = subprocess.Popen(
            ["uv", "run", "--project", str(Path(cfg.ltx_repo).resolve()), "python",
             str(ltx.REPO_ROOT / "ltxgif" / "serve_ltx.py"), str(self.dir)],
            cwd=cfg.ltx_repo, env=env, stdout=self._log, stderr=subprocess.STDOUT, start_new_session=True)
        (self.dir / "pid").write_text(str(self.proc.pid))
        print(f"started LTX worker (pid {self.proc.pid}) · LoRA: {self.lora or 'none (base model)'} · log: {self.log_path}")

    @property
    def alive(self):
        return self.proc.poll() is None

    def _tail(self, show):
        text = self.log_path.read_text(errors="replace")
        new, self._pos = text[self._pos:], len(text)
        if show and new.strip():
            print(new, end="" if new.endswith("\n") else "\n", flush=True)
        return text

    def generate(self, image, prompt, seed=42, output="out.mp4", show_log=True, timeout=3600):
        """Queue one image-to-video job and wait for it. Returns the MP4 path."""
        output = str(Path(output).resolve())
        name = f"{time.time_ns()}_{uuid.uuid4().hex[:6]}.json"
        job = {"image": str(Path(image).resolve()), "prompt": prompt, "seed": seed, "output": output}
        tmp = self.dir / "in" / (name + ".tmp")
        tmp.write_text(json.dumps(job))
        tmp.replace(self.dir / "in" / name)
        done, t0 = self.dir / "done" / name, time.time()
        while not done.exists():
            if not self.alive:
                log = self._tail(False)
                raise RuntimeError(f"LTX worker exited (code {self.proc.returncode}); last log lines:\n"
                                   + "\n".join(log.splitlines()[-30:]))
            if time.time() - t0 > timeout:
                raise TimeoutError(f"job not finished after {timeout}s; see {self.log_path}")
            self._tail(show_log)
            time.sleep(1)
        self._tail(show_log)
        result = json.loads(done.read_text())
        if not result["ok"]:
            raise RuntimeError(f"generation failed: {result['error']} (traceback in {self.log_path})")
        print(f"generated in {result['seconds']}s")
        return output

    def close(self):
        if self.alive:
            (self.dir / "stop").touch()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=30)
        self._log.close()
        print("LTX worker stopped")


def _kill_stale(queue_dir):
    """Stop a worker left over from before a kernel restart (it would still hold GPU memory)."""
    pid_file = Path(queue_dir) / "pid"
    if pid_file.exists():
        try:
            os.killpg(int(pid_file.read_text()), signal.SIGTERM)
            print("stopped a leftover LTX worker")
            time.sleep(3)
        except (ProcessLookupError, ValueError, PermissionError):
            pass


def get_worker(cfg, lora=None, steps=30):
    """The running worker if it serves the same checkpoint and settings; otherwise restart it."""
    global _current
    want = ltx.generation_args(cfg, lora, steps=steps)
    if _current is not None and _current.alive and _current.args == want:
        return _current
    stop_worker()
    _current = Worker(cfg, lora, steps=steps)
    return _current


def stop_worker():
    global _current
    if _current is not None:
        _current.close()
        _current = None


atexit.register(stop_worker)

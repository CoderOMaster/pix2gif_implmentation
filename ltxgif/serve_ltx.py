"""Persistent LTX generation worker (runs inside the LTX-2 environment; started by ltxgif.worker.Worker).

    uv run --project LTX-2 python ltxgif/serve_ltx.py <queue dir>

<queue dir>/common.json holds the shared CLI arguments (model paths, LoRA, size...). Jobs are JSON files in
<queue dir>/in ({"image", "prompt", "seed", "output"}); each result is written to <queue dir>/done/<same name> as
{"ok": true} or {"ok": false, "error": ...}. The pipeline is built on the first job and kept for the worker's lifetime.
Creating <queue dir>/stop ends the worker."""
import json
import sys
import time
import traceback
from pathlib import Path


def main(queue):
    queue = Path(queue)
    common = json.loads((queue / "common.json").read_text())
    (queue / "done").mkdir(exist_ok=True)
    from ltx_pipelines import ti2vid_one_stage as one_stage

    cache = {}
    build = one_stage.TI2VidOneStagePipeline

    def reuse(**kwargs):
        if "pipeline" not in cache:
            cache["pipeline"] = build(**kwargs)
        return cache["pipeline"]

    one_stage.TI2VidOneStagePipeline = reuse
    print("worker ready", flush=True)
    while not (queue / "stop").exists():
        jobs = sorted((queue / "in").glob("*.json"))
        if not jobs:
            time.sleep(0.3)
            continue
        path = jobs[0]
        job = json.loads(path.read_text())
        t0 = time.time()
        try:
            Path(job["output"]).parent.mkdir(parents=True, exist_ok=True)
            sys.argv = ["ti2vid_one_stage", *common, "--image", job["image"], "0", "1.0", "--prompt", job["prompt"],
                        "--seed", str(job.get("seed", 42)), "--output-path", job["output"]]
            one_stage.main()
            result = {"ok": True, "seconds": round(time.time() - t0, 1)}
        except Exception as e:
            traceback.print_exc()
            result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        tmp = queue / "done" / (path.name + ".tmp")
        tmp.write_text(json.dumps(result))
        tmp.replace(queue / "done" / path.name)
        path.unlink()
        print(f"job {path.stem}: {result}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

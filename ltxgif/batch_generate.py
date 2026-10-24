"""Generate many clips with one model load (runs inside the LTX-2 environment).

    uv run --project LTX-2 python ltxgif/batch_generate.py jobs.json

jobs.json: {"common": [CLI args shared by every job: model paths, --lora, size, frames, ...],
            "jobs": [{"image": path, "prompt": str, "seed": int, "output": path}, ...]}
Each job goes through LTX's own ti2vid_one_stage command-line entry point, so argument handling matches the CLI
exactly; the pipeline object is built on the first job and reused for the rest. Existing outputs are skipped."""
import json
import sys
from pathlib import Path


def main(jobs_path):
    spec = json.loads(Path(jobs_path).read_text())
    from ltx_pipelines import ti2vid_one_stage as one_stage

    cache = {}
    build = one_stage.TI2VidOneStagePipeline

    def reuse(**kwargs):
        if "pipeline" not in cache:
            cache["pipeline"] = build(**kwargs)
        return cache["pipeline"]

    one_stage.TI2VidOneStagePipeline = reuse
    jobs = [j for j in spec["jobs"] if not Path(j["output"]).exists()]
    print(f"{len(spec['jobs']) - len(jobs)} outputs exist, generating {len(jobs)}", flush=True)
    for i, job in enumerate(jobs):
        Path(job["output"]).parent.mkdir(parents=True, exist_ok=True)
        sys.argv = ["ti2vid_one_stage", *spec["common"], "--image", job["image"], "0", "1.0",
                    "--prompt", job["prompt"], "--seed", str(job.get("seed", 42)), "--output-path", job["output"]]
        one_stage.main()
        print(f"[{i + 1}/{len(jobs)}] {job['output']}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

"""Optional: rewrite TGIF captions as explicit animation instructions with a video-capable Gemini model.

The model watches the clip and returns a caption in the form
  subject and scene + explicit action + direction / type of movement + camera behaviour,
or flags the clip when no identifiable action happens (those clips are dropped)."""
import json
from concurrent.futures import ThreadPoolExecutor

from .tgif import load_records, write_manifest

PROMPT = """This is a short animated clip. Its first frame is the starting image of an image-animation training example.
Original human caption: "{caption}"

Write one caption for the clip in this form: subject and scene, then the explicit action, then the direction or type
of movement, then camera behaviour (for example "The camera remains static." or "The camera pans left.").
Describe only motion that is visible in the clip; do not invent actions. One to three sentences, at most 60 words.
Set action_visible to false if no identifiable action happens, the clip is a slideshow or collage, or it is mostly text."""


def _schema():
    from google.genai import types
    return types.Schema(type="OBJECT", required=["action_visible", "caption"], properties={
        "action_visible": types.Schema(type="BOOLEAN"),
        "caption": types.Schema(type="STRING"),
    })


def recaption(cfg, split, api_key, model="gemini-flash-latest", workers=8, limit=None):
    """Recaption every kept clip of a split that has not been recaptioned yet, then rewrite the manifest."""
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    conf = types.GenerateContentConfig(response_mime_type="application/json", response_schema=_schema(),
                                       temperature=0.2)
    todo = [r for r in load_records(cfg, split) if "video" in r and not r.get("reject") and "recaptioned" not in r]
    todo = sorted(todo, key=lambda r: r["id"])[:limit]
    print(f"{split}: recaptioning {len(todo)} clips with {model}")

    def one(r):
        video = (cfg.data_dir / r["video"]).read_bytes()
        try:
            resp = client.models.generate_content(model=model, config=conf, contents=[
                types.Part.from_bytes(data=video, mime_type="video/mp4"),
                PROMPT.format(caption=r["caption_orig"])])
            out = json.loads(resp.text)
        except Exception as e:
            return {"id": r["id"], "recaption_error": f"{type(e).__name__}: {e}"[:200]}
        upd = {"id": r["id"], "recaptioned": True, "caption": out["caption"].strip()}
        if not out["action_visible"]:
            upd["reject"] = "no_visible_action"
        return upd

    stats = {"ok": 0, "rejected": 0, "error": 0}
    with ThreadPoolExecutor(workers) as ex, open(cfg.records(split), "a") as f:
        for i, upd in enumerate(ex.map(one, todo)):
            f.write(json.dumps(upd) + "\n")
            stats["error" if "recaption_error" in upd else "rejected" if upd.get("reject") else "ok"] += 1
            if (i + 1) % 500 == 0:
                f.flush()
                print(f"  {i + 1}/{len(todo)} {stats}")
    print(f"{split}: {stats} (errors keep the original caption; rerun to retry them)")
    return write_manifest(cfg, split)


def show_examples(cfg, split, n=10):
    """Print original vs rewritten captions for a quick hallucination check."""
    recs = [r for r in load_records(cfg, split) if r.get("recaptioned")][:n]
    for r in recs:
        flag = "  [REJECTED]" if r.get("reject") else ""
        print(f"{r['id']}{flag}\n  TGIF:   {r['caption_orig']}\n  Gemini: {r['caption']}\n")

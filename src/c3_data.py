#@title 3 · Build the TGIF motion-pair dataset (paper Sec. 3.3 + Algorithm 1)
import io, json, random, tarfile, time
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np, requests
from PIL import Image, ImageSequence

TSV_URL = "https://raw.githubusercontent.com/raingo/TGIF-Release/master/data/tgif-v1.0.tsv"
JSON_NAME = "train_val_dataset_motion_2_20_reduced.json"  # same file name/format the official edit_dataset.py reads
TAR_PATH = f"{ROOT}/tgif_pairs_{N_GIFS}.tar"


def flow_magnitude(f0, f1):
    """Algorithm 1: grayscale -> features to track (Shi-Tomasi) -> Lucas-Kanade -> |displacement|,
    drop small values, average."""
    g0, g1 = cv2.cvtColor(f0, cv2.COLOR_RGB2GRAY), cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
    p0 = cv2.goodFeaturesToTrack(g0, maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7)
    if p0 is None or len(p0) < 5:
        return None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, winSize=(15, 15), maxLevel=2,
                                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    if p1 is None:
        return None
    ok = st.reshape(-1) == 1
    if ok.sum() < 5:
        return None
    mag = np.linalg.norm((p1 - p0).reshape(-1, 2)[ok], axis=1)
    mag = mag[mag > FLOW_EPS]
    return float(mag.mean()) if len(mag) else 0.0


def gif_frames(data, max_frames=96):
    im = Image.open(io.BytesIO(data))
    frames = []
    for i, f in enumerate(ImageSequence.Iterator(im)):
        if i >= max_frames:
            break
        # official edit_dataset.py resizes straight to 256×256 (no crop); flow is measured at that scale too
        frames.append(np.asarray(f.convert("RGB").resize((RES, RES), Image.Resampling.LANCZOS)))
    return frames


def process_gif(args):
    gid, url, caption = args
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        frames = gif_frames(r.content)
    except Exception:
        return []
    n = len(frames)
    if n < 2:
        return []
    cands = [(i, j) for i in range(n) for j in range(i + 1, n)]  # forward pairs (source earlier than target)
    random.Random(gid).shuffle(cands)
    kept = []
    for i, j in cands[:80]:
        m = flow_magnitude(frames[i], frames[j])
        if m is not None and MOTION_RANGE[0] <= m <= MOTION_RANGE[1]:
            kept.append((i, j, m))
        if len(kept) >= PAIRS_PER_GIF:
            break
    out = []
    for i, j, m in kept:
        paths = []
        for k in (i, j):
            p = f"{DATA_DIR}/frames/{gid:06d}_{k:03d}.jpg"
            if not os.path.exists(p):
                Image.fromarray(frames[k]).save(p, quality=95)
            paths.append(p)
        out.append([paths[0], paths[1], caption.strip(), round(m, 3), gid])
    return out


def build_dataset():
    os.makedirs(f"{DATA_DIR}/frames", exist_ok=True)
    rows = [l.split("\t") for l in requests.get(TSV_URL, timeout=60).text.strip().split("\n")]
    random.Random(SEED).shuffle(rows)
    jobs = [(i, u, c) for i, (u, c) in enumerate(rows[:N_GIFS])]
    pairs, t0 = [], time.time()
    with ThreadPoolExecutor(32) as ex:
        for k, res in enumerate(ex.map(process_gif, jobs), 1):
            pairs += res
            if k % 250 == 0:
                print(f"  {k}/{len(jobs)} GIFs · {len(pairs)} pairs · {time.time() - t0:.0f}s")
    gids = sorted({p[4] for p in pairs})
    val_gids = set(random.Random(SEED).sample(gids, max(1, int(len(gids) * VAL_FRACTION))))  # split by GIF, no leakage
    split = {"train": [p[:4] for p in pairs if p[4] not in val_gids],
             "val": [p[:4] for p in pairs if p[4] in val_gids]}
    with open(f"{DATA_DIR}/{JSON_NAME}", "w") as f:
        json.dump(split, f)
    with tarfile.open(TAR_PATH, "w") as t:
        t.add(DATA_DIR, arcname=os.path.basename(DATA_DIR))
    return split


if os.path.exists(f"{DATA_DIR}/{JSON_NAME}"):
    split = json.load(open(f"{DATA_DIR}/{JSON_NAME}"))
elif os.path.exists(TAR_PATH):
    print("restoring dataset from", TAR_PATH)
    with tarfile.open(TAR_PATH) as t:
        t.extractall(os.path.dirname(DATA_DIR))
    split = json.load(open(f"{DATA_DIR}/{JSON_NAME}"))
else:
    print(f"downloading {N_GIFS} TGIF GIFs and mining motion pairs (~{N_GIFS // 400 + 1} min)...")
    split = build_dataset()

print(f"train pairs: {len(split['train'])} · val pairs: {len(split['val'])}")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 4, figsize=(16, 3.2))
ax[0].hist([p[3] for p in split["train"]], bins=36)
ax[0].set_title("flow magnitude of training pairs")
for a, (s, t, cap, m) in zip(ax[1:], random.Random(1).sample(split["train"], 3)):
    a.imshow(np.hstack([np.asarray(Image.open(s)), np.asarray(Image.open(t))]))
    a.set_title(f"{cap[:38]}… | flow {m:.1f}", fontsize=8)
    a.axis("off")
plt.show()

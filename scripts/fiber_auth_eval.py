#!/usr/bin/env python3
"""
Fiber-PUF Authentication Evaluation
====================================

Per-fiber model training + 5x5 cross-fiber attack matrix.
With tqdm progress bars and parallel video decoding.

Usage:
    python -u scripts/fiber_auth_eval.py
    python -u scripts/fiber_auth_eval.py --epochs 20 --img_size 224
"""

import os, sys, io, argparse, time, copy, json, csv, re, glob
import numpy as np, cv2, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models import get_model

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        total = kw.get("total", None)
        desc = kw.get("desc", "")
        for i, x in enumerate(it):
            if total and (i % max(1, total // 20) == 0 or i == total - 1):
                print(f"\r  {desc} {i+1}/{total}", end="", flush=True)
            yield x
        if total:
            print()

DOMAIN_DIRS = {"Green": "green_only", "GreenAndRed": "red_green_fixed", "RedChange": "red_green_dynamic"}
ALL_DOMAINS = ["Green", "GreenAndRed", "RedChange"]
LETTERS = [chr(i) for i in range(65, 91)]
CLASS_TO_IDX = {c: i for i, c in enumerate(LETTERS)}
TRAIN_R, VAL_R = 0.70, 0.15
OUT_DIR = os.path.join(ROOT, "results", "fiber_auth")


# ─── Data helpers ──────────────────────────────────────────────────────────

def collect_videos(data_root, fiber, domains=ALL_DOMAINS):
    vids = []
    for df in domains:
        dpath = os.path.join(data_root, df, fiber)
        if not os.path.isdir(dpath):
            continue
        for f in sorted(glob.glob(os.path.join(dpath, "*.avi"))):
            base = re.sub(r"\(\d+\)$", "", os.path.splitext(os.path.basename(f))[0]).strip().upper()
            if len(base) == 1 and base in LETTERS:
                vids.append({"path": f, "letter": base, "domain": DOMAIN_DIRS[df]})
    return vids


def _decode_one(args):
    """Decode a single video file (worker function for ThreadPoolExecutor)."""
    vi, v, img_size = args
    cap = cv2.VideoCapture(v["path"])
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
    cap.release()
    if not frames:
        return vi, v, None
    return vi, v, np.stack(frames)


def _load_videos_parallel(vids, img_size, desc="Loading", workers=6):
    """Decode all videos using a thread pool with tqdm progress."""
    results = [None] * len(vids)
    work = [(i, v, img_size) for i, v in enumerate(vids)]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_decode_one, w) for w in work]
        for fut in tqdm(
            [f.result() for f in futures] if len(futures) < 2 else
            (f.result() for f in futures),
            total=len(vids), desc=f"    {desc}", ncols=80, leave=True,
        ):
            results[fut[0]] = fut

    out = []
    for vi, v, arr in results:
        if arr is not None:
            out.append((vi, v, arr))
    return out


def load_and_split(vids, clip_len, stride, img_size, workers=6):
    all_frames, train_clips, val_clips, test_clips = {}, [], [], []
    decoded = _load_videos_parallel(vids, img_size, desc="Decode+split", workers=workers)
    for vi, v, arr in decoded:
        key = f"{v['domain']}/{v['letter']}/{vi}"
        all_frames[key] = arr
        n = len(arr)
        label = CLASS_TO_IDX[v["letter"]]
        t_end = int(n * TRAIN_R)
        v_end = int(n * (TRAIN_R + VAL_R))
        meta = {"label": label, "label_name": v["letter"], "video_name": key, "domain": v["domain"]}
        for clip_list, s0, s1 in [(train_clips, 0, t_end), (val_clips, t_end, v_end), (test_clips, v_end, n)]:
            for s in range(s0, s1 - clip_len + 1, stride):
                clip_list.append({**meta, "start_frame": s, "end_frame": s + clip_len})
    return all_frames, train_clips, val_clips, test_clips


def load_test_only(vids, clip_len, stride, img_size, workers=6):
    all_frames, clips = {}, []
    decoded = _load_videos_parallel(vids, img_size, desc="Decode", workers=workers)
    for vi, v, arr in decoded:
        key = f"{v['domain']}/{v['letter']}/{vi}"
        all_frames[key] = arr
        label = CLASS_TO_IDX[v["letter"]]
        meta = {"label": label, "label_name": v["letter"], "video_name": key, "domain": v["domain"]}
        for s in range(0, len(arr) - clip_len + 1, stride):
            clips.append({**meta, "start_frame": s, "end_frame": s + clip_len})
    return all_frames, clips


class ClipDS(Dataset):
    MEAN = np.float32([0.485, 0.456, 0.406])
    STD  = np.float32([0.229, 0.224, 0.225])

    def __init__(self, clips, frames, clip_len, augment=False):
        self.clips, self.frames, self.clip_len, self.aug = clips, frames, clip_len, augment

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        c = self.clips[idx]
        f = self.frames[c["video_name"]][c["start_frame"]:c["end_frame"]].copy()
        if len(f) < self.clip_len:
            f = np.concatenate([f, np.repeat(f[-1:], self.clip_len - len(f), axis=0)])
        f = f.astype(np.float32) / 255.0
        if self.aug and np.random.random() > 0.5:
            f = f[:, :, ::-1].copy()
        f3 = np.stack([f, f, f], axis=1)
        for ch in range(3):
            f3[:, ch] = (f3[:, ch] - self.MEAN[ch]) / self.STD[ch]
        return torch.from_numpy(f3), c["label"]


# ─── Training ─────────────────────────────────────────────────────────────

def train_fiber_model(fiber, data_root, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vids = collect_videos(data_root, fiber)
    print(f"    Videos: {len(vids)}  Letters: {len(set(v['letter'] for v in vids))}")

    af, tr, va, te = load_and_split(vids, args.clip_len, args.stride, args.img_size, workers=args.workers)
    print(f"    Clips: train={len(tr)} val={len(va)} test={len(te)}")

    bs = args.batch_size
    pin = device.type == "cuda"
    train_ld = DataLoader(ClipDS(tr, af, args.clip_len, augment=True), batch_size=bs, shuffle=True, num_workers=0, pin_memory=pin)
    val_ld   = DataLoader(ClipDS(va, af, args.clip_len), batch_size=bs, shuffle=False, num_workers=0, pin_memory=pin)
    test_ld  = DataLoader(ClipDS(te, af, args.clip_len), batch_size=bs, shuffle=False, num_workers=0, pin_memory=pin)

    model = get_model("cnn_pool", 26, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val, best_state = 0.0, None
    ep_bar = tqdm(range(1, args.epochs + 1), desc=f"    Train {fiber}", ncols=80, leave=True)
    for ep in ep_bar:
        model.train()
        for clips, labels in train_ld:
            clips, labels = clips.to(device), labels.to(device)
            optimizer.zero_grad()
            criterion(model(clips), labels).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        vc = vt = 0
        with torch.no_grad():
            for clips, labels in val_ld:
                clips, labels = clips.to(device), labels.to(device)
                vc += (model(clips).argmax(1) == labels).sum().item()
                vt += labels.size(0)
        val_acc = 100 * vc / vt if vt else 0

        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())

        ep_bar.set_postfix_str(f"val={val_acc:.1f}% best={best_val:.1f}%")

    model.load_state_dict(best_state)
    model.eval()

    # Same-fiber test
    all_preds = []
    tc = tt = 0
    with torch.no_grad():
        for clips, labels in test_ld:
            clips, labels = clips.to(device), labels.to(device)
            preds = model(clips).argmax(1)
            tc += (preds == labels).sum().item()
            tt += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
    test_acc = 100 * tc / tt if tt else 0

    dom_acc = defaultdict(lambda: {"c": 0, "t": 0})
    for i, c in enumerate(te):
        dom_acc[c["domain"]]["t"] += 1
        if all_preds[i] == c["label"]:
            dom_acc[c["domain"]]["c"] += 1
    dom_result = {d: round(100 * v["c"] / v["t"], 1) if v["t"] else 0 for d, v in dom_acc.items()}

    return model, best_val, test_acc, dom_result


# ─── Cross-fiber evaluation ───────────────────────────────────────────────

def cross_eval(model, fiber_data, data_root, args):
    device = next(model.parameters()).device
    vids = collect_videos(data_root, fiber_data)
    af, clips = load_test_only(vids, args.clip_len, args.stride, args.img_size, workers=args.workers)
    if not clips:
        return 0.0
    ld = DataLoader(ClipDS(clips, af, args.clip_len), batch_size=args.batch_size,
                    shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for cl, lb in ld:
            cl, lb = cl.to(device), lb.to(device)
            correct += (model(cl).argmax(1) == lb).sum().item()
            total += lb.size(0)
    return 100 * correct / total if total else 0.0


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Fiber-PUF authentication evaluation")
    p.add_argument("--data_root", default="videocapture")
    p.add_argument("--fibers", nargs="+", default=["Fiber1", "Fiber2", "Fiber3", "Fiber4", "Fiber5"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--clip_len", type=int, default=16)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--img_size", type=int, default=112)
    p.add_argument("--workers", type=int, default=8, help="Threads for parallel video decoding")
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    model_dir = os.path.join(OUT_DIR, "fiber_models")
    os.makedirs(model_dir, exist_ok=True)

    fibers = args.fibers
    models = {}
    same_fiber_acc = {}
    same_fiber_domain = {}

    total_t0 = time.perf_counter()

    # ── Phase 1: Train per-fiber models ─────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Phase 1: Training {len(fibers)} fiber-specific models")
    print(f"  (3 domains, temporal split, {args.epochs} epochs, {args.workers} decode threads)")
    print(f"{'='*70}")

    for fi, fiber in enumerate(fibers):
        print(f"\n  [{fi+1}/{len(fibers)}] {fiber}")
        t0 = time.perf_counter()
        model, best_val, test_acc, dom_acc = train_fiber_model(fiber, args.data_root, args)

        ckpt_path = os.path.join(model_dir, f"{fiber}.pth")
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_type": "cnn_pool", "num_classes": 26,
            "class_names": LETTERS, "clip_len": args.clip_len,
            "img_size": args.img_size, "input_mode": "gray",
            "fiber_name": fiber,
        }, ckpt_path)

        models[fiber] = model
        same_fiber_acc[fiber] = test_acc
        same_fiber_domain[fiber] = dom_acc
        elapsed = time.perf_counter() - t0

        print(f"    SAME-FIBER test: {test_acc:.1f}%  (best_val={best_val:.1f}%)  [{elapsed:.0f}s]")
        for d, a in sorted(dom_acc.items()):
            print(f"      {d:25s}: {a:.1f}%")

    # ── Phase 2: Cross-fiber attack matrix ──────────────────────────────
    n_cross = len(fibers) * (len(fibers) - 1)
    print(f"\n{'='*70}")
    print(f"  Phase 2: Cross-fiber attack ({n_cross} pairs)")
    print(f"{'='*70}")

    matrix = {}
    pair_i = 0
    for mf in fibers:
        matrix[mf] = {}
        for df in fibers:
            if mf == df:
                matrix[mf][df] = same_fiber_acc[mf]
                continue
            pair_i += 1
            print(f"\n  [{pair_i}/{n_cross}] {mf} model -> {df} data", flush=True)
            t0 = time.perf_counter()
            acc = cross_eval(models[mf], df, args.data_root, args)
            matrix[mf][df] = acc
            print(f"    Accuracy: {acc:.1f}%  [{time.perf_counter()-t0:.0f}s]")

    total_elapsed = time.perf_counter() - total_t0

    # ── Results ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  RESULTS: Fiber-PUF Authentication Matrix")
    print(f"  (total time: {total_elapsed/60:.1f} min)")
    print(f"{'='*70}\n")

    cw = 10
    label = "Model\\Data"
    header = f"  {label:<12}" + "".join(f"{f:>{cw}}" for f in fibers) + f"{'Avg':>{cw}}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    auth_accs, attack_accs = [], []
    for mf in fibers:
        row = f"  {mf:<12}"
        vals = []
        for df in fibers:
            a = matrix[mf][df]
            tag = " *" if mf == df else ""
            row += f"{a:>{cw-len(tag)}.1f}{tag}"
            vals.append(a)
            (auth_accs if mf == df else attack_accs).append(a)
        row += f"{sum(vals)/len(vals):>{cw}.1f}"
        print(row)

    avg_auth = sum(auth_accs) / len(auth_accs)
    avg_attack = sum(attack_accs) / len(attack_accs)
    gap = avg_auth - avg_attack

    print(f"\n  Authorized (same-fiber) avg:   {avg_auth:.1f}%")
    print(f"  Unauthorized (cross-fiber) avg: {avg_attack:.1f}%")
    print(f"  Auth gap:                       {gap:.1f} pp")
    print(f"  Chance level:                   {100/26:.1f}%")

    if avg_attack < 100 / 26 * 1.5:
        print(f"  --> Cross-fiber near chance: STRONG fiber separability")
    elif avg_attack < 20:
        print(f"  --> Cross-fiber low: GOOD fiber separability")
    else:
        print(f"  --> Cross-fiber elevated: fiber separability may be weak")

    # ── Save outputs ────────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, "auth_matrix.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_fiber"] + fibers + ["same_fiber_acc"])
        for mf in fibers:
            w.writerow([mf] + [f"{matrix[mf][df]:.2f}" for df in fibers] + [f"{same_fiber_acc[mf]:.2f}"])
        w.writerow([]); w.writerow(["authorized_avg", f"{avg_auth:.2f}"])
        w.writerow(["unauthorized_avg", f"{avg_attack:.2f}"])
        w.writerow(["auth_gap_pp", f"{gap:.2f}"])

    json_path = os.path.join(OUT_DIR, "auth_matrix.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "matrix": matrix, "same_fiber_accuracy": same_fiber_acc,
            "same_fiber_per_domain": same_fiber_domain,
            "authorized_avg": round(avg_auth, 2), "unauthorized_avg": round(avg_attack, 2),
            "auth_gap_pp": round(gap, 2), "chance_level": round(100/26, 2),
        }, f, indent=2)

    txt_path = os.path.join(OUT_DIR, "auth_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Fiber-PUF Authentication Summary\n" + "="*50 + "\n\n")
        f.write("Same-fiber (authorized) accuracy:\n")
        for fi in fibers:
            f.write(f"  {fi}: {same_fiber_acc[fi]:.1f}%\n")
            for d, a in sorted(same_fiber_domain[fi].items()):
                f.write(f"    {d}: {a:.1f}%\n")
        f.write(f"\n  Average: {avg_auth:.1f}%\n")
        f.write(f"\nCross-fiber (unauthorized) average: {avg_attack:.1f}%\n")
        f.write(f"Authentication gap: {gap:.1f} pp\n")

    print(f"\n  Saved: {csv_path}")
    print(f"  Saved: {json_path}")
    print(f"  Saved: {txt_path}")
    print(f"  Models: {model_dir}/")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()

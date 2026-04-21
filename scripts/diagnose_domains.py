#!/usr/bin/env python3
"""
Quick domain-ablation diagnostic.

Tests three configurations on a SINGLE fiber with temporal split
(the proven approach), isolating the effect of domain mixing:

  1. Green only
  2. Green + GreenAndRed
  3. Green + GreenAndRed + RedChange

Uses temporal split within each video (no cross-fiber confound).
Runs 15 epochs each — enough to see whether learning happens.

Usage:
    python scripts/diagnose_domains.py
    python scripts/diagnose_domains.py --fiber Fiber1 --epochs 15
"""

import os, sys, io, argparse, time, copy, json
import numpy as np, cv2, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter, defaultdict

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models import get_model

DOMAIN_DIRS = {
    "Green":       "green_only",
    "GreenAndRed": "red_green_fixed",
    "RedChange":   "red_green_dynamic",
}
LETTERS = [chr(i) for i in range(65, 91)]
CLASS_TO_IDX = {c: i for i, c in enumerate(LETTERS)}
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15


def collect_videos(data_root, fiber, domain_folders):
    """Return list of (path, letter, domain) for the given fiber and domains."""
    vids = []
    for dfolder in domain_folders:
        dpath = os.path.join(data_root, dfolder, fiber)
        if not os.path.isdir(dpath):
            print(f"  [SKIP] {dpath} not found")
            continue
        import glob, re
        for f in sorted(glob.glob(os.path.join(dpath, "*.avi"))):
            base = os.path.splitext(os.path.basename(f))[0]
            base = re.sub(r"\(\d+\)$", "", base).strip().upper()
            if len(base) == 1 and base in LETTERS:
                vids.append({"path": f, "letter": base, "domain": DOMAIN_DIRS[dfolder]})
    return vids


def load_and_split(vids, clip_len, stride, img_size):
    """Load frames, temporal split within each video, generate clips."""
    all_frames = {}
    train_clips, val_clips, test_clips = [], [], []

    for vi, v in enumerate(vids):
        if (vi + 1) % 10 == 0 or vi == 0 or vi == len(vids) - 1:
            print(f"    Loading video [{vi+1}/{len(vids)}] {os.path.basename(v['path'])}")
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
            continue

        arr = np.stack(frames)
        key = f"{v['domain']}/{v['letter']}/{vi}"
        all_frames[key] = arr
        n = len(arr)
        label = CLASS_TO_IDX[v["letter"]]

        t_end = int(n * TRAIN_RATIO)
        v_end = int(n * (TRAIN_RATIO + VAL_RATIO))

        meta = {"label": label, "label_name": v["letter"],
                "video_name": key, "domain": v["domain"]}

        for split_clips, s_start, s_end in [
            (train_clips, 0, t_end),
            (val_clips, t_end, v_end),
            (test_clips, v_end, n),
        ]:
            for s in range(s_start, s_end - clip_len + 1, stride):
                split_clips.append({**meta, "start_frame": s, "end_frame": s + clip_len})

    return all_frames, train_clips, val_clips, test_clips


class ClipDataset(Dataset):
    MEAN = np.float32([0.485, 0.456, 0.406])
    STD  = np.float32([0.229, 0.224, 0.225])

    def __init__(self, clips, frames, clip_len, augment=False):
        self.clips, self.frames, self.clip_len, self.augment = clips, frames, clip_len, augment

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        c = self.clips[idx]
        f = self.frames[c["video_name"]][c["start_frame"]:c["end_frame"]].copy()
        if len(f) < self.clip_len:
            f = np.concatenate([f, np.repeat(f[-1:], self.clip_len - len(f), axis=0)])
        f = f.astype(np.float32) / 255.0
        if self.augment and np.random.random() > 0.5:
            f = f[:, :, ::-1].copy()
        f3 = np.stack([f, f, f], axis=1)
        for ch in range(3):
            f3[:, ch] = (f3[:, ch] - self.MEAN[ch]) / self.STD[ch]
        return torch.from_numpy(f3), c["label"]


def quick_train(all_frames, train_clips, val_clips, test_clips, tag, args):
    """Train for a few epochs and return val/test accuracy."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = ClipDataset(train_clips, all_frames, args.clip_len, augment=True)
    val_ds   = ClipDataset(val_clips,   all_frames, args.clip_len)
    test_ds  = ClipDataset(test_clips,  all_frames, args.clip_len)

    kw = dict(batch_size=args.batch_size, num_workers=0, pin_memory=(device.type=="cuda"))
    train_ld = DataLoader(train_ds, shuffle=True, **kw)
    val_ld   = DataLoader(val_ds,   shuffle=False, **kw)
    test_ld  = DataLoader(test_ds,  shuffle=False, **kw)

    model = get_model("cnn_pool", 26, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val, best_state = 0.0, None
    print(f"\n  [{tag}] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}  device={device}")

    for ep in range(1, args.epochs + 1):
        model.train()
        correct = total = 0
        for clips, labels in train_ld:
            clips, labels = clips.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(clips), labels)
            loss.backward()
            optimizer.step()
            correct += (model(clips).argmax(1) == labels).sum().item()
            total += labels.size(0)
        train_acc = 100 * correct / total if total else 0
        scheduler.step()

        model.eval()
        vc = vt = 0
        all_vpreds = []
        with torch.no_grad():
            for clips, labels in val_ld:
                clips, labels = clips.to(device), labels.to(device)
                preds = model(clips).argmax(1)
                vc += (preds == labels).sum().item()
                vt += labels.size(0)
                all_vpreds.extend(preds.cpu().tolist())
        val_acc = 100 * vc / vt if vt else 0

        marker = "*" if val_acc > best_val else " "
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())

        if ep <= 3 or ep % 5 == 0 or ep == args.epochs:
            print(f"  {marker} ep {ep:02d}  train={train_acc:5.1f}%  val={val_acc:5.1f}%")

    # Val prediction histogram (using last epoch)
    pred_hist = Counter(all_vpreds)
    top3 = pred_hist.most_common(3)
    print(f"  Val pred histogram (top 3): {[(LETTERS[k],v) for k,v in top3]}")
    print(f"  Val unique classes predicted: {len(pred_hist)}")

    # Test with best model
    model.load_state_dict(best_state)
    model.eval()
    tc = tt = 0
    all_tpreds, all_tlabels = [], []
    with torch.no_grad():
        for clips, labels in test_ld:
            clips, labels = clips.to(device), labels.to(device)
            preds = model(clips).argmax(1)
            tc += (preds == labels).sum().item()
            tt += labels.size(0)
            all_tpreds.extend(preds.cpu().tolist())
            all_tlabels.extend(labels.cpu().tolist())
    test_acc = 100 * tc / tt if tt else 0

    # Per-domain test accuracy
    domain_acc = defaultdict(lambda: {"c": 0, "t": 0})
    for i, cl in enumerate(test_clips):
        domain_acc[cl["domain"]]["t"] += 1
        if all_tpreds[i] == all_tlabels[i]:
            domain_acc[cl["domain"]]["c"] += 1

    print(f"  RESULT: best_val={best_val:.1f}%  test={test_acc:.1f}%")
    for d in sorted(domain_acc):
        da = domain_acc[d]
        print(f"    {d:25s}: {100*da['c']/da['t']:5.1f}%  ({da['c']}/{da['t']})")

    return best_val, test_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="videocapture")
    p.add_argument("--fiber", default="Fiber1")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--clip_len", type=int, default=16)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--img_size", type=int, default=224)
    args = p.parse_args()

    configs = [
        ("Green_only",          ["Green"]),
        ("Green+GreenAndRed",   ["Green", "GreenAndRed"]),
        ("All_three_domains",   ["Green", "GreenAndRed", "RedChange"]),
    ]

    results = {}
    for tag, domains in configs:
        print(f"\n{'='*70}")
        print(f"  Experiment: {tag}  |  fiber={args.fiber}  |  domains={domains}")
        print(f"{'='*70}")

        t0 = time.perf_counter()
        vids = collect_videos(args.data_root, args.fiber, domains)
        print(f"  Videos found: {len(vids)}")

        label_check = all(CLASS_TO_IDX[v['letter']] == CLASS_TO_IDX.get(v['letter'], -1) for v in vids)
        letters_found = sorted(set(v['letter'] for v in vids))
        print(f"  Letters: {len(letters_found)} ({letters_found[0]}-{letters_found[-1]})")
        print(f"  Label mapping consistent: {label_check}")

        af, tr, va, te = load_and_split(vids, args.clip_len, args.stride, args.img_size)

        # Verify labels
        for name, clips in [("train", tr), ("val", va), ("test", te)]:
            lbls = Counter(c["label"] for c in clips)
            print(f"  {name}: {len(clips)} clips, {len(lbls)} classes, "
                  f"labels {min(lbls)}-{max(lbls)}")

        best_val, test_acc = quick_train(af, tr, va, te, tag, args)
        elapsed = time.perf_counter() - t0
        results[tag] = {"val": best_val, "test": test_acc, "time": elapsed}
        print(f"  Elapsed: {elapsed:.0f}s")

    print(f"\n{'='*70}")
    print(f"  SUMMARY: domain ablation on {args.fiber}")
    print(f"{'='*70}")
    for tag, r in results.items():
        print(f"  {tag:30s}  val={r['val']:5.1f}%  test={r['test']:5.1f}%")
    print()


if __name__ == "__main__":
    main()

"""Stage 1: cache table-fixed ORB features for every frame.

One .npz per (view, desktop) under .cache/tmp/e1.  Reading the ~11 k frames off
F: dominates the runtime, so this is done once and every later pass works from
the cache.
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C          # noqa: E402
import tablemask as T       # noqa: E402

WORK = 640          # long side of the working image
NFEAT = 800
ORIG = {"oak1": 1280, "oak2": 1280, "rs": 1280, "scan": 1600}


def cache_path(view: str, desktop: int) -> str:
    return os.path.join(C.TMP, "feat_%s_%02d.npz" % (view, desktop))


def _one(args):
    view, desktop = args
    out = cache_path(view, desktop)
    if os.path.exists(out):
        return view, desktop, -1
    frames = C.load_frames(view, [desktop])
    steps, offs, kps, dess, nkp, shapes = [], [], [], [], [], []
    off = 0
    for f in frames:
        steps.append(f.step)
        if f.missing or not f.path:
            offs.append(off)
            nkp.append(0)
            shapes.append((0, 0))
            continue
        img = C.imread(f.image_path(), WORK)
        if img is None:
            offs.append(off)
            nkp.append(0)
            shapes.append((0, 0))
            continue
        kp, des, safe, _y = T.detect(img, view, NFEAT)
        n = 0 if des is None else len(kp)
        offs.append(off)
        nkp.append(n)
        shapes.append(img.shape[:2])
        if n:
            kps.append(np.float32([k.pt for k in kp]))
            dess.append(des.astype(np.uint8))
            off += n
    np.savez_compressed(
        out,
        steps=np.asarray(steps, np.int32),
        offs=np.asarray(offs, np.int64),
        nkp=np.asarray(nkp, np.int32),
        shapes=np.asarray(shapes, np.int32),
        kp=np.concatenate(kps) if kps else np.zeros((0, 2), np.float32),
        des=np.concatenate(dess) if dess else np.zeros((0, 32), np.uint8),
    )
    return view, desktop, sum(nkp)


class Store:
    """Random access to the cached features of one (view, desktop)."""

    def __init__(self, view: str, desktop: int):
        z = np.load(cache_path(view, desktop))
        self.steps = z["steps"]
        self.offs = z["offs"]
        self.nkp = z["nkp"]
        self.shapes = z["shapes"]
        self.kp = z["kp"]
        self.des = z["des"]
        self.view = view
        self.desktop = desktop
        self.upscale = ORIG[view] / float(WORK)

    def by_step(self, step: int):
        idx = np.nonzero(self.steps == step)[0]
        if not len(idx):
            return None, None
        return self.at(int(idx[0]))

    def at(self, i: int):
        n = int(self.nkp[i])
        if n == 0:
            return None, None
        o = int(self.offs[i])
        return self.kp[o:o + n], self.des[o:o + n]


def main():
    views = sys.argv[1:] or list(C.VIEWS)
    os.makedirs(C.TMP, exist_ok=True)
    jobs = [(v, d) for v in views for d in range(1, 67)]
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=6) as ex:
        for view, desktop, n in ex.map(_one, jobs):
            done += 1
            if done % 20 == 0 or n == -1:
                print("[%5.1fs] %3d/%3d %s d%02d kp=%s" %
                      (time.time() - t0, done, len(jobs), view, desktop, n), flush=True)
    print("done in %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()

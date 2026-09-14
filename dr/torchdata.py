"""Torch Dataset and DataLoader construction.

Decoding and resizing multi-megapixel fundus JPEGs is the real bottleneck in
this pipeline -- far more than the GPU forward pass -- so the loader supports a
pre-resized cache. Building that cache once turns an EyePACS epoch from
hours of JPEG decoding into minutes.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from . import transforms as T
from .splits import class_balanced_weights


class FundusDataset(Dataset):
    def __init__(self, records, indices=None, size=512, train=False,
                 graham=True, cache_dir=None, seed=0):
        self.records = records
        self.indices = list(range(len(records))) if indices is None else list(indices)
        self.size = size
        self.train = train
        self.graham = graham
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

    def __len__(self):
        return len(self.indices)

    def _cache_path(self, rec):
        # Key on the settings that change the pixels, so switching resolution
        # or normalisation cannot silently reuse the wrong cache.
        key = f"{rec.image_path}|{self.size}|{int(self.graham)}"
        return self.cache_dir / (hashlib.md5(key.encode()).hexdigest() + ".png")

    def _load(self, rec):
        if self.cache_dir:
            cp = self._cache_path(rec)
            if cp.exists():
                img = cv2.imread(str(cp), cv2.IMREAD_COLOR)
                if img is not None:
                    return img
            img = T.load_and_prepare(rec.image_path, self.size, self.graham)
            cv2.imwrite(str(cp), img)
            return img
        return T.load_and_prepare(rec.image_path, self.size, self.graham)

    def __getitem__(self, i):
        rec = self.records[self.indices[i]]
        img = self._load(rec)
        if self.train:
            # Seed per (epoch-agnostic) worker+index so augmentation differs
            # across workers but stays reproducible for a fixed seed.
            rng = np.random.default_rng((self.seed * 1_000_003 + i) % (2**32))
            img = T.augment(img, rng)
        x = torch.from_numpy(T.to_tensor(img))
        return x, torch.tensor(rec.grade, dtype=torch.long), self.indices[i]


def build_loaders(records, train_idx, val_idx, size=512, batch_size=16,
                  workers=2, balanced=True, cache_dir=None, seed=0,
                  balance_power=0.5):
    train_ds = FundusDataset(records, train_idx, size, True, True, cache_dir, seed)
    val_ds = FundusDataset(records, val_idx, size, False, True, cache_dir, seed)

    if balanced:
        w = class_balanced_weights(records, train_idx, power=balance_power)
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                        num_samples=len(train_idx), replacement=True)
        shuffle = False
    else:
        sampler, shuffle = None, True

    common = dict(num_workers=workers, pin_memory=torch.cuda.is_available(),
                  persistent_workers=workers > 0)
    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                          shuffle=shuffle, drop_last=True, **common)
    val_dl = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, **common)
    return train_dl, val_dl


def build_cache(records, size=512, graham=True, cache_dir=None, workers=4):
    """Materialise the resized cache up front.

    Worth running once before training: it removes full-resolution JPEG
    decoding from every epoch, which otherwise dominates wall-clock on EyePACS.
    """
    from concurrent.futures import ThreadPoolExecutor
    ds = FundusDataset(records, size=size, graham=graham, cache_dir=cache_dir)
    done = 0

    def one(i):
        try:
            ds._load(records[i])
            return True
        except Exception as e:
            print(f"  skip {records[i].image_path}: {e}")
            return False

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for ok in ex.map(one, range(len(records))):
            done += bool(ok)
            if done % 500 == 0:
                print(f"  cached {done}/{len(records)}", flush=True)
    return done

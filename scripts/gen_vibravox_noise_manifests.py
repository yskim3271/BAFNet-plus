#!/usr/bin/env python3
"""Re-partition the DNS noise_fullband_16k pool into Vibravox-specific
train/valid/test noise manifests (disjoint, held-out eval noise).

Vibravox dev (2,523) / test (3,064) are larger than TAPS (1,000 / 1,000). The
deterministic valid/test noise indexing in src/data.py needs one held-out noise
file per sample: noise_valid >= dev size, noise_test >= test size * test_augment_numb.
The shared 63,810-file DNS pool is fully partitioned by TAPS (60,810/1,000/2,000),
leaving no room for Vibravox's larger held-out sets, so we re-partition the SAME
pool for Vibravox: disjoint train / valid / test, seeded for reproducibility.
Same noise pool/distribution as TAPS, sized to Vibravox (same held-out design:
noise_valid = dev, noise_test = test * test_augment_numb, all disjoint).

Usage: python scripts/gen_vibravox_noise_manifests.py
"""
import argparse
import glob
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise_dir",
                    default="/home/yskim/workspace/dataset/datasets_fullband/noise_fullband_16k")
    ap.add_argument("--out_dir", default="dataset/vibravox")
    ap.add_argument("--n_valid", type=int, default=2523,
                    help="Vibravox dev size (one held-out noise file per dev utt).")
    ap.add_argument("--n_test", type=int, default=6128,
                    help="Vibravox test size * test_augment_numb (3064 * 2).")
    ap.add_argument("--seed", type=int, default=2039)
    args = ap.parse_args()

    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(args.noise_dir, "*.wav")))
    if not names:
        raise SystemExit(f"No .wav files under {args.noise_dir}")
    if args.n_valid + args.n_test > len(names):
        raise SystemExit(f"pool {len(names)} < valid {args.n_valid} + test {args.n_test}")

    # sort first (glob order is arbitrary) then seeded shuffle -> reproducible partition
    random.Random(args.seed).shuffle(names)
    valid = names[:args.n_valid]
    test = names[args.n_valid:args.n_valid + args.n_test]
    train = names[args.n_valid + args.n_test:]

    os.makedirs(args.out_dir, exist_ok=True)
    for fn, lst in [("noise_train", train), ("noise_dev", valid), ("noise_test", test)]:
        with open(os.path.join(args.out_dir, f"{fn}.txt"), "w") as f:
            f.write("\n".join(lst) + "\n")
        print(f"  {fn}.txt: {len(lst)}")
    assert len(set(valid) & set(test)) == 0
    assert len(set(valid) & set(train)) == 0
    assert len(set(test) & set(train)) == 0
    print(f"  total {len(train) + len(valid) + len(test)} (seed {args.seed}); disjoint train/valid/test")


if __name__ == "__main__":
    main()

"""
unpack_results.py — decompress the raw results into this folder.

Usage (from the code/ folder, on any operating system)
    python unpack_results.py                       # decompress all raw results into code/
    python unpack_results.py --pack a.jsonl b.jsonl  # compress results into ../results/raw/

Unpacking reads ../results/raw/*.jsonl.gz and writes the .jsonl files next to
this script, where the analysis scripts expect them. Existing files are kept.
"""

import argparse
import glob
import gzip
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "results", "raw")


def unpack():
    for src in sorted(glob.glob(os.path.join(RAW, "*.jsonl.gz"))):
        dst = os.path.join(HERE, os.path.basename(src)[:-3])
        if os.path.exists(dst):
            print(f"exists, skipped: {os.path.basename(dst)}")
            continue
        with gzip.open(src, "rb") as fin, open(dst, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        print(f"unpacked: {os.path.basename(dst)} ({os.path.getsize(dst) / 1e6:.1f} MB)")


def pack(paths):
    os.makedirs(RAW, exist_ok=True)
    for src in paths:
        dst = os.path.join(RAW, os.path.basename(src) + ".gz")
        with open(src, "rb") as fin, gzip.open(dst, "wb", compresslevel=9) as fout:
            shutil.copyfileobj(fin, fout)
        print(f"packed: {dst} ({os.path.getsize(dst) / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", nargs="+", metavar="FILE", help="compress these .jsonl files into results/raw/")
    args = ap.parse_args()
    pack(args.pack) if args.pack else unpack()

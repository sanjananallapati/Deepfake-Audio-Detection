"""
Dataset sanity-check helper.

After downloading the Fake-or-Real dataset from Kaggle and placing it under
``data/`` (see the README / the user-action-items PDF), run:

    python scripts/prepare_dataset.py --root data/for-norm

It reports, for each split it finds (training / validation / testing, or
whatever sub-folders exist), how many genuine vs deepfake clips were detected
using the same tolerant folder-name matching the training code uses. This lets
you confirm the layout is understood *before* launching a long training run.

It does not move or modify any files — it only inspects them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a loose script (python scripts/prepare_dataset.py).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import class_distribution, scan_dataset  # noqa: E402
from src.metrics import CLASS_NAMES  # noqa: E402


def _report_dir(path: Path) -> bool:
    try:
        samples = scan_dataset(str(path))
    except Exception as e:  # noqa: BLE001
        print(f"  [skip] {path}  ({e})")
        return False
    dist = class_distribution(samples)
    g = dist.get(0, 0)
    d = dist.get(1, 0)
    print(f"  {path.name:<14} total={len(samples):<6} "
          f"{CLASS_NAMES[0]}={g:<6} {CLASS_NAMES[1]}={d:<6}")
    if g == 0 or d == 0:
        print(f"     ⚠️  one class is empty under {path} — check the folder names")
    return True


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Validate the dataset folder layout.")
    p.add_argument("--root", default="data/for-norm",
                   help="Directory containing split sub-folders (training/…).")
    args = p.parse_args(argv)

    root = Path(args.root)
    if not root.exists():
        print(f"Dataset root not found: {root.resolve()}")
        print("Download the Fake-or-Real dataset from Kaggle and extract it here.")
        sys.exit(1)

    print(f"Scanning splits under: {root.resolve()}\n")
    subdirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not subdirs:
        # No split sub-folders — treat the root itself as one dataset.
        _report_dir(root)
    else:
        any_ok = False
        for sub in subdirs:
            any_ok = _report_dir(sub) or any_ok
        if not any_ok:
            print("\nNo labelled audio detected. Expected class sub-folders such as "
                  "real/ and fake/ (aliases: genuine/spoof/bonafide/…).")
            sys.exit(1)

    print("\nLabel convention: 0 = Genuine (Human), 1 = Deepfake (AI-Generated).")
    print("If the counts look right, you're ready to train:  python -m src.train")


if __name__ == "__main__":
    main()

"""Download / verify datasets from the command line.

Run this on a **login node** (compute nodes often have no internet access):

    python datasets/download.py --dataset mnist --root ./data
    python datasets/download.py --dataset thfood100 --root ./data/thfood100

For MNIST this downloads the data; for THFOOD-100 (which must be obtained
separately) it verifies the expected directory layout and prints per-split
statistics.
"""

import argparse
from pathlib import Path

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def download_mnist(root: str) -> None:
    """Download the MNIST train and test sets into ``root``.

    Args:
        root: Directory the dataset is stored in (created if missing).
    """
    from torchvision import datasets  # imported here so --help stays fast

    print(f"Downloading MNIST into {root} ...")
    datasets.MNIST(root, train=True, download=True)
    datasets.MNIST(root, train=False, download=True)
    print("MNIST ready: 60,000 train / 10,000 test images.")


def verify_thfood(root: str) -> None:
    """Verify the THFOOD-100 ImageFolder layout and print statistics.

    Args:
        root: Directory expected to contain train/ val/ [test/] splits, or a
            flat layout of one sub-directory per class (e.g. a data preview).
    """
    root_path = Path(root)
    if not root_path.is_dir():
        print(f"Directory not found: {root_path}\n")
        print("THFOOD-100 is not downloaded automatically. Obtain it from your")
        print("instructor (or the shared project directory on LANTA) and arrange")
        print("it like this:\n")
        print(f"    {root_path}/train/<class_name>/*.jpg")
        print(f"    {root_path}/val/<class_name>/*.jpg")
        print(f"    {root_path}/test/<class_name>/*.jpg   (optional)")
        return

    if not (root_path / "train").is_dir():
        class_dirs = [d for d in root_path.iterdir() if d.is_dir()]
        if not class_dirs:
            print(f"No train/ split and no class sub-directories under {root_path}.\n")
            print(f"    {root_path}/train/<class_name>/*.jpg")
            print(f"    {root_path}/val/<class_name>/*.jpg")
            print(f"    {root_path}/test/<class_name>/*.jpg   (optional)")
            print(f"or a flat layout: {root_path}/<class_name>/*.jpg")
            return

        n_images = sum(
            1
            for d in class_dirs
            for f in d.iterdir()
            if f.suffix.lower() in _IMAGE_EXTENSIONS
        )
        print(f"Checking THFOOD-100 layout under {root_path} ...\n")
        print(f"{'classes':>10}{'images':>12}")
        print(f"{len(class_dirs):>10}{n_images:>12}")
        print(
            "\nNo train/val/test folders found — this looks like a flat, "
            "unsplit layout (e.g. a sample/preview). datasets/thfood100.py "
            "will derive a per-class train/val/test split automatically."
        )
        return

    print(f"Checking THFOOD-100 layout under {root_path} ...\n")
    print(f"{'split':<8}{'classes':>10}{'images':>12}")
    print("-" * 30)
    for split in ("train", "val", "test"):
        split_dir = root_path / split
        if not split_dir.is_dir():
            marker = "(optional)" if split == "test" else "MISSING!"
            print(f"{split:<8}{'-':>10}{'-':>12}  {marker}")
            continue
        class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        n_images = sum(
            1
            for d in class_dirs
            for f in d.iterdir()
            if f.suffix.lower() in _IMAGE_EXTENSIONS
        )
        print(f"{split:<8}{len(class_dirs):>10}{n_images:>12}")
    print("\nIf train and val are present, you are ready for Lab 2.")


def main() -> None:
    """Entry point for the download/verify CLI."""
    parser = argparse.ArgumentParser(description="Download or verify a dataset.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["mnist", "thfood100"],
        help="Which dataset to prepare.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Dataset directory (default: ./data for mnist, ./data/thfood100 for thfood100).",
    )
    args = parser.parse_args()

    if args.dataset == "mnist":
        download_mnist(args.root or "./data")
    else:
        verify_thfood(args.root or "./data/thfood100")


if __name__ == "__main__":
    main()

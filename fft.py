import os
import shutil
import ssl
import tempfile
from glob import glob

ssl._create_default_https_context = ssl._create_unverified_context

# --- [SciPy sqrtm 호환성 및 언패킹 패치] ---
import scipy.linalg

_orig_sqrtm = scipy.linalg.sqrtm


def _patched_sqrtm(A, *args, **kwargs):
    kwargs.pop("disp", None)
    kwargs.pop("blocksize", None)
    res = _orig_sqrtm(A, *args, **kwargs)

    if isinstance(res, tuple) or isinstance(res, list):
        return res
    return res, 0.0


scipy.linalg.sqrtm = _patched_sqrtm
# ------------------------------------------

import pytorch_fid.fid_score as fid_core
import torch
from colorama import Fore, Style
from dotenv import load_dotenv
from PIL import Image
from pytorch_fid.fid_score import calculate_fid_given_paths

load_dotenv()

MIN_SAMPLES = 10

# 13개 표준 카테고리 목록
STANDARD_CATEGORIES = [
    "Airplanes",
    "Bicycles",
    "Birds",
    "Boats",
    "Cars",
    "Cats",
    "Computers",
    "Dogs",
    "Livestocks",
    "Living Rooms",
    "Motorcycles",
    "People",
    "Plants",
    "Trains",
]


class SafeFIDImageDataset(torch.utils.data.Dataset):

    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        try:
            img = Image.open(path).convert("RGB")
            if img.size != (1024, 1024):
                img = img.resize((1024, 1024), Image.Resampling.BILINEAR)
        except Exception:
            img = Image.new("RGB", (1024, 1024), (0, 0, 0))
        if self.transforms is not None:
            img = self.transforms(img)
        return img


fid_core.ImageFolderDataset = SafeFIDImageDataset


def load_category_files(gen_dir: str) -> dict[str, list[str]]:
    images_dir = (
        os.path.join(gen_dir, "images")
        if os.path.exists(os.path.join(gen_dir, "images"))
        else gen_dir
    )

    all_categories = STANDARD_CATEGORIES + ["Others"]
    category_files: dict[str, list[str]] = {cat: [] for cat in all_categories}

    sub_dirs = [
        d for d in os.listdir(images_dir)
        if os.path.isdir(os.path.join(images_dir, d))
    ]

    if sub_dirs:
        for cat_dir in sub_dirs:
            category = cat_dir if cat_dir in STANDARD_CATEGORIES else "Others"
            cat_path = os.path.join(images_dir, cat_dir)

            img_paths = (
                    glob(os.path.join(cat_path, "*.png"))
                    + glob(os.path.join(cat_path, "*.jpg"))
                    + glob(os.path.join(cat_path, "*.jpeg"))
            )
            category_files[category].extend(img_paths)
    else:
        img_paths = (
                glob(os.path.join(images_dir, "*.png"))
                + glob(os.path.join(images_dir, "*.jpg"))
                + glob(os.path.join(images_dir, "*.jpeg"))
        )
        category_files["Others"].extend(img_paths)

    return category_files


def evaluate_category_fid(
        real_files: list[str],
        fake_files: list[str],
        category: str,
        gen: int,
        device: str = "cuda",
) -> float | None:
    if len(real_files) < MIN_SAMPLES or len(fake_files) < MIN_SAMPLES:
        print(
            f"{Fore.BLUE}{'[FID-C]':<9}{Fore.CYAN}Gen {Fore.MAGENTA}{gen}{Fore.WHITE} "
            f"[{category}]{Fore.RED}: Skipped < {MIN_SAMPLES} images "
            f"(real={len(real_files)}, fake={len(fake_files)}){Style.RESET_ALL}"
        )
        return None

    with (
        tempfile.TemporaryDirectory() as tmp_real,
        tempfile.TemporaryDirectory() as tmp_fake,
    ):
        for p in real_files:
            shutil.copy(p, os.path.join(tmp_real, os.path.basename(p)))
        for p in fake_files:
            shutil.copy(p, os.path.join(tmp_fake, os.path.basename(p)))

        try:
            fid_value = calculate_fid_given_paths(
                [tmp_real, tmp_fake], batch_size=32, device=device, dims=2048
            )
            print(
                f"{Fore.BLUE}{'[FID-C]':<9}{Fore.CYAN}Gen {Fore.MAGENTA}{gen}{Fore.WHITE} "
                f"[{category}] (R:{len(real_files)}, F:{len(fake_files)}): {Fore.GREEN}{fid_value:.4f}{Style.RESET_ALL}"
            )
            return fid_value
        except Exception as e:
            print(
                f"{Fore.BLUE}{'[FID-C]':<9}{Fore.CYAN}Gen {Fore.MAGENTA}{gen}{Fore.WHITE} "
                f"[{category}]{Fore.RED}: Error ({e}){Style.RESET_ALL}"
            )
            return None


if __name__ == "__main__":
    generations = int(os.environ.get("GENERATIONS", 20))
    data_root = "./fft_data"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_categories = STANDARD_CATEGORIES + ["Others"]

    gen0_dir = os.path.join(data_root, "gen_0")
    if not os.path.exists(gen0_dir) and os.path.exists("./dataset"):
        gen0_dir = "./dataset"

    gen0_files = load_category_files(gen0_dir)
    print(
        f"{Fore.YELLOW}[SYSTEM] Loading Gen 0 baseline from '{gen0_dir}'...{Style.RESET_ALL}"
    )
    for cat, files in gen0_files.items():
        print(
            f"{Fore.WHITE}[gen_0] {cat}: {len(files)} images{Style.RESET_ALL}"
        )

    results: dict[str, dict[int, float]] = {cat: {} for cat in all_categories}
    gen_file_counts: dict[int, dict[str, int]] = {}

    for gen in range(1, generations):
        gen_dir = os.path.join(data_root, f"gen_{gen}")
        if not os.path.exists(gen_dir):
            continue

        print(
            f"\n{Fore.BLUE}{'[FID-C]':<9}{Fore.CYAN}Generation {Fore.MAGENTA}{gen}{Fore.WHITE}: Category FID Start{Style.RESET_ALL}"
        )
        gen_files = load_category_files(gen_dir)
        gen_file_counts[gen] = {
            cat: len(files) for cat, files in gen_files.items()
        }

        for category in all_categories:
            score = evaluate_category_fid(
                real_files=gen0_files[category],
                fake_files=gen_files[category],
                category=category,
                gen=gen,
                device=device,
            )
            if score is not None:
                results[category][gen] = score

    print(f"\n{'=' * 115}")
    print(
        f"=== Category FID Evaluation Summary (vs gen_0) [Format: FID (fake_count)] ==="
    )
    print(f"{'=' * 115}")

    header = f"{'Category':<22}" + "".join(
        f"  Gen{g:<9}" for g in range(1, generations)
    )
    print(header)
    print("-" * len(header))

    for cat, gen_scores in results.items():
        row = f"{cat:<22}"
        for gen in range(1, generations):
            val = gen_scores.get(gen)
            cnt = gen_file_counts.get(gen, {}).get(cat, 0)
            if val is not None:
                row += f"  {val:>5.1f}({cnt:<4})"
            else:
                row += f"  {'N/A':>5}({cnt:<4})"
        print(row)

    weighted_with_others: dict[int, float] = {}
    weighted_without_others: dict[int, float] = {}

    for gen in range(1, generations):
        counts = gen_file_counts.get(gen, {})

        total_count_inc = sum(
            counts.get(cat, 0)
            for cat in all_categories
            if gen in results[cat]
        )
        if total_count_inc > 0:
            weighted_sum_inc = sum(
                counts.get(cat, 0) * results[cat][gen]
                for cat in all_categories
                if gen in results[cat]
            )
            weighted_with_others[gen] = weighted_sum_inc / total_count_inc

        total_count_exc = sum(
            counts.get(cat, 0)
            for cat in STANDARD_CATEGORIES
            if gen in results[cat]
        )
        if total_count_exc > 0:
            weighted_sum_exc = sum(
                counts.get(cat, 0) * results[cat][gen]
                for cat in STANDARD_CATEGORIES
                if gen in results[cat]
            )
            weighted_without_others[gen] = weighted_sum_exc / total_count_exc

    print(f"\n{'=' * 115}")
    print("=== Aggregated Weighted Average FID Summary ===")
    print(f"{'=' * 115}")

    row_inc = f"{'With Others':<22}"
    row_exc = f"{'Without Others':<22}"

    for gen in range(1, generations):
        v_inc = weighted_with_others.get(gen)
        v_exc = weighted_without_others.get(gen)
        row_inc += f"  {v_inc:>12.2f}" if v_inc is not None else f"  {'N/A':>12}"
        row_exc += f"  {v_exc:>12.2f}" if v_exc is not None else f"  {'N/A':>12}"

    print(header)
    print("-" * len(header))
    print(f"{Fore.YELLOW}{row_inc}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{row_exc}{Style.RESET_ALL}")

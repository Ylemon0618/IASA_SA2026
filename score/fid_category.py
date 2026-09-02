import json
import os
import shutil
import ssl
import tempfile
from glob import glob

ssl._create_default_https_context = ssl._create_unverified_context

import scipy.linalg

_orig_sqrtm = scipy.linalg.sqrtm


def _patched_sqrtm(A, *args, **kwargs):
    kwargs.pop("disp", None)
    kwargs.pop("blocksize", None)
    return _orig_sqrtm(A, *args, **kwargs)


scipy.linalg.sqrtm = _patched_sqrtm

import pytorch_fid.fid_score as fid_core
import torch
from colorama import Fore, Style
from dotenv import load_dotenv
from PIL import Image
from pytorch_fid.fid_score import calculate_fid_given_paths

load_dotenv()

CATEGORIES = {
    "Airplanes": [
        "airplane",
        "aircraft",
        "hangar",
        "runway",
        "airfield",
        "tarmac",
        "baggage carts",
        "fueling truck",
        "runway lights",
        "clouds",
        "sky",
        "flight",
        "plane",
        "jet",
        "aviation",
        "propeller",
        "cockpit",
        "wing",
        "landing gear",
        "airstrip",
    ],
    "Cars": [
        "car",
        "minivan",
        "race car",
        "sports car",
        "driveway",
        "parking garage",
        "avenue",
        "skyscrapers",
        "traffic light",
        "pedestrians",
        "sedan",
        "automobile",
        "coupe",
        "suv",
        "headlight",
        "windshield",
        "bumper",
        "trunk",
        "parking lot",
        "vehicle",
    ],
    "Cats": [
        "cat",
        "kitten",
        "feline",
        "tabby",
        "scratching post",
        "cat toy",
        "feathered",
        "kibble",
        "food bowl",
        "laundry",
        "windowsill",
        "whiskers",
        "purr",
        "meow",
        "paws",
        "litter box",
        "ginger cat",
        "catnip",
        "furry",
        "claw",
    ],
    "Dogs": [
        "dog",
        "puppy",
        "canine",
        "doghouse",
        "leash",
        "tennis ball",
        "water bowl",
        "porch",
        "autumn forest",
        "barking",
        "tail",
        "snout",
        "golden retriever",
        "bulldog",
        "poodle",
        "hound",
        "terrier",
        "fetch",
        "fur",
        "doggy",
    ],
    "Motorcycles": [
        "motorcycle",
        "motorbike",
        "riding gloves",
        "helmet",
        "biker",
        "alleyway",
        "city intersection",
        "racing track",
        "finish line",
        "toolbox",
        "handlebars",
        "exhaust",
        "chopper",
        "scooter",
        "moped",
        "kickstand",
        "visor",
        "cruiser",
        "engine bay",
        "two-wheeler",
    ],
    "Trains": [
        "train",
        "subway platform",
        "railway station",
        "canyon bridge",
        "forest track",
        "coastal rail",
        "signal tower",
        "overhead power cables",
        "schedule board",
        "cargo platform",
        "passengers",
        "locomotive",
        "railroad",
        "boxcar",
        "track",
        "caboose",
        "commuter",
        "express train",
        "rail",
        "depot",
    ],
    "Boats": [
        "boat",
        "ship",
        "marina dock",
        "ocean harbor",
        "tropical river",
        "mountain lake",
        "morning bay",
        "swimming pool deck",
        "fishing rods",
        "life jackets",
        "seagulls",
        "wooden dock",
        "vessel",
        "deck",
        "stern",
        "canoe",
        "sailboat",
        "yacht",
        "kayak",
        "oar",
    ],
    "Bicycles": [
        "bicycle",
        "bike",
        "bike rack",
        "beach boardwalk",
        "cobblestone street",
        "park pathway",
        "mountain trail",
        "water bottle mounted",
        "canvas backpack",
        "streetlamps",
        "cyclist",
        "pedal",
        "frame",
        "chain",
        "cycling",
        "spokes",
        "saddle",
        "wheel",
        "kickstand bike",
        "handlebar basket",
    ],
    "Living Rooms": [
        "sofa",
        "couch",
        "flatscreen tv",
        "coffee table",
        "bookshelf",
        "floor lamp",
        "rustic cabin",
        "apartment corner",
        "open-plan house",
        "studio",
        "home theater",
        "interior",
        "living room",
        "cushion",
        "armchair",
        "fireplace",
        "rug",
        "curtains",
        "living space",
        "lounge",
    ],
    "Computers": [
        "computer",
        "pc",
        "monitor",
        "workstation",
        "gaming setup",
        "office desk",
        "co-working space",
        "study table",
        "mechanical keyboard",
        "rgb lights",
        "wireless mouse",
        "headphones",
        "mug of coffee",
        "notebooks and pens",
        "desktop",
        "laptop",
        "screen",
        "mousepad",
        "processor",
        "cpu",
    ],
}


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


def classify_image(caption: str) -> str:
    caption_lower = caption.lower()
    for category, keywords in CATEGORIES.items():
        if any(kw in caption_lower for kw in keywords):
            return category
    return "Others"


def load_category_files(gen_dir: str) -> dict[str, list[str]]:
    images_dir = (
        os.path.join(gen_dir, "images")
        if os.path.exists(os.path.join(gen_dir, "images"))
        else gen_dir
    )
    metadata_path = os.path.join(images_dir, "metadata.jsonl")

    all_categories = list(CATEGORIES.keys()) + ["Others"]
    category_files: dict[str, list[str]] = {cat: [] for cat in all_categories}

    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line.strip())
                img_path = os.path.join(images_dir, data["file_name"])
                if not os.path.exists(img_path):
                    continue
                category = classify_image(data.get("text", ""))
                category_files[category].append(img_path)
        return category_files

    img_paths = glob(os.path.join(images_dir, "*.png")) + glob(
        os.path.join(images_dir, "*.jpg")
    )
    for img_p in img_paths:
        txt_p = os.path.splitext(img_p)[0] + ".txt"
        caption = ""
        if os.path.exists(txt_p):
            with open(txt_p, "r", encoding="utf-8") as f:
                caption = f.read().strip()
        category = classify_image(caption)
        category_files[category].append(img_p)

    return category_files


def evaluate_category_fid(
        real_files: list[str],
        fake_files: list[str],
        category: str,
        gen: int,
        device: str = "cuda",
) -> float | None:
    if len(real_files) < 2 or len(fake_files) < 2:
        print(
            f"{Fore.BLUE}{'[FID-C]':<9}{Fore.CYAN}Gen {Fore.MAGENTA}{gen}{Fore.WHITE} "
            f"[{category}]{Fore.RED}: Not enough images "
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
                f"[{category}]: {Fore.GREEN}{fid_value:.4f}{Style.RESET_ALL}"
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

    all_categories = list(CATEGORIES.keys()) + ["Others"]

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

    print(f"\n{'=' * 85}")
    print("=== Category FID Evaluation Summary (vs gen_0) ===")
    print(f"{'=' * 85}")
    header = f"{'Category':<22}" + "".join(
        f"  Gen{g:<4}" for g in range(1, generations)
    )
    print(header)
    print("-" * len(header))
    for cat, gen_scores in results.items():
        row = f"{cat:<22}"
        for gen in range(1, generations):
            val = gen_scores.get(gen)
            row += f"  {val:>7.2f}" if val is not None else f"  {'N/A':>7}"
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
            for cat in CATEGORIES.keys()
            if gen in results[cat]
        )
        if total_count_exc > 0:
            weighted_sum_exc = sum(
                counts.get(cat, 0) * results[cat][gen]
                for cat in CATEGORIES.keys()
                if gen in results[cat]
            )
            weighted_without_others[gen] = weighted_sum_exc / total_count_exc

    print(f"\n{'=' * 85}")
    print("=== Aggregated Weighted Average FID Summary ===")
    print(f"{'=' * 85}")

    row_inc = f"{'With Others':<22}"
    row_exc = f"{'Without Others':<22}"

    for gen in range(1, generations):
        v_inc = weighted_with_others.get(gen)
        v_exc = weighted_without_others.get(gen)
        row_inc += f"  {v_inc:>7.2f}" if v_inc is not None else f"  {'N/A':>7}"
        row_exc += f"  {v_exc:>7.2f}" if v_exc is not None else f"  {'N/A':>7}"

    print(header)
    print("-" * len(header))
    print(f"{Fore.YELLOW}{row_inc}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{row_exc}{Style.RESET_ALL}")

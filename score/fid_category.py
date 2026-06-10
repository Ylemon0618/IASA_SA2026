import json
import os
import shutil
import tempfile

import pytorch_fid.fid_score as fid_core
import torch
from PIL import Image
from colorama import Fore, Style
from dotenv import load_dotenv
from pytorch_fid.fid_score import calculate_fid_given_paths

load_dotenv()

# 카테고리별 키워드 정의
CATEGORIES = {
    "Boat_Ship": [
        "boat", "ship", "fishing boat", "cargo ship", "sailboat", "canoe",
        "rowing", "vessel", "sailing", "anchor", "shore", "sea", "river", "water"
    ],
    "Bottle_Tabletop": [
        "bottle", "wine", "beer", "soda", "juice", "drink", "glass",
        "refrigerator", "kitchen", "pour", "tabletop", "counter", "dinner table"
    ],
    "Bus_Transport": [
        "bus", "double decker", "school bus", "tour bus", "trolley", "tram",
        "transit", "station"
    ],
    "Car": [
        "car", "sports car", "minivan", "race car", "truck", "vehicle",
        "driving", "parked", "drifting", "highway", "road", "garage"
    ],
    "Cat": [
        "cat", "kitten", "tabby", "feline", "ginger cat"
    ],
    "Chair_Living": [
        "chair", "rocking chair", "office chair", "folding chair", "sofa",
        "couch", "dining", "furniture", "living room", "desk", "sitting"
    ],
    "Cow": [
        "cow", "calf", "bull", "cattle", "livestock", "grazing", "barn",
        "pasture", "farm", "straw"
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
            img = Image.open(path).convert('RGB')
            if img.size != (1024, 1024):
                img = img.resize((1024, 1024), Image.Resampling.BILINEAR)
        except Exception:
            img = Image.new('RGB', (1024, 1024), (0, 0, 0))
        if self.transforms is not None:
            img = self.transforms(img)
        return img


fid_core.ImageFolderDataset = SafeFIDImageDataset


def classify_image(caption: str) -> str | None:
    """캡션 키워드 매칭으로 카테고리 반환. 매칭 없으면 None."""
    caption_lower = caption.lower()
    for category, keywords in CATEGORIES.items():
        if any(kw in caption_lower for kw in keywords):
            return category
    return None


def load_category_files(gen_dir: str) -> dict[str, list[str]]:
    """metadata.jsonl을 읽어 카테고리별 이미지 경로 목록 반환."""
    images_dir = os.path.join(gen_dir, "images")
    metadata_path = os.path.join(images_dir, "metadata.jsonl")

    category_files: dict[str, list[str]] = {cat: [] for cat in CATEGORIES}

    if not os.path.exists(metadata_path):
        return category_files

    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            img_path = os.path.join(images_dir, data["file_name"])
            if not os.path.exists(img_path):
                continue
            category = classify_image(data["text"])
            if category:
                category_files[category].append(img_path)

    return category_files


def evaluate_category_fid(real_files: list[str], fake_files: list[str],
                          category: str, gen: int, device: str = "cuda") -> float | None:
    if len(real_files) < 2 or len(fake_files) < 2:
        print(
            f"{Fore.BLUE}{'[FID-C]':<9}{Fore.CYAN}Gen {Fore.MAGENTA}{gen}{Fore.WHITE} "
            f"[{category}]{Fore.RED}: Not enough images "
            f"(real={len(real_files)}, fake={len(fake_files)}){Style.RESET_ALL}")
        return None

    # 임시 폴더에 심볼릭 링크 없이 경로만 넘기기 위해 파일 목록을 txt로 전달하는
    # pytorch_fid의 내부 API를 우회 → 임시 디렉토리에 복사
    with tempfile.TemporaryDirectory() as tmp_real, tempfile.TemporaryDirectory() as tmp_fake:
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
                f"[{category}]: {Fore.GREEN}{fid_value:.4f}{Style.RESET_ALL}")
            return fid_value
        except Exception as e:
            print(
                f"{Fore.BLUE}{'[FID-C]':<9}{Fore.CYAN}Gen {Fore.MAGENTA}{gen}{Fore.WHITE} "
                f"[{category}]{Fore.RED}: Error ({e}){Style.RESET_ALL}")
            return None


if __name__ == "__main__":
    generations = int(os.environ.get("GENERATIONS", 20))
    data_root = "./fft_data"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # gen_0 기준 카테고리별 파일 목록 로드
    gen0_files = load_category_files(os.path.join(data_root, "gen_0"))
    for cat, files in gen0_files.items():
        print(f"{Fore.WHITE}[gen_0] {cat}: {len(files)} images{Style.RESET_ALL}")

    # 결과: results[category][gen] = fid_value
    results: dict[str, dict[int, float]] = {cat: {} for cat in CATEGORIES}

    for gen in range(1, generations):
        gen_dir = os.path.join(data_root, f"gen_{gen}")
        if not os.path.exists(gen_dir):
            continue

        print(
            f"\n{Fore.BLUE}{'[FID-C]':<9}{Fore.CYAN}Generation {Fore.MAGENTA}{gen}{Fore.WHITE}: Category FID Start{Style.RESET_ALL}")
        gen_files = load_category_files(gen_dir)

        for category in CATEGORIES:
            score = evaluate_category_fid(
                real_files=gen0_files[category],
                fake_files=gen_files[category],
                category=category,
                gen=gen,
                device=device
            )
            if score is not None:
                results[category][gen] = score

    # 요약 출력
    print(f"\n{'=' * 60}")
    print("=== Category FID Evaluation Summary (vs gen_0) ===")
    print(f"{'=' * 60}")
    header = f"{'Category':<22}" + "".join(f"  Gen{g:<4}" for g in range(1, generations))
    print(header)
    print("-" * len(header))
    for cat, gen_scores in results.items():
        row = f"{cat:<22}"
        for gen in range(1, generations):
            val = gen_scores.get(gen)
            row += f"  {val:>7.2f}" if val is not None else f"  {'N/A':>7}"
        print(row)

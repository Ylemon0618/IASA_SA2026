import glob
import os
import shutil

import numpy as np
from PIL import Image
from colorama import Fore, Style
from dotenv import load_dotenv

load_dotenv()


def add_gaussian_noise(image_path, std_dev):
    with Image.open(image_path).convert("RGB") as img:
        img_array = np.array(img, dtype=np.float32)

        noise = np.random.normal(0, std_dev, img_array.shape)

        noisy_img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

        return Image.fromarray(noisy_img_array)


def inject_gaussian_noise_to_dataset(source_dir, target_root, std_dev):
    folder_name = os.path.basename(os.path.normpath(source_dir))
    corrupted_dir = os.path.join(
        target_root, f"gaussian_std_{std_dev}", folder_name
    )

    if os.path.exists(corrupted_dir):
        print(
            f"{Fore.YELLOW}[SYSTEM] Existing corrupted directory found. Overwriting: '{corrupted_dir}'{Style.RESET_ALL}"
        )
        shutil.rmtree(corrupted_dir)

    print(
        f"{Fore.CYAN}[SYSTEM] Cloning dataset & adding Gaussian Noise (std={std_dev})...{Style.RESET_ALL}"
    )
    shutil.copytree(source_dir, corrupted_dir)

    img_extensions = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    img_paths = []
    for ext in img_extensions:
        img_paths.extend(
            glob.glob(
                os.path.join(corrupted_dir, "**", ext), recursive=True
            )
        )
    img_paths = sorted(img_paths)

    if not img_paths:
        print(
            f"{Fore.RED}[ERROR] No images found in '{corrupted_dir}'.{Style.RESET_ALL}"
        )
        return corrupted_dir

    print(
        f"{Fore.GREEN}[INFO] Total images to process: {len(img_paths)}{Style.RESET_ALL}"
    )

    for img_p in img_paths:
        noisy_img = add_gaussian_noise(img_p, std_dev)
        noisy_img.save(img_p)

    print(
        f"{Fore.GREEN}[SUCCESS] Gaussian Noise injection complete! Output path: {corrupted_dir}{Style.RESET_ALL}\n"
    )
    return corrupted_dir


if __name__ == "__main__":
    dataset_path = os.environ.get("DATASET_PATH", "./dataset")
    SRC_DIR = f"{dataset_path}"
    TARGET_ROOT = "./dataset_corrupted"

    std_input = input(
        "Enter Gaussian Noise standard deviation (std) (e.g., 5, 15, 25): "
    ).strip()
    try:
        std_dev = float(std_input)
        if std_dev >= 0:
            inject_gaussian_noise_to_dataset(SRC_DIR, TARGET_ROOT, std_dev)
        else:
            print(
                f"{Fore.RED}[ERROR] Standard deviation must be non-negative.{Style.RESET_ALL}"
            )
    except ValueError as e:
        print(
            f"{Fore.RED}[ERROR] Please enter a valid numerical value. Details: {e}{Style.RESET_ALL}"
        )

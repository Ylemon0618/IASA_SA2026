import glob
import math
import os
import random
import shutil

from colorama import Fore, Style
from dotenv import load_dotenv

load_dotenv()


def swap_txt_contents(file1_path, file2_path):
    with open(file1_path, "r", encoding="utf-8") as f1:
        content1 = f1.read().strip()
    with open(file2_path, "r", encoding="utf-8") as f2:
        content2 = f2.read().strip()

    with open(file1_path, "w", encoding="utf-8") as f1:
        f1.write(content2)
    with open(file2_path, "w", encoding="utf-8") as f2:
        f1.write(content1)


def inject_label_noise(source_dir, target_root, error_rate_percent):
    folder_name = os.path.basename(os.path.normpath(source_dir))
    corrupted_dir = os.path.join(
        target_root, f"error_{error_rate_percent}pct", folder_name
    )

    if os.path.exists(corrupted_dir):
        print(
            f"{Fore.YELLOW}[SYSTEM] Existing directory found. Deleting and re-cloning.{Style.RESET_ALL}"
        )
        shutil.rmtree(corrupted_dir)

    print(
        f"{Fore.CYAN}[SYSTEM] Cloning directory... {Fore.WHITE}({source_dir} -> {corrupted_dir}){Style.RESET_ALL}"
    )
    shutil.copytree(source_dir, corrupted_dir)

    txt_files = sorted(
        glob.glob(os.path.join(corrupted_dir, "**", "*.txt"), recursive=True)
    )
    total_files = len(txt_files)

    if total_files < 2:
        print(
            f"{Fore.RED}[ERROR] Insufficient text files to perform a swap. (Total files: {total_files}){Style.RESET_ALL}"
        )
        return corrupted_dir

    target_corrupted_count = int(
        math.floor(total_files * (error_rate_percent / 100.0))
    )
    pair_count = target_corrupted_count // 2
    actual_corrupted_count = pair_count * 2

    print(
        f"{Fore.GREEN}[INFO] Total Files: {total_files} | Target Error Rate: {error_rate_percent}%"
    )
    print(
        f"[INFO] File pairs to swap: {pair_count} pairs ({actual_corrupted_count} total files corrupted){Style.RESET_ALL}"
    )

    if pair_count == 0:
        print(
            f"{Fore.YELLOW}[WARN] Error rate too low or file count too small. No files swapped.{Style.RESET_ALL}"
        )
        return corrupted_dir

    chosen_indices = random.sample(range(total_files), actual_corrupted_count)

    for i in range(0, len(chosen_indices), 2):
        idx1 = chosen_indices[i]
        idx2 = chosen_indices[i + 1]

        file1_path = txt_files[idx1]
        file2_path = txt_files[idx2]

        # 안전한 캡션 스왑 처리
        swap_txt_contents(file1_path, file2_path)

        rel_p1 = os.path.relpath(file1_path, corrupted_dir)
        rel_p2 = os.path.relpath(file2_path, corrupted_dir)
        print(
            f"  {Fore.MAGENTA}↳ [SWAP]{Fore.WHITE} {rel_p1} ⇄ {rel_p2}"
        )

    print(
        f"{Fore.GREEN}[SUCCESS] Label noise injection complete! Output path: {corrupted_dir}{Style.RESET_ALL}\n"
    )
    return corrupted_dir


if __name__ == "__main__":
    dataset_path = os.environ.get("DATASET_PATH", "./dataset")
    SRC_DIR = f"{dataset_path}"
    TARGET_ROOT = "./dataset_corrupted"

    try:
        error_input = input("Enter target error rate percentage: ")
        error_rate = float(error_input)

        if 0 <= error_rate <= 100:
            new_data_path = inject_label_noise(
                SRC_DIR, TARGET_ROOT, error_rate
            )
        else:
            print(
                f"{Fore.RED}[ERROR] Error rate must be between 0 and 100.{Style.RESET_ALL}"
            )
    except ValueError:
        print(
            f"{Fore.RED}[ERROR] Please enter a valid numerical value.{Style.RESET_ALL}"
        )

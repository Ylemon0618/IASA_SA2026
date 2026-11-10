import glob
import json
import math
import os
import random
import shutil

from colorama import Fore, Style
from dotenv import load_dotenv

load_dotenv()


def inject_label_noise_metadata(corrupted_dir, error_rate_percent):
    metadata_path = os.path.join(corrupted_dir, "metadata.jsonl")
    if not os.path.exists(metadata_path):
        return False

    with open(metadata_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    entries = [json.loads(line) for line in lines]
    total_files = len(entries)

    if total_files < 2:
        return True

    target_corrupted_count = int(math.floor(total_files * (error_rate_percent / 100.0)))
    pair_count = target_corrupted_count // 2
    actual_corrupted_count = pair_count * 2

    print(
        f"{Fore.GREEN}[INFO] (metadata.jsonl found) Total Records: {total_files} | Target Error Rate: {error_rate_percent}%")
    print(
        f"[INFO] Entries to swap: {pair_count} pairs ({actual_corrupted_count} total entries corrupted){Style.RESET_ALL}")

    if pair_count == 0:
        return True

    chosen_indices = random.sample(range(total_files), actual_corrupted_count)

    for i in range(0, len(chosen_indices), 2):
        idx1 = chosen_indices[i]
        idx2 = chosen_indices[i + 1]

        entries[idx1]["text"], entries[idx2]["text"] = entries[idx2]["text"], entries[idx1]["text"]

        print(f"  {Fore.MAGENTA}↳ [JSONL SWAP]{Fore.WHITE} {entries[idx1]['file_name']} ⇄ {entries[idx2]['file_name']}")

    with open(metadata_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return True


def swap_txt_contents(file1_path, file2_path):
    with open(file1_path, "r", encoding="utf-8") as f1:
        content1 = f1.read().strip()
    with open(file2_path, "r", encoding="utf-8") as f2:
        content2 = f2.read().strip()

    with open(file1_path, "w", encoding="utf-8") as f1:
        f1.write(content2)
    with open(file2_path, "w", encoding="utf-8") as f2:
        f2.write(content1)


def inject_label_noise(source_dir, target_root, error_rate_percent):
    folder_name = os.path.basename(os.path.normpath(source_dir))
    corrupted_dir = os.path.join(
        target_root, f"error_{error_rate_percent}pct", folder_name
    )

    if os.path.exists(corrupted_dir):
        print(f"{Fore.YELLOW}[SYSTEM] Existing directory found. Deleting and re-cloning.{Style.RESET_ALL}")
        shutil.rmtree(corrupted_dir)

    print(f"{Fore.CYAN}[SYSTEM] Cloning directory... {Fore.WHITE}({source_dir} -> {corrupted_dir}){Style.RESET_ALL}")
    shutil.copytree(source_dir, corrupted_dir)

    if inject_label_noise_metadata(corrupted_dir, error_rate_percent):
        print(
            f"{Fore.GREEN}[SUCCESS] Label noise injection via metadata.jsonl complete! Output path: {corrupted_dir}{Style.RESET_ALL}\n")
        return corrupted_dir

    txt_files = sorted(glob.glob(os.path.join(corrupted_dir, "**", "*.txt"), recursive=True))
    total_files = len(txt_files)

    if total_files < 2:
        print(f"{Fore.RED}[ERROR] Insufficient text files. (Total files: {total_files}){Style.RESET_ALL}")
        return corrupted_dir

    target_corrupted_count = int(math.floor(total_files * (error_rate_percent / 100.0)))
    pair_count = target_corrupted_count // 2
    actual_corrupted_count = pair_count * 2

    print(f"{Fore.GREEN}[INFO] Total Files: {total_files} | Target Error Rate: {error_rate_percent}%")
    print(
        f"[INFO] File pairs to swap: {pair_count} pairs ({actual_corrupted_count} total files corrupted){Style.RESET_ALL}")

    if pair_count == 0:
        return corrupted_dir

    chosen_indices = random.sample(range(total_files), actual_corrupted_count)

    for i in range(0, len(chosen_indices), 2):
        file1_path = txt_files[chosen_indices[i]]
        file2_path = txt_files[chosen_indices[i + 1]]

        swap_txt_contents(file1_path, file2_path)

        rel_p1 = os.path.relpath(file1_path, corrupted_dir)
        rel_p2 = os.path.relpath(file2_path, corrupted_dir)
        print(f"  {Fore.MAGENTA}↳ [SWAP]{Fore.WHITE} {rel_p1} ⇄ {rel_p2}")

    print(f"{Fore.GREEN}[SUCCESS] Label noise injection complete! Output path: {corrupted_dir}{Style.RESET_ALL}\n")
    return corrupted_dir


if __name__ == "__main__":
    dataset_path = os.environ.get("DATASET_PATH", "./dataset")
    SRC_DIR = f"{dataset_path}"
    TARGET_ROOT = "./dataset_corrupted"

    error_input = input("Enter target error rate percentage: ")
    try:
        error_rate = float(error_input)
        if 0 <= error_rate <= 100:
            inject_label_noise(SRC_DIR, TARGET_ROOT, error_rate)
        else:
            print(f"{Fore.RED}[ERROR] Error rate must be between 0 and 100.{Style.RESET_ALL}")
    except ValueError as e:
        print(f"{Fore.RED}[ERROR] Valid number required: {e}{Style.RESET_ALL}")

import json
import os
from glob import glob

from colorama import Fore, Style
from dotenv import load_dotenv

load_dotenv()


def generate_metadata_from_current_structure(target_dir):
    """
    현재 실제 파일 시스템 위치를 기준으로 file_name 상대 경로를 생성하고,
    .txt 파일에서는 경로가 아닌 '순수 캡션 텍스트'만 읽어서 metadata.jsonl을 생성합니다.
    """
    if not os.path.exists(target_dir):
        print(f"{Fore.RED}[ERROR] Target directory not found: '{target_dir}'{Style.RESET_ALL}")
        return

    print(f"{Fore.CYAN}[SYSTEM] Scanning actual filesystem structure in: '{target_dir}'...{Style.RESET_ALL}")

    img_paths = (
            glob(os.path.join(target_dir, "**", "*.png"), recursive=True)
            + glob(os.path.join(target_dir, "**", "*.jpg"), recursive=True)
            + glob(os.path.join(target_dir, "**", "*.jpeg"), recursive=True)
    )
    img_paths = sorted(img_paths)

    if not img_paths:
        print(f"{Fore.RED}[ERROR] No image files found in '{target_dir}'.{Style.RESET_ALL}")
        return

    metadata_entries = []
    missing_txt_count = 0

    for img_p in img_paths:
        real_rel_path = os.path.relpath(img_p, target_dir)

        txt_p = os.path.splitext(img_p)[0] + ".txt"
        caption = ""

        if os.path.exists(txt_p):
            with open(txt_p, "r", encoding="utf-8") as f_txt:
                content = f_txt.read().strip()

                try:
                    json_data = json.loads(content)
                    if isinstance(json_data, dict) and "text" in json_data:
                        caption = json_data["text"]
                    else:
                        caption = content
                except (json.JSONDecodeError, TypeError):
                    caption = content
        else:
            missing_txt_count += 1
            category = os.path.basename(os.path.dirname(img_p))
            caption = f"A photo of {category}" if category else "A photo"

        metadata_entries.append({
            "file_name": real_rel_path,
            "text": caption
        })

    metadata_path = os.path.join(target_dir, "metadata.jsonl")
    with open(metadata_path, "w", encoding="utf-8") as f_meta:
        for entry in metadata_entries:
            f_meta.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(
        f"{Fore.GREEN}[SUCCESS] Corrected 'metadata.jsonl' created successfully in '{target_dir}'!"
        f"\n  - Total processed images: {len(metadata_entries)}"
        f"\n  - Missing .txt files fallback: {missing_txt_count}{Style.RESET_ALL}\n"
    )


if __name__ == "__main__":
    dataset_path = os.environ.get("DATASET_PATH", "./dataset")

    generate_metadata_from_current_structure(dataset_path)

    corrupted_root = "./dataset_corrupted"
    if os.path.exists(corrupted_root):
        sub_dirs = glob(os.path.join(corrupted_root, "error_*", "*"))
        for s_dir in sub_dirs:
            if os.path.isdir(s_dir):
                generate_metadata_from_current_structure(s_dir)

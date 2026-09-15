import os
import shutil
import zipfile
from glob import glob

from colorama import Fore, Style
from dotenv import load_dotenv

load_dotenv()


def compress_sampled_images(
        data_root="./fft_data",
        output_zip="sampled_generations.zip",
        samples_per_category=10,
):
    temp_dir = "./temp_sampled_images"

    if not os.path.exists(data_root):
        print(f"{Fore.RED}[ERROR] Data root path '{data_root}' not found.{Style.RESET_ALL}")
        return

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    print(
        f"{Fore.CYAN}[SYSTEM] Sampling top {samples_per_category} images per category across all generations...{Style.RESET_ALL}")

    total_copied = 0

    # gen_* 디렉터리 탐색
    gen_dirs = sorted(glob(os.path.join(data_root, "gen_*")))

    if not gen_dirs:
        print(f"{Fore.YELLOW}[WARN] No generation directories found in '{data_root}'.{Style.RESET_ALL}")
        return

    for g_dir in gen_dirs:
        gen_name = os.path.basename(g_dir)
        images_dir = os.path.join(g_dir, "images")

        if not os.path.exists(images_dir):
            images_dir = g_dir

        cat_dirs = [
            d for d in glob(os.path.join(images_dir, "*")) if os.path.isdir(d)
        ]

        if not cat_dirs:
            cat_dirs = [images_dir]

        for c_dir in cat_dirs:
            cat_name = (
                os.path.basename(c_dir)
                if c_dir != images_dir
                else "Uncategorized"
            )

            img_files = []
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
                img_files.extend(glob(os.path.join(c_dir, ext)))

            img_files = sorted(img_files)[:samples_per_category]

            if not img_files:
                continue

            dest_dir = os.path.join(temp_dir, gen_name, cat_name)
            os.makedirs(dest_dir, exist_ok=True)

            for img_path in img_files:
                file_name = os.path.basename(img_path)
                shutil.copy2(img_path, os.path.join(dest_dir, file_name))
                total_copied += 1

            print(
                f"  {Fore.MAGENTA}↳ [{gen_name}]{Fore.WHITE} {cat_name}: {len(img_files)} images sampled"
            )

    if total_copied == 0:
        print(f"{Fore.RED}[ERROR] No images found to compress.{Style.RESET_ALL}")
        shutil.rmtree(temp_dir)
        return

    print(f"\n{Fore.CYAN}[SYSTEM] Creating zip archive: '{output_zip}'...{Style.RESET_ALL}")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(temp_dir):
            for file in files:
                abs_file = os.path.join(root, file)
                rel_file = os.path.relpath(abs_file, temp_dir)
                zipf.write(abs_file, rel_file)

    shutil.rmtree(temp_dir)

    print(
        f"{Fore.GREEN}[SUCCESS] Done! Total {total_copied} images compressed into '{output_zip}'.{Style.RESET_ALL}\n"
    )


if __name__ == "__main__":
    data_root_dir = os.environ.get("DATASET_PATH", "./fft_data")
    compress_sampled_images(
        data_root=data_root_dir,
        output_zip="sampled_generations.zip",
        samples_per_category=10,
    )

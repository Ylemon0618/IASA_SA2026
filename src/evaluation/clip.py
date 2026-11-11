import json
import os
from glob import glob

import torch
import torch.nn.functional as F
from PIL import Image
from colorama import Fore, Style
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

load_dotenv()


@torch.no_grad()
def calculate_clip_score(gen_dir, model, processor, device="cuda"):
    images_dir = os.path.join(gen_dir, "images")
    if not os.path.exists(images_dir):
        print(
            f"{Fore.BLUE}{'[CLIP]':<9}{Fore.MAGENTA}{gen_dir}{Fore.RED}: images directory not found{Style.RESET_ALL}"
        )
        return None

    captions = {}
    metadata_path = os.path.join(images_dir, "metadata.jsonl")

    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    captions[data["file_name"]] = data["text"]
                except (json.JSONDecodeError, KeyError):
                    continue

    if not captions:
        txt_paths = glob(
            os.path.join(images_dir, "**", "*.txt"), recursive=True
        )
        for txt_p in txt_paths:
            base_p = os.path.splitext(txt_p)[0]
            img_p = None
            for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
                if os.path.exists(base_p + ext):
                    img_p = base_p + ext
                    break

            if img_p:
                with open(txt_p, "r", encoding="utf-8") as f_txt:
                    content = f_txt.read().strip()
                    try:
                        j_data = json.loads(content)
                        caption = (
                            j_data.get("text", content)
                            if isinstance(j_data, dict)
                            else content
                        )
                    except (json.JSONDecodeError, TypeError):
                        caption = content

                rel_img_path = os.path.relpath(img_p, images_dir)
                captions[rel_img_path] = caption

    if not captions:
        print(
            f"{Fore.BLUE}{'[CLIP]':<9}{Fore.MAGENTA}{gen_dir}{Fore.RED}: No image-caption pairs found{Style.RESET_ALL}"
        )
        return None

    scores = []
    for img_rel_path, text in tqdm(
            captions.items(),
            desc=f"CLIP {os.path.basename(gen_dir)}",
            leave=False,
    ):
        img_path = os.path.join(images_dir, img_rel_path)
        if not os.path.exists(img_path):
            continue
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(
                text=[text],
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(device)
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

            outputs = model(**inputs)
            img_emb = F.normalize(outputs.image_embeds.float(), dim=-1)
            txt_emb = F.normalize(outputs.text_embeds.float(), dim=-1)
            scores.append((img_emb * txt_emb).sum().item())
        except Exception as e:
            print(
                f"{Fore.RED}  [CLIP] Error on {img_rel_path}: {e}{Style.RESET_ALL}"
            )
            continue

    if not scores:
        return None

    return float(sum(scores) / len(scores))


if __name__ == "__main__":
    generations = int(os.environ.get("GENERATIONS", 20))
    data_root = "./fft_data"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f"{Fore.BLUE}{'[CLIP]':<9}{Fore.WHITE}Loading CLIP model...{Style.RESET_ALL}"
    )
    model = (
        CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        .to(device)
        .to(torch.float16)
    )
    model.eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    results = {}
    for gen in range(generations):
        gen_dir = os.path.join(data_root, f"gen_{gen}")
        if not os.path.exists(gen_dir):
            continue

        print(
            f"{Fore.BLUE}{'[CLIP]':<9}{Fore.CYAN}Generation {Fore.MAGENTA}{gen}{Fore.WHITE}: Calculate CLIP score{Style.RESET_ALL}"
        )
        score = calculate_clip_score(gen_dir, model, processor, device=device)

        if score is not None:
            results[f"Gen_{gen}"] = score
            print(
                f"{Fore.BLUE}{'[CLIP]':<9}{Fore.CYAN}Generation {Fore.MAGENTA}{gen}{Fore.WHITE}: CLIP score is {Fore.GREEN}{score:.4f}{Style.RESET_ALL}"
            )

    del model
    torch.cuda.empty_cache()

    print("\n=== CLIP Score Evaluation Summary ===")
    for gen, score in results.items():
        print(f"{gen}: {score:.4f}")

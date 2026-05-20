import os
import math
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights
import torchvision.transforms as transforms
from tqdm import tqdm
from scipy.stats import entropy
from colorama import Fore, Style
from dotenv import load_dotenv

load_dotenv()


@torch.no_grad()
def calculate_inception_score(img_dir, batch_size=32, splits=10, device="cuda"):
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights, transform_input=False).to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    valid_extensions = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
    img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(valid_extensions)]

    if not img_paths:
        print(f"{Fore.BLUE}{'[IS]':<9}{Fore.CYAN}{Fore.MAGENTA}{img_dir}{Fore.RED}: No image in path{Style.RESET_ALL}")
        return None, None

    preds = []
    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:i + batch_size]
        batch_tensors = []

        for p in batch_paths:
            try:
                with Image.open(p).convert('RGB') as img:
                    batch_tensors.append(preprocess(img))
            except Exception as e:
                continue

        if not batch_tensors:
            continue

        batch_tensor = torch.stack(batch_tensors).to(device)

        out = model(batch_tensor)
        prob = F.softmax(out, dim=1).data.cpu().numpy()
        preds.append(prob)

    if not preds:
        return None, None

    preds = np.concatenate(preds, axis=0)

    split_scores = []
    split_size = len(img_paths) // splits

    for k in range(splits):
        part = preds[k * split_size: (k + 1) * split_size]

        p_yx = part
        p_y = np.mean(part, axis=0)

        kl_scores = []
        for i in range(part.shape[0]):
            kl = p_yx[i] * (np.log(p_yx[i] + 1e-10) - np.log(p_y + 1e-10))
            kl_scores.append(np.sum(kl))

        split_scores.append(np.exp(np.mean(kl_scores)))

    final_mean = np.mean(split_scores)
    final_std = np.std(split_scores)

    return final_mean, final_std


if __name__ == "__main__":
    generation = int(os.environ.get("GENERATIONS", 20))
    target_generations = list(range(generation))
    data_root = "./lora_data"

    results = {}
    for gen in target_generations:
        image_dir = os.path.join(data_root, f"gen_{gen}", "images")
        if os.path.exists(image_dir):
            print(f"{Fore.BLUE}{'[IS]':<9}{Fore.CYAN}Generation {Fore.MAGENTA}{gen}{Fore.WHITE}: Calculate IS score{Style.RESET_ALL}")
            mean, std = calculate_inception_score(image_dir)
            if None not in (mean, std):
                results[f"Gen_{gen}"] = (mean, std)
                print(f"{Fore.BLUE}{'[IS]':<9}{Fore.CYAN}Generation {Fore.MAGENTA}{gen}{Fore.WHITE}: IS score is {Fore.GREEN}{mean:.4f} (± {std:.4f}){Style.RESET_ALL}")

    print("\n=== IS Evaluation Summary ===")
    for gen, score in results.items():
        mean, std = score
        print(f"{gen}: {mean:.4f} (± {std:.4f})")
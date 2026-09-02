import os

import numpy as np
import torch
from colorama import Fore, Style
from dotenv import load_dotenv
from prdc import compute_prdc
from pytorch_fid.fid_score import ImagePathDataset
from pytorch_fid.inception import InceptionV3
from torch.utils.data import DataLoader

load_dotenv()


# ==========================================================
# Feature Extraction
# ==========================================================

def get_features(image_dir, device="cuda", batch_size=32):
    files = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]

    if len(files) == 0:
        raise ValueError(f"No images found in {image_dir}")

    dataset = ImagePathDataset(files)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]

    model = InceptionV3([block_idx]).to(device)
    model.eval()

    features = []

    with torch.no_grad():

        for batch in dataloader:

            batch = batch.to(device)

            pred = model(batch)[0]

            if pred.shape[2] != 1 or pred.shape[3] != 1:
                pred = torch.nn.functional.adaptive_avg_pool2d(
                    pred,
                    output_size=(1, 1)
                )

            pred = pred.squeeze(3).squeeze(2)

            features.append(pred.cpu().numpy())

    return np.concatenate(features, axis=0)


# ==========================================================
# PRDC Evaluation
# ==========================================================

def evaluate_prdc(
        base_gen,
        target_gen,
        data_root="./fft_data",
        device="cuda"
):
    path_real = os.path.join(
        data_root,
        f"gen_{base_gen}",
        "images"
    )

    path_fake = os.path.join(
        data_root,
        f"gen_{target_gen}",
        "images"
    )

    if not os.path.exists(path_real) or not os.path.exists(path_fake):
        print(
            f"{Fore.BLUE}{'[PRDC]':<9}"
            f"{Fore.CYAN}Generation "
            f"{Fore.MAGENTA}{target_gen}"
            f"{Fore.RED}: Path Not Found"
            f"{Style.RESET_ALL}"
        )

        return None

    print(
        f"{Fore.BLUE}{'[PRDC]':<9}"
        f"{Fore.CYAN}Generation "
        f"{Fore.MAGENTA}{target_gen}"
        f"{Fore.WHITE}: Evaluation Start"
        f"{Style.RESET_ALL}"
    )

    try:

        real_features = get_features(
            path_real,
            device=device
        )

        fake_features = get_features(
            path_fake,
            device=device
        )

        metrics = compute_prdc(
            real_features=real_features,
            fake_features=fake_features,
            nearest_k=5
        )

        print(
            f"{Fore.BLUE}{'[PRDC]':<9}"
            f"{Fore.CYAN}Generation "
            f"{Fore.MAGENTA}{target_gen}"
            f"{Fore.WHITE}: "
            f"P={Fore.GREEN}{metrics['precision']:.6f}"
            f"{Fore.WHITE}, "
            f"R={Fore.GREEN}{metrics['recall']:.6f}"
            f"{Fore.WHITE}, "
            f"D={Fore.GREEN}{metrics['density']:.6f}"
            f"{Fore.WHITE}, "
            f"C={Fore.GREEN}{metrics['coverage']:.6f}"
            f"{Style.RESET_ALL}"
        )

        return metrics

    except Exception as e:

        print(
            f"{Fore.BLUE}{'[PRDC]':<9}"
            f"{Fore.CYAN}Generation "
            f"{Fore.MAGENTA}{target_gen}"
            f"{Fore.RED}: Error Occurred ({e})"
            f"{Style.RESET_ALL}"
        )

        return None


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    generations = int(
        os.environ.get("GENERATIONS", 20)
    )

    results = {}

    for gen in range(1, generations):

        metrics = evaluate_prdc(
            base_gen=0,
            target_gen=gen
        )

        if metrics is not None:
            results[f"Gen_{gen}"] = metrics

    print("\n=== PRDC Evaluation Summary ===")

    for gen, metrics in results.items():
        print(
            f"{gen:<10} | "
            f"Precision: {metrics['precision']:.6f} | "
            f"Recall: {metrics['recall']:.6f} | "
            f"Density: {metrics['density']:.6f} | "
            f"Coverage: {metrics['coverage']:.6f}"
        )

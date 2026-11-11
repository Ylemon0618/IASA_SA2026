import os

import numpy as np
import torch
from PIL import Image
from colorama import Fore, Style
from dotenv import load_dotenv
from prdc import compute_prdc
from pytorch_fid.inception import InceptionV3
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

load_dotenv()


class SafePRDCImageDataset(Dataset):
    def __init__(self, files, img_size=(1024, 1024)):
        self.files = files
        self.transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),  # PIL -> Tensor [0, 1] Range
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # 손상된 이미지 처리
            img = Image.new("RGB", (1024, 1024), (0, 0, 0))

        return self.transform(img)


def get_features(image_dir, device="cuda", batch_size=32):
    files = []
    for root, _, filenames in os.walk(image_dir):
        for f in filenames:
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                files.append(os.path.join(root, f))

    files = sorted(files)

    if len(files) == 0:
        raise ValueError(f"No images found in {image_dir}")

    dataset = SafePRDCImageDataset(files)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True if device == "cuda" else False
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
                    pred, output_size=(1, 1)
                )

            pred = pred.squeeze(3).squeeze(2)
            features.append(pred.cpu().numpy())

    return np.concatenate(features, axis=0)


def resolve_image_path(base_dir):
    images_subdir = os.path.join(base_dir, "images")
    if os.path.exists(images_subdir):
        return images_subdir
    return base_dir


def evaluate_prdc(
        base_gen_dir,
        target_gen,
        data_root="./fft_data",
        device="cuda"
):
    target_dir = os.path.join(data_root, f"gen_{target_gen}")
    path_fake = resolve_image_path(target_dir)
    path_real = resolve_image_path(base_gen_dir)

    if not os.path.exists(path_real) or not os.path.exists(path_fake):
        print(
            f"{Fore.BLUE}{'[PRDC]':<9}"
            f"{Fore.CYAN}Generation "
            f"{Fore.MAGENTA}{target_gen}"
            f"{Fore.RED}: Path Not Found (Real: {path_real}, Fake: {path_fake})"
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
        real_features = get_features(path_real, device=device)
        fake_features = get_features(path_fake, device=device)

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


if __name__ == "__main__":
    generations = int(os.environ.get("GENERATIONS", 20))
    data_root = "./fft_data"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gen0_dir = os.path.join(data_root, "gen_0")
    if not os.path.exists(gen0_dir) and os.path.exists("./dataset"):
        gen0_dir = "./dataset"

    print(f"{Fore.YELLOW}[SYSTEM] Loading Baseline (gen_0) from: '{gen0_dir}'{Style.RESET_ALL}")

    results = {}

    for gen in range(1, generations):
        metrics = evaluate_prdc(
            base_gen_dir=gen0_dir,
            target_gen=gen,
            data_root=data_root,
            device=device
        )

        if metrics is not None:
            results[f"Gen_{gen}"] = metrics

    print("\n" + "=" * 80)
    print("=== PRDC Evaluation Summary ===")
    print("=" * 80)

    header = f"{'Gen':<10} | {'Precision':<10} | {'Recall':<10} | {'Density':<10} | {'Coverage':<10}"
    print(header)
    print("-" * len(header))

    for gen, metrics in results.items():
        print(
            f"{gen:<10} | "
            f"{metrics['precision']:>10.6f} | "
            f"{metrics['recall']:>10.6f} | "
            f"{metrics['density']:>10.6f} | "
            f"{metrics['coverage']:>10.6f}"
        )

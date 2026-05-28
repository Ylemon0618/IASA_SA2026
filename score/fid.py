import os
from dotenv import load_dotenv
from colorama import Fore, Style
from PIL import Image
import torch

import pytorch_fid.fid_score as fid_core
from pytorch_fid.fid_score import calculate_fid_given_paths

load_dotenv()


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
        except Exception as e:
            img = Image.new('RGB', (1024, 1024), (0, 0, 0))

        if self.transforms is not None:
            img = self.transforms(img)

        return img


fid_core.ImageFolderDataset = SafeFIDImageDataset


def evaluate_model_collapse(base_gen, target_gen, data_root="./lora_data", device="cuda"):
    path_real = os.path.join(data_root, f"gen_{base_gen}", "images")
    path_fake = os.path.join(data_root, f"gen_{target_gen}", "images")

    if not os.path.exists(path_real) or not os.path.exists(path_fake):
        print(
            f"{Fore.BLUE}{'[FID]':<9}{Fore.CYAN}Generation {Fore.MAGENTA}{target_gen}{Fore.RED}: Path Not Found{Style.RESET_ALL}")
        return None

    print(
        f"{Fore.BLUE}{'[FID]':<9}{Fore.CYAN}Generation {Fore.MAGENTA}{target_gen}{Fore.WHITE}: Evaluation Start{Style.RESET_ALL}")
    print(f"{Fore.WHITE} - Real Data: {Fore.MAGENTA}{path_real}{Style.RESET_ALL}")
    print(f"{Fore.WHITE} - Fake Data: {Fore.MAGENTA}{path_fake}{Style.RESET_ALL}")

    paths = [path_real, path_fake]

    batch_size = 32
    dims = 2048

    try:
        fid_value = calculate_fid_given_paths(paths, batch_size, device, dims)
        print(
            f"{Fore.BLUE}{'[FID]':<9}{Fore.CYAN}Generation {Fore.MAGENTA}{target_gen}{Fore.WHITE}: Evaluated value for {Fore.GREEN}{fid_value}{Style.RESET_ALL}")
        return fid_value

    except Exception as e:
        print(
            f"{Fore.BLUE}{'[FID]':<9}{Fore.CYAN}Generation {Fore.MAGENTA}{target_gen}{Fore.RED}: Error Occurred({e}){Style.RESET_ALL}")
        return None


if __name__ == "__main__":
    generations = int(os.environ.get("GENERATIONS", 20))
    target_generations = list(range(1, generations))

    results = {}
    for gen in target_generations:
        score = evaluate_model_collapse(base_gen=0, target_gen=gen)
        if score is not None:
            results[f"Gen_{gen}"] = score

    print("\n=== FID Evaluation Summary ===")
    for gen, score in results.items():
        print(f"{gen}: {score:.6f}")
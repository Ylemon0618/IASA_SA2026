import json
import logging
import os
import random
import subprocess
import sys
import warnings
from glob import glob

import torch
from PIL import Image
from colorama import Fore, Style
from diffusers import DiffusionPipeline
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import BitsAndBytesConfig
from transformers import logging as tf_logging
from transformers import pipeline as tf_pipeline

from modules.measures import measure_time

load_dotenv()

tf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

original_open = Image.open


def patched_open(*args, **kwargs):
    img = original_open(*args, **kwargs)
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.size != (1024, 1024):
        img = img.resize((1024, 1024), Image.Resampling.BILINEAR)
    return img


Image.open = patched_open


class FFTResearchPipeline:
    def __init__(
            self,
            base_model_path,
            total_gens=20,
            prompt_pool_path="synthetic_image_prompts.json",
            gen_0_data_dir=None,
    ):
        self.current_model = base_model_path
        self.total_gens = total_gens
        self.gen_0_data_dir = gen_0_data_dir

        self.output_root = os.environ.get("DATASET_PATH", "./fft_data")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if os.path.exists(prompt_pool_path):
            with open(prompt_pool_path, "r", encoding="utf-8") as f:
                self.category_prompts = json.load(f)

            self.flat_prompt_pool = []
            for cat, p_list in self.category_prompts.items():
                for p in p_list:
                    self.flat_prompt_pool.append((cat, p))

            print(
                f"{Fore.GREEN}[SYSTEM] Successfully loaded {len(self.flat_prompt_pool)} synthetic prompts across {len(self.category_prompts)} categories from '{prompt_pool_path}'.{Style.RESET_ALL}"
            )
        else:
            print(
                f"{Fore.RED}[SYSTEM] Critical Error: '{prompt_pool_path}' not found! Please run the prompt generation first.{Style.RESET_ALL}"
            )
            sys.exit(1)

        os.makedirs(self.output_root, exist_ok=True)

    @measure_time
    def generate_images(self, gen_num, count=100):
        base_dir = f"{self.output_root}/gen_{gen_num}/images"

        existing_images = glob(f"{base_dir}/**/*.png", recursive=True)
        if os.path.exists(base_dir) and len(existing_images) >= count:
            print(
                f"{Fore.YELLOW}{'[SDXL]':<9}{Fore.BLUE}Generation {Fore.MAGENTA}{gen_num}{Fore.WHITE}: All images already generated. Skipping generation.{Fore.RESET}"
            )
            return

        print(
            f"{Fore.YELLOW}{'[SDXL]':<9}{Fore.CYAN}Generation {Fore.MAGENTA}{gen_num}{Fore.WHITE}: Starting SDXL Image Synthesis{Style.RESET_ALL}"
        )
        os.makedirs(base_dir, exist_ok=True)

        is_hub_model = not os.path.isdir(self.current_model)
        pipe = DiffusionPipeline.from_pretrained(
            self.current_model,
            torch_dtype=torch.float16,
            variant="fp16" if is_hub_model else None,
        ).to(self.device)

        sampling_pool = []
        while len(sampling_pool) < count:
            shuffled_pool = list(self.flat_prompt_pool)
            random.shuffle(shuffled_pool)
            sampling_pool.extend(shuffled_pool)

        final_prompts = sampling_pool[:count]

        for i, (cat, prompt) in enumerate(
                tqdm(final_prompts, desc=f"Gen {gen_num} Image Progress")
        ):
            cat_dir = os.path.join(base_dir, cat)
            os.makedirs(cat_dir, exist_ok=True)

            image = pipe(prompt).images[0]
            image.save(os.path.join(cat_dir, f"img_{i:05d}.png"))

        del pipe
        torch.cuda.empty_cache()

        print(
            f"{Fore.YELLOW}{'[SDXL]':<9}{Fore.GREEN}Generation {Fore.MAGENTA}{gen_num}{Fore.WHITE}: SDXL Image Synthesis Finished{Style.RESET_ALL}",
            end="\n",
        )

    @measure_time
    def caption_images(self, gen_num):
        save_dir = f"{self.output_root}/gen_{gen_num}/images"
        img_paths = glob(f"{save_dir}/**/*.png", recursive=True)
        metadata = []

        metadata_path = os.path.join(save_dir, "metadata.jsonl")
        total_images = len(img_paths)

        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                existing_captions = f.readlines()
            if len(existing_captions) >= total_images and total_images > 0:
                print(
                    f"{Fore.YELLOW}{'[Llava]':<9}{Fore.BLUE}Generation {Fore.MAGENTA}{gen_num}{Fore.WHITE}: All captions already exist. Skipping captioning.{Fore.RESET}"
                )
                return

        print(
            f"{Fore.YELLOW}{'[Llava]':<9}{Fore.CYAN}Generation {Fore.MAGENTA}{gen_num}{Fore.WHITE}: Starting Llava Captioning{Style.RESET_ALL}"
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        captioner = tf_pipeline(
            "image-text-to-text",
            model="llava-hf/llava-1.5-7b-hf",
            model_kwargs={"quantization_config": bnb_config},
        )

        for img_p in tqdm(img_paths, desc="Captioning Progress"):
            raw_image = Image.open(img_p).convert("RGB")
            prompt = (
                "USER: <image>\nDescribe this image in detail.\nASSISTANT:"
            )

            outputs = captioner(
                images=raw_image,
                text=prompt,
                generate_kwargs={
                    "max_new_tokens": 100,
                    "max_length": None,
                    "do_sample": False,
                },
            )

            caption = (
                outputs[0]["generated_text"].split("ASSISTANT:")[-1].strip()
            )

            rel_file_path = os.path.relpath(img_p, save_dir)
            metadata.append({"file_name": rel_file_path, "text": caption})

            txt_path = img_p.rsplit(".", 1)[0] + ".txt"
            with open(txt_path, "w", encoding="utf-8") as f_txt:
                f_txt.write(caption)

        with open(metadata_path, "w", encoding="utf-8") as f:
            for entry in metadata:
                f.write(json.dumps(entry) + "\n")

        del captioner
        torch.cuda.empty_cache()

        print(
            f"{Fore.YELLOW}{'[Llava]':<9}{Fore.GREEN}Generation {Fore.MAGENTA}{gen_num}{Fore.WHITE}: Llava Captioning Finished{Style.RESET_ALL}",
            end="\n",
        )

    @measure_time
    def train_next_gen(self, gen_num, custom_train_dir=None):
        next_gen = gen_num + 1
        output_dir = f"./models/gen_{next_gen}"

        if os.path.exists(os.path.join(output_dir, "model_index.json")):
            print(
                f"{Fore.YELLOW}{'[FFT]':<9}{Fore.BLUE}Generation {Fore.MAGENTA}{gen_num}{Fore.WHITE}: Full Model already exists. Skipping training.{Fore.RESET}"
            )
            self.current_model = output_dir
            return

        print(
            f"{Fore.YELLOW}{'[FFT]':<9}{Fore.CYAN}Generation {Fore.MAGENTA}{gen_num}{Fore.WHITE}: Starting Full Fine-tuning{Style.RESET_ALL}"
        )

        train_data_path = (
            custom_train_dir
            if custom_train_dir
            else f"{self.output_root}/gen_{gen_num}/images"
        )

        subprocess.run(
            [
                "bash",
                "fft_pilot.sh",
                f"--pretrained_model_name_or_path={self.current_model}",
                f"--train_data_dir={train_data_path}",
                f"--output_dir={output_dir}",
            ],
            check=True,
        )

        self.current_model = output_dir

        print(
            f"{Fore.YELLOW}{'[FFT]':<9}{Fore.GREEN}Generation {Fore.MAGENTA}{gen_num}{Fore.WHITE}: Full Fine-tuning Finished{Style.RESET_ALL}",
            end="\n",
        )

    def auto_detect_start(self):
        completed = []
        for path in glob("./models/gen_*/model_index.json"):
            try:
                gen_n = int(path.split("gen_")[1].split("/")[0])
                completed.append(gen_n)
            except (IndexError, ValueError):
                continue

        if not completed:
            return 0, self.current_model

        latest = max(completed)
        model_path = f"./models/gen_{latest}"
        print(
            f"{Fore.YELLOW}{'[SYSTEM]':<9}{Fore.GREEN}Resume detected: gen_{latest} model found. "
            f"Resuming from generation {latest}.{Fore.RESET}"
        )
        return latest, model_path

    def run(self, count):
        print(f"{Fore.GREEN}=== FFT Pipeline Running ==={Style.RESET_ALL}")

        env_start = os.environ.get("START_GEN")
        if env_start is not None:
            start_gen = int(env_start)
            if start_gen > 0:
                self.current_model = f"./models/gen_{start_gen}"
                print(
                    f"{Fore.YELLOW}{'[SYSTEM]':<9}{Fore.GREEN}START_GEN={start_gen} set. "
                    f"Using model: {self.current_model}{Fore.RESET}"
                )
        else:
            start_gen, self.current_model = self.auto_detect_start()

        for gen in range(start_gen, self.total_gens):
            if gen == 0:
                if (
                        not self.gen_0_data_dir
                        or not os.path.exists(self.gen_0_data_dir)
                ):
                    print(
                        f"{Fore.RED}{'[SYSTEM]':<9}Generation {Fore.MAGENTA}{gen}{Fore.RED}: Valid Gen 0 dataset path not provided or does not exist ({self.gen_0_data_dir}).{Fore.RESET}"
                    )
                    sys.exit(1)

                print(
                    f"{Fore.YELLOW}{'[SYSTEM]':<9}{Fore.GREEN}Using custom external dataset for Gen 0 training: '{self.gen_0_data_dir}'.{Fore.RESET}"
                )
                self.train_next_gen(gen, custom_train_dir=self.gen_0_data_dir)
            else:
                self.generate_images(gen, count=count)
                self.caption_images(gen)
                self.train_next_gen(gen)


if __name__ == "__main__":
    model_path = os.environ.get(
        "MODEL_PATH", "stabilityai/stable-diffusion-xl-base-1.0"
    )
    generations = int(os.environ.get("GENERATIONS", 20))

    gen_0_dataset = input("[INPUT] Enter the dataset path for Generation 0: ").strip()

    pipeline = FFTResearchPipeline(
        base_model_path=model_path,
        total_gens=generations,
        prompt_pool_path="synthetic_image_prompts.json",
        gen_0_data_dir=gen_0_dataset,
    )
    pipeline.run(3000)

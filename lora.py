import json
import logging
import os
import subprocess
import warnings
import random
import sys
from glob import glob
from colorama import Fore, Style

import torch
from PIL import Image
from diffusers import DiffusionPipeline
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import logging as tf_logging
from transformers import pipeline as tf_pipeline

from modules.measures import *

load_dotenv()

tf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")


class LoRAResearchPipeline:
    def __init__(self, base_model_path, total_gens=20):
        self.current_model = base_model_path
        self.current_lora = None
        self.total_gens = total_gens
        self.output_root = "./lora_data"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prompt_pool = [
            "A high quality digital painting of a futuristic city with neon lights",
            "A majestic fantasy castle on top of a mountain, cinematic lighting",
            "A cute fluffy cat sitting on a wooden bench in a sunny garden",
            "A cyberpunk street kitchen at night with steam rising, realistic style",
            "An astronaut floating in deep space, colorful nebula in the background",
            "A serene tropical beach during a golden sunset, waves crashing",
            "A highly detailed portrait of an ancient wizard, mystical atmosphere",
            "A futuristic sports car driving through a rain-slicked highway at night"
        ]

    @measure_time
    def generate_images(self, gen_num, count=100):
        save_path = f"{self.output_root}/gen_{gen_num}/images"
        if os.path.exists(save_path) and len(glob(f"{save_path}/*.png")) >= count:
            print(
                f"{Fore.YELLOW}{'[SDXL]':<9}{Fore.BLUE}Generation {Fore.MAGENTA}{gen_num}{Fore.WHITE}: All image generated. Jumping generation.{Fore.RESET}")
            return

        print(
            f"{Fore.YELLOW}{'[SDXL]':<9}{Fore.CYAN}Generation {Fore.MAGENTA}{gen_num}{Fore.WHITE}: SDXL Image Synthesis Start{Style.RESET_ALL}")

        os.makedirs(save_path, exist_ok=True)

        pipe = DiffusionPipeline.from_pretrained(
            self.current_model,
            dtype=torch.float16,
            variant="fp16"
        ).to(self.device)

        if self.current_lora and os.path.exists(self.current_lora):
            print(f"{Fore.YELLOW}{'[SDXL]':<9}{Fore.WHITE}Loading LoRA weights from: {self.current_lora}{Fore.RESET}")
            pipe.load_lora_weights(self.current_lora)

        pipe.to(device="cuda", dtype=torch.float16)

        for i in range(count):
            prompt = random.choice(self.prompt_pool)
            image = pipe(prompt).images[0]
            image.save(f"{save_path}/img_{i:05d}.png")

        del pipe
        torch.cuda.empty_cache()

        print(
            f"{Fore.YELLOW}{'[SDXL]':<9}{Fore.GREEN}Generation {Fore.MAGENTA}{gen_num}{Fore.WHITE}: SDXL Image Synthesis End{Style.RESET_ALL}",
            end=' ')

    @measure_time
    def caption_images(self, gen_num):
        save_dir = f"{self.output_root}/gen_{gen_num}/images"
        img_paths = glob(f"{save_dir}/*.png")
        metadata = []

        metadata_path = os.path.join(save_dir, "metadata.jsonl")
        image_files = glob(os.path.join(save_dir, "*.png"))
        total_images = len(image_files)

        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                existing_captions = f.readlines()

            if len(existing_captions) >= total_images:
                print(
                    f"{Fore.YELLOW}{'[Llava]':<9}{Fore.BLUE}Generation {Fore.MAGENTA}{gen_num}{Fore.WHITE}: All caption existing. Jumping captioning.{Fore.RESET}")
                return

        print(
            f"{Fore.YELLOW}{'[Llava]':<9}{Fore.CYAN}Generation {Fore.MAGENTA}{gen_num}{Fore.WHITE}: Llava Captioning Start{Style.RESET_ALL}")

        captioner = tf_pipeline(
            "image-text-to-text",
            model="llava-hf/llava-1.5-7b-hf",
            device=0,
            model_kwargs={"dtype": torch.float16}
        )

        for i, img_path in enumerate(tqdm(img_paths, desc="Captioning Progress")):
            raw_image = Image.open(img_path).convert("RGB")
            prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"
            outputs = captioner(
                images=raw_image,
                text=prompt,
                generate_kwargs={
                    "max_new_tokens": 100,
                    "max_length": None,
                    "do_sample": False
                }
            )

            caption = outputs[0]['generated_text'].split("ASSISTANT:")[-1].strip()

            file_name = os.path.basename(img_path)
            metadata.append({"file_name": file_name, "text": caption})

            with open(f"{save_dir}/{file_name.replace('.png', '.txt')}", "w") as f_txt:
                f_txt.write(caption)

        with open(f"{save_dir}/metadata.jsonl", 'w') as f:
            for entry in metadata:
                f.write(json.dumps(entry) + "\n")

        del captioner
        torch.cuda.empty_cache()

        print(
            f"{Fore.YELLOW}{'[Llava]':<9}{Fore.GREEN}Generation {Fore.MAGENTA}{gen_num}{Fore.WHITE}: Llava Captioning End{Style.RESET_ALL}",
            end=' ')

    @measure_time
    def run_lora_train(self, gen_num):
        next_gen = gen_num + 1

        if os.path.exists(f"./models/lora_gen_{next_gen}/pytorch_lora_weights.safetensors"):
            print(
                f"{Fore.YELLOW}{'[LoRA]':<9}{Fore.BLUE}Generation {Fore.MAGENTA}{gen_num}{Fore.WHITE}: LoRA existing. Jumping training.{Fore.RESET}")
            self.current_lora = f"./models/lora_gen_{next_gen}/pytorch_lora_weights.safetensors"
            return

        print(
            f"{Fore.YELLOW}{'[LoRA]':<9}{Fore.CYAN}Generation {Fore.MAGENTA}{gen_num}{Fore.WHITE}: LoRA Training Start{Style.RESET_ALL}")

        subprocess.run([
            "bash", "lora_pilot.sh",
            f"--pretrained_model_name_or_path={self.current_model}",
            f"--train_data_dir={self.output_root}/gen_{gen_num}/images",
            f"--output_dir=./models/lora_gen_{next_gen}",
        ], check=True)

        self.current_lora = f"./models/lora_gen_{next_gen}/pytorch_lora_weights.safetensors"
        print(
            f"{Fore.YELLOW}{'[LoRA]':<9}{Fore.GREEN}Generation {Fore.MAGENTA}{gen_num}{Fore.WHITE}: LoRA Training End{Style.RESET_ALL}",
            end=' ')

    def run(self):
        print(f"{Fore.GREEN}=== LoRA Pipeline Running ==={Style.RESET_ALL}")

        for gen in range(self.total_gens):
            if gen == 0:
                gen_0_dir = f"{self.output_root}/gen_0/images"
                if not os.path.exists(gen_0_dir) or len(glob(f"{gen_0_dir}/*.png")) == 0:
                    print(
                        f"{Fore.RED}{'[SYSTEM]':<9}Generation {Fore.MAGENTA}{gen}{Fore.RED}: Seed images not found in '{gen_0_dir}'. Please provide source images.{Fore.RESET}")
                    sys.exit(1)

                print(
                    f"{Fore.YELLOW}{'[SYSTEM]':<9}{Fore.GREEN}Detected seed images in gen_0. Proceeding to Caption & Train.{Fore.RESET}")
                self.caption_images(gen)
                self.run_lora_train(gen)
            else:
                self.generate_images(gen)
                self.caption_images(gen)
                self.run_lora_train(gen)


if __name__ == "__main__":
    model_path = os.environ.get("MODEL_PATH", "stabilityai/stable-diffusion-xl-base-1.0")
    generations = int(os.environ.get("GENERATIONS", 20))

    pipeline = LoRAResearchPipeline(base_model_path=model_path, total_gens=generations)
    pipeline.run()

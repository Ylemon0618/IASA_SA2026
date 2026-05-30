import os
import json
import random
import sys
import warnings
import logging
from glob import glob
from colorama import Fore, Style

import torch
import subprocess
from PIL import Image
from dotenv import load_dotenv
from tqdm import tqdm

from diffusers import DiffusionPipeline
from transformers import logging as tf_logging
from transformers import pipeline as tf_pipeline, BitsAndBytesConfig

from modules.measures import *

load_dotenv()

tf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

original_open = Image.open


def patched_open(*args, **kwargs):
    img = original_open(*args, **kwargs)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    if img.size != (1024, 1024):
        img = img.resize((1024, 1024), Image.Resampling.BILINEAR)
    return img


Image.open = patched_open


class FFTResearchPipeline:
    def __init__(self, base_model_path, total_gens=20):
        self.current_model = base_model_path
        self.total_gens = total_gens
        self.output_root = "./fft_data"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.prompt_pool = [
            "A wooden fishing boat anchored near a rocky shore with calm blue water",
            "A large cargo ship traveling through the open sea under rain clouds",
            "Two people rowing a small wooden canoe on a peaceful city river",
            "A majestic sailboat at sea with white sails catching the wind",
            "A group of friends holding green beer bottles at an outdoor party",
            "A clear glass bottle of soda sitting on a messy kitchen counter",
            "A man pouring red wine from a bottle into a glass at a dinner table",
            "A close-up of an open refrigerator with food and colorful juice bottles",
            "A red double decker bus driving down a wet city street in the afternoon",
            "A yellow school bus stopped on the road with its door wide open",
            "A large white tour bus parked in a crowded station next to a brick building",
            "A vintage green trolley passing by a park garden and a low fence",
            "A shiny silver sports car parked in front of a modern brick building",
            "A black truck driving through a rural highway with trees in the distance",
            "A clean white minivan parked inside a garage with a blurry background",
            "A race car drifting along a dirt road, kicking up a cloud of dust",
            "A grey tabby cat sitting on a wooden chair by a sunny kitchen window",
            "A fluffy ginger cat sleeping on a sofa with a blanket underneath",
            "A small white cat with bright eyes looking up from beneath a table",
            "A black and white cat stretching its body on a patterned rug",
            "An empty wooden rocking chair in a dimly lit living room with patterned fabric",
            "A modern black office chair placed next to a cluttered desk with a computer",
            "A couple of people sitting on red folding chairs in a backyard garden",
            "A dining table surrounded by four matching wooden chairs in a quiet apartment",
            "A black and white cow grazing in a lush green pasture near a fence",
            "A small calf resting on a bed of dry straw inside a wooden barn",
            "A group of brown cows standing near a water trough under a big tree",
            "A large dark bull standing behind a metal wire fence on a remote farm"
        ]
        os.makedirs(self.output_root, exist_ok=True)

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

        for i in range(count):
            prompt = random.choice(self.prompt_pool)
            image = pipe(prompt).images[0]
            image.save(f"{save_path}/img_{i:05d}.png")

        del pipe
        torch.cuda.empty_cache()

        print(
            f"{Fore.YELLOW}{'[SDXL]':<9}{Fore.GREEN}Generation {Fore.MAGENTA}{gen_num}{Fore.WHITE}: SDXL Image Synthesis End{Style.RESET_ALL}",
            end='\n')

    @measure_time
    def caption_images(self, gen_num):
        save_dir = f"{self.output_root}/gen_{gen_num}/images"
        img_paths = glob(f"{save_dir}/*.png")
        metadata = []

        metadata_path = os.path.join(save_dir, "metadata.jsonl")
        total_images = len(img_paths)

        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                existing_captions = f.readlines()
            if len(existing_captions) >= total_images and total_images > 0:
                print(
                    f"{Fore.YELLOW}{'[Llava]':<9}{Fore.BLUE}Generation {Fore.MAGENTA}{gen_num}{Fore.WHITE}: All caption existing. Jumping captioning.{Fore.RESET}")
                return

        print(
            f"{Fore.YELLOW}{'[Llava]':<9}{Fore.CYAN}Generation {Fore.MAGENTA}{gen_num}{Fore.WHITE}: Llava Captioning Start{Style.RESET_ALL}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )

        captioner = tf_pipeline(
            "image-text-to-text",
            model="llava-hf/llava-1.5-7b-hf",
            model_kwargs={"quantization_config": bnb_config}
        )

        for img_p in tqdm(img_paths, desc="Captioning Progress"):
            raw_image = Image.open(img_p).convert("RGB")
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
            file_name = os.path.basename(img_p)
            metadata.append({"file_name": file_name, "text": caption})

            with open(f"{save_dir}/{file_name.replace('.png', '.txt')}", "w", encoding="utf-8") as f_txt:
                f_txt.write(caption)

        with open(metadata_path, 'w', encoding="utf-8") as f:
            for entry in metadata:
                f.write(json.dumps(entry) + "\n")

        del captioner
        torch.cuda.empty_cache()

        print(
            f"{Fore.YELLOW}{'[Llava]':<9}{Fore.GREEN}Generation {Fore.MAGENTA}{gen_num}{Fore.WHITE}: Llava Captioning End{Style.RESET_ALL}",
            end='\n')

    @measure_time
    def train_next_gen(self, gen_num):
        next_gen = gen_num + 1
        output_dir = f"./models/gen_{next_gen}"

        if os.path.exists(os.path.join(output_dir, "model_index.json")):
            print(
                f"{Fore.YELLOW}{'[FFT]':<9}{Fore.BLUE}Generation {Fore.MAGENTA}{gen_num}{Fore.WHITE}: Full Model existing. Jumping training.{Fore.RESET}")
            self.current_model = output_dir
            return

        print(
            f"{Fore.YELLOW}{'[FFT]':<9}{Fore.CYAN}Generation {Fore.MAGENTA}{gen_num}{Fore.WHITE}: Full Fine-tuning Start{Style.RESET_ALL}")

        subprocess.run([
            "bash", "fft_pilot.sh",
            f"--pretrained_model_name_or_path={self.current_model}",
            f"--train_data_dir={self.output_root}/gen_{gen_num}/images",
            f"--output_dir={output_dir}"
        ], check=True)

        self.current_model = output_dir

        print(
            f"{Fore.YELLOW}{'[FFT]':<9}{Fore.GREEN}Generation {Fore.MAGENTA}{gen_num}{Fore.WHITE}: Full Fine-tuning End{Style.RESET_ALL}",
            end='\n')

    def run(self):
        print(f"{Fore.GREEN}=== FFT Pipeline Running ==={Style.RESET_ALL}")

        for gen in range(self.total_gens):
            if gen == 0:
                gen_0_dir = f"{self.output_root}/gen_0/images"
                if not os.path.exists(gen_0_dir) or len(glob(f"{gen_0_dir}/*.jpg")) == 0:
                    print(
                        f"{Fore.RED}{'[SYSTEM]':<9}Generation {Fore.MAGENTA}{gen}{Fore.RED}: Seed images (.jpg) not found in '{gen_0_dir}'.{Fore.RESET}")
                    sys.exit(1)

                print(
                    f"{Fore.YELLOW}{'[SYSTEM]':<9}{Fore.GREEN}Detected seed images in gen_0. Proceeding to Caption & Train.{Fore.RESET}")
                self.caption_images(gen)
                self.train_next_gen(gen)
            else:
                self.generate_images(gen)
                self.caption_images(gen)
                self.train_next_gen(gen)


if __name__ == "__main__":
    model_path = os.environ.get("MODEL_PATH", "stabilityai/stable-diffusion-xl-base-1.0")
    generations = int(os.environ.get("GENERATIONS", 20))

    pipeline = FFTResearchPipeline(base_model_path=model_path, total_gens=generations)
    pipeline.run()
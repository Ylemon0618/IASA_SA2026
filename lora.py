import os
import json
import torch
from glob import glob
from PIL import Image
import subprocess
from dotenv import load_dotenv
from diffusers import DiffusionPipeline
from transformers import pipeline as tf_pipeline
from colorama import Fore, Style
from tqdm import tqdm
from modules.measures import *

load_dotenv()


class LoRAResearchPipeline:
    def __init__(self, base_model_path, total_gens=20):
        self.current_model = base_model_path
        self.current_lora = None
        self.total_gens = total_gens
        self.output_root = "./lora_data"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @measure_time
    def generate_images(self, gen_num, count=10):
        save_path = f"{self.output_root}/gen_{gen_num}/images"
        if os.path.exists(save_path) and len(glob(f"{save_path}/*.png")) >= count:
            print(f"{Fore.CYAN}Generation {Fore.MAGENTA}{gen_num}{Fore.WHITE}: All image generated. Jumping generation.{Fore.RESET}")
            return

        print(f"{Fore.CYAN}Generation {Fore.MAGENTA}{gen_num}{Fore.WHITE}: SDXL Image Synthesis Start{Style.RESET_ALL}")

        os.makedirs(save_path, exist_ok=True)

        pipe = DiffusionPipeline.from_pretrained(
            self.current_model,
            dtype=torch.float16,
            variant="fp16"
        ).to(self.device)

        if self.current_lora and os.path.exists(self.current_lora):
            pipe.load_lora_weights(self.current_lora)

        prompt = "A high quality digital painting of a futuristic city"

        for i in range(count):
            image = pipe(prompt).images[0]
            image.save(f"{save_path}/img_{i:05d}.png")

        del pipe
        torch.cuda.empty_cache()

        print(f"{Fore.GREEN}Generation {Fore.MAGENTA}{gen_num}{Fore.WHITE}: SDXL Image Synthesis End{Style.RESET_ALL}",
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
                print(f"{Fore.CYAN}Generation {Fore.MAGENTA}{gen_num}{Fore.WHITE}: All caption existing. Jumping captioning.{Fore.RESET}")
                return

        print(f"{Fore.CYAN}Generation {Fore.MAGENTA}{gen_num}{Fore.WHITE}: Llava Captioning Start{Style.RESET_ALL}")

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

        print(f"{Fore.CYAN}Generation {Fore.MAGENTA}{gen_num}{Fore.WHITE}: Llava Captioning End{Style.RESET_ALL}", end=' ')

    @measure_time
    def run_lora_train(self, gen_num):
        next_gen = gen_num + 1
        print(f"{Fore.CYAN}Generation {Fore.MAGENTA}{gen_num}{Fore.WHITE}: LoRA Training Start{Style.RESET_ALL}")

        subprocess.run([
            "bash", "lora_pilot.sh",
            f"--pretrained_model_name_or_path={self.current_model}",
            f"--train_data_dir={self.output_root}/gen_{gen_num}/images",
            f"--output_dir=./models/lora_gen_{next_gen}",
        ], check=True)

        self.current_lora = f"./models/lora_gen_{next_gen}/pytorch_lora_weights.safetensors"
        print(f"{Fore.CYAN}Generation {Fore.MAGENTA}{gen_num}{Fore.WHITE}: LoRA Training End{Style.RESET_ALL}", end=' ')

    def run(self):
        for gen in range(self.total_gens):
            self.generate_images(gen)
            self.caption_images(gen)
            self.run_lora_train(gen)


if __name__ == "__main__":
    model_path = os.environ.get("MODEL_PATH", "stabilityai/stable-diffusion-xl-base-1.0")
    generations = int(os.environ.get("GENERATIONS", 20))

    pipeline = LoRAResearchPipeline(base_model_path=model_path, total_gens=generations)
    pipeline.run()

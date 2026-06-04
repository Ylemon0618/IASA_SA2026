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
from diffusers import DiffusionPipeline
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import logging as tf_logging
from transformers import pipeline as tf_pipeline, BitsAndBytesConfig

from modules.measures import *

load_dotenv()

tf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")


class LoRAResearchPipeline:
    def __init__(self, base_model_path, total_gens=20):
        self.base_model_path = base_model_path  # 원본 베이스 모델 고정 참조용
        self.current_model = base_model_path  # 이미지 생성에 사용할 현재 모델
        self.current_lora = None
        self.total_gens = total_gens
        self.output_root = "./lora_data"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prompt_pool = [
            # 1. Boat & Ship (선박)
            "A wooden fishing boat anchored near a rocky shore with calm blue water",
            "A large cargo ship traveling through the open sea under rain clouds",
            "Two people rowing a small wooden canoe on a peaceful city river",
            "A majestic sailboat at sea with white sails catching the wind",

            # 2. Bottle & Tabletop (병 및 식탁 소품)
            "A group of friends holding green beer bottles at an outdoor party",
            "A clear glass bottle of soda sitting on a messy kitchen counter",
            "A man pouring red wine from a bottle into a glass at a dinner table",
            "A close-up of an open refrigerator with food and colorful juice bottles",

            # 3. Bus & Transport (버스 및 대중교통)
            "A red double decker bus driving down a wet city street in the afternoon",
            "A yellow school bus stopped on the road with its door wide open",
            "A large white tour bus parked in a crowded station next to a brick building",
            "A vintage green trolley passing by a park garden and a low fence",

            # 4. Car (자동차)
            "A shiny silver sports car parked in front of a modern brick building",
            "A black truck driving through a rural highway with trees in the distance",
            "A clean white minivan parked inside a garage with a blurry background",
            "A race car drifting along a dirt road, kicking up a cloud of dust",

            # 5. Cat (고양이)
            "A grey tabby cat sitting on a wooden chair by a sunny kitchen window",
            "A fluffy ginger cat sleeping on a sofa with a blanket underneath",
            "A small white cat with bright eyes looking up from beneath a table",
            "A black and white cat stretching its body on a patterned rug",

            # 6. Chair & Living Room (의자 및 가구 실내)
            "An empty wooden rocking chair in a dimly lit living room with patterned fabric",
            "A modern black office chair placed next to a cluttered desk with a computer",
            "A couple of people sitting on red folding chairs in a backyard garden",
            "A dining table surrounded by four matching wooden chairs in a quiet apartment",

            # 7. Cow (소 및 가축)
            "A black and white cow grazing in a lush green pasture near a fence",
            "A small calf resting on a bed of dry straw inside a wooden barn",
            "A group of brown cows standing near a water trough under a big tree",
            "A large dark bull standing behind a metal wire fence on a remote farm"
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

        # FFT와 동일하게: 베이스 모델에 최신 LoRA를 머지한 파이프라인으로 생성
        is_hub_model = not os.path.isdir(self.base_model_path)
        pipe = DiffusionPipeline.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            variant="fp16" if is_hub_model else None
        ).to(self.device)

        if self.current_lora and os.path.exists(self.current_lora):
            print(f"{Fore.YELLOW}{'[SDXL]':<9}{Fore.WHITE}Merging LoRA weights from: {self.current_lora}{Fore.RESET}")
            pipe.load_lora_weights(self.current_lora)
            pipe.unet = pipe.unet.merge_and_unload()

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

        # FFT와 동일하게: 4bit quantization으로 LLaVA 로드
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

        for img_path in tqdm(img_paths, desc="Captioning Progress"):
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
    def run_lora_train(self, gen_num):
        next_gen = gen_num + 1
        output_dir = f"./models/lora_gen_{next_gen}"

        if os.path.exists(f"{output_dir}/pytorch_lora_weights.safetensors"):
            print(
                f"{Fore.YELLOW}{'[LoRA]':<9}{Fore.BLUE}Generation {Fore.MAGENTA}{gen_num}{Fore.WHITE}: LoRA existing. Jumping training.{Fore.RESET}")
            self.current_lora = f"{output_dir}/pytorch_lora_weights.safetensors"
            return

        print(
            f"{Fore.YELLOW}{'[LoRA]':<9}{Fore.CYAN}Generation {Fore.MAGENTA}{gen_num}{Fore.WHITE}: LoRA Training Start{Style.RESET_ALL}")

        # 매 세대 항상 베이스 모델 기준으로 학습 → FFT처럼 이전 세대 오염 차단
        subprocess.run([
            "bash", "lora_pilot.sh",
            f"--pretrained_model_name_or_path={self.base_model_path}",
            f"--train_data_dir={self.output_root}/gen_{gen_num}/images",
            f"--output_dir={output_dir}",
        ], check=True)

        self.current_lora = f"{output_dir}/pytorch_lora_weights.safetensors"
        print(
            f"{Fore.YELLOW}{'[LoRA]':<9}{Fore.GREEN}Generation {Fore.MAGENTA}{gen_num}{Fore.WHITE}: LoRA Training End{Style.RESET_ALL}",
            end='\n')

    def auto_detect_start(self):
        completed = []
        for path in glob("./models/lora_gen_*/pytorch_lora_weights.safetensors"):
            try:
                gen_n = int(path.split("lora_gen_")[1].split("/")[0])
                completed.append(gen_n)
            except (IndexError, ValueError):
                continue

        if not completed:
            return 0, None

        latest = max(completed)
        lora_path = f"./models/lora_gen_{latest}/pytorch_lora_weights.safetensors"
        print(
            f"{Fore.YELLOW}{'[SYSTEM]':<9}{Fore.GREEN}Resume detected: lora_gen_{latest} found. "
            f"Resuming from generation {latest}.{Fore.RESET}")
        return latest, lora_path

    def run(self):
        print(f"{Fore.GREEN}=== LoRA Pipeline Running ==={Style.RESET_ALL}")

        env_start = os.environ.get("START_GEN")
        if env_start is not None:
            start_gen = int(env_start)
            if start_gen > 0:
                self.current_lora = f"./models/lora_gen_{start_gen}/pytorch_lora_weights.safetensors"
                print(
                    f"{Fore.YELLOW}{'[SYSTEM]':<9}{Fore.GREEN}START_GEN={start_gen} set.{Fore.RESET}")
        else:
            start_gen, self.current_lora = self.auto_detect_start()

        for gen in range(start_gen, self.total_gens):
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
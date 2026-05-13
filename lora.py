import os
import json
import torch
from glob import glob
from PIL import Image
import subprocess
from dotenv import load_dotenv
from diffusers import DiffusionPipeline
from transformers import pipeline as tf_pipeline
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
        print(f"Generation {gen_num}: SDXL Image Synthesis Start")
        save_path = f"{self.output_root}/gen_{gen_num}/images"
        if os.path.exists(save_path) and len(glob(f"{save_path}/*.png")) >= count:
            return

        os.makedirs(save_path, exist_ok=True)

        pipe = DiffusionPipeline.from_pretrained(
            self.current_model,
            torch_dtype=torch.float16,
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
        print(f"Generation {gen_num}: SDXL Image Synthesis End", end=' ')

    @measure_time
    def caption_images(self, gen_num):
        print(f"Generation {gen_num}: Llava Captioning Start")
        save_dir = f"{self.output_root}/gen_{gen_num}"
        img_paths = glob(f"{save_dir}/images/*.png")
        metadata = []

        captioner = tf_pipeline(
            "image-text-to-text",
            model="llava-hf/llava-1.5-7b-hf",
            device=0,
            model_kwargs={"torch_dtype": torch.float16}  # 반정밀도 사용으로 메모리 절약
        )

        for img_p in img_paths:
            raw_image = Image.open(img_p)
            prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"
            outputs = captioner(
                images=raw_image,
                text=prompt,
                generate_kwargs={"max_new_tokens": 50, "max_length": None}
            )

            caption = outputs[0]['generated_text'].split("ASSISTANT:")[-1].strip()

            file_name = os.path.basename(img_p)
            metadata.append({"file_name": file_name, "text": caption})

            with open(f"{save_dir}/images/{file_name.replace('.png', '.txt')}", "w") as f_txt:
                f_txt.write(caption)

        with open(f"{save_dir}/metadata.jsonl", 'w') as f:
            for entry in metadata:
                f.write(json.dumps(entry) + "\n")

        del captioner
        torch.cuda.empty_cache()
        print(f"Generation {gen_num}: Llava Captioning End", end=' ')

    @measure_time
    def run_lora_train(self, gen_num):
        next_gen = gen_num + 1
        print(f"Generation {gen_num}: LoRA Training Start")

        subprocess.run([
            "bash", "lora_pilot.sh",
            f"--pretrained_model_name_or_path={self.current_model}",
            f"--train_data_dir={self.output_root}/gen_{gen_num}/images",
            f"--output_dir=./models/lora_gen_{next_gen}",
        ], check=True)

        self.current_lora = f"./models/lora_gen_{next_gen}/pytorch_lora_weights.safetensors"

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

import os
import json
import torch
from glob import glob
from PIL import Image
import subprocess
from dotenv import load_dotenv
from modules.measures import *

load_dotenv()


class FFTResearchPipeline:
    def __init__(self, base_model_path, total_gens=20):
        self.current_model = base_model_path
        self.total_gens = total_gens
        self.output_root = "./fft_data"

        os.makedirs(self.output_root, exist_ok=True)

    @measure_time
    def generate_images(self, gen_num, count=100000):
        print(f"Generation {gen_num}: SDXL Image Synthesis Start")

        save_path = f"{self.output_root}/gen_{gen_num}/images"
        os.makedirs(save_path, exist_ok=True)

        # TODO: SDXL Pipeline Load and Loop

        print(f"Generation {gen_num}: SDXL Image Synthesis End", end=' ')

    @measure_time
    def caption_images(self, gen_num):
        print(f"Generation {gen_num}: Llava Captioning Start")

        img_paths = glob(f"{self.output_root}/gen_{gen_num}/images/*.png")
        metadata = []

        # TODO: Llava Model Load (4-bit)
        for img_p in img_paths:
            # metadata.append({"file_name": os.path.basename(img_p), "text": caption})
            pass

        with open(f"{self.output_root}/gen_{gen_num}/metadata.jsonl", 'w') as f:
            for entry in metadata:
                f.write(json.dumps(entry) + "\n")

        print(f"Generation {gen_num}: Llava Captioning End", end=' ')

    @measure_time
    def train_next_gen(self, gen_num):
        print(f"Generation {gen_num}: Full Fine-tuning Start")

        subprocess.run([
            "bash", "fft_pilot.sh",
            f"--train_data_dir={self.output_root}/gen_{gen_num}",
            f"--output_dir=./models/gen_{gen_num + 1}"
        ])

        self.current_model = f"./models/gen_{gen_num + 1}"

        print(f"Generation {gen_num}: Full Fine-tuning End", end=' ')

    def run(self):
        for gen in range(self.total_gens):
            self.generate_images(gen)
            self.caption_images(gen)
            # TODO: Measure and store FID, IS, etc.
            self.train_next_gen(gen)


if __name__ == "__main__":
    generations = int(os.environ.get("GENERATIONS"))
    pipeline = FFTResearchPipeline(base_model_path="stabilityai/stable-diffusion-xl-base-1.0")
    pipeline.run()

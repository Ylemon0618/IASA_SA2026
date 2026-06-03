import os
import json
from glob import glob
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel


def calculate_generation_clip_score(gen_dir):
    metadata_path = os.path.join(gen_dir, "images", "metadata.jsonl")
    images_dir = os.path.join(gen_dir, "images")

    if not os.path.exists(metadata_path):
        return None

    # CLIP 오버헤드 최소화를 위해 FP16 및 CUDA 세팅
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_id).to(device).to(torch.float16)
    processor = CLIPProcessor.from_pretrained(model_id)

    captions_dict = {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            captions_dict[data["file_name"]] = data["text"]

    total_score = 0.0
    match_count = 0

    # 해당 세대의 모든 이미지와 캡션을 1:1 매칭하여 배치 연산
    for img_name, text in tqdm(captions_dict.items(), desc=f"Evaluating {os.path.basename(gen_dir)}", leave=False):
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            continue

        try:
            image = Image.open(img_path).convert("RGB")

            # CLIP 전처리 및 텐서 배치화
            inputs = processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)

            with torch.no_grad():
                outputs = model(**inputs)
                # 이미지 텐서와 텍스트 텐서 간의 코사인 유사도 추출
                logits_per_image = outputs.logits_per_image
                score = logits_per_image.item()

            total_score += score
            match_count += 1
        except Exception as e:
            continue

    # 전역 메모리 해제 (SDXL/LLaVA 루프 복귀 시 OOM 방지)
    del model, processor
    torch.cuda.empty_cache()

    return total_score / match_count if match_count > 0 else 0.0


def main():
    data_root = "./fft_data"
    # gen_0, gen_1, gen_2 ... 폴더 숫자에 맞춰 정렬 탐색
    gen_folders = sorted(glob(os.path.join(data_root, "gen_*")), key=lambda x: int(x.split("_")[-1]))

    print(f"\n=========================================")
    print(f"📊 CLIP Score Evaluation Matrix (FFT Tracks)")
    print(f"=========================================\n")

    results = {}
    for gen_dir in gen_folders:
        gen_num = int(gen_dir.split("_")[-1])
        # 0세대는 소스 이미지 정보이므로 프롬프트 정렬 점수 비교군으로 사용하거나 스킵 가능
        score = calculate_generation_clip_score(gen_dir)

        if score is not None:
            results[gen_num] = score
            print(f"Generation {gen_num:2d}: CLIP Score = {score:.4f}")
        else:
            print(f"Generation {gen_num:2d}: metadata.jsonl 미검출 (스킵)")

    # 향후 논문 그래프 플로팅(matplotlib)용 JSON 로그 저장
    with open(os.path.join(data_root, "clip_scores_report.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\n[+] 리포트 저장 완료: {data_root}/clip_scores_report.json")


if __name__ == "__main__":
    main()
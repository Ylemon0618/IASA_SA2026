# 이종 멀티모달 재귀 학습에서 초기 데이터 오염 수준에 따른 모델 붕괴 양상 분석

**언어 / Language:** [🇰🇷 한국어 (현재)](README.ko.md) | [🇺🇸 English](README.md)

이 저장소는 이종 멀티모달 재귀 학습 과정에서 초기 데이터 오염의 유형과 수준이 모델 성능에 미치는 영향을 분석한 연구의 코드와 실험 자료를 포함한다.

이미지 생성 모델과 이미지 캡셔닝 모델 사이의 재귀 학습을 대상으로 라벨 노이즈 및 가우시안 노이즈가 세대별 이미지 품질, 다양성, 이미지–텍스트 의미적 정합성에 미치는 영향을 살펴본다.

## 1. 연구 개요

생성형 AI가 대규모 웹 크롤링 데이터에 더욱 의존함에 따라, AI 생성 콘텐츠가 다시 학습 데이터셋에 유입되어 후속 모델에 반복적으로 사용될 수 있다. 이러한 합성 데이터의 재귀적 사용은 원본 데이터 분포의 특성과 다양성이 점차 소실되는 **모델 붕괴(Model Collapse)** 로 이어질 수 있다.

실제 환경에서는 한 AI 모델의 결과물이 다른 모델에 의해 처리된 뒤 학습 자료로 재사용되기도 한다. 본 연구는 이미지 생성 모델과 이미지 캡셔닝 모델을 결합하여 이종 멀티모달 환경을 분석한다.

주요 목적은 초기 학습 데이터의 노이즈 유형과 수준이 재귀 학습 과정에서 모델 성능과 성능 저하 양상에 어떤 영향을 미치는지 조사하는 것이다.

## 2. 모델 및 데이터셋

| 구성 요소 | 모델 / 데이터셋 |
|---|---|
| 이미지 생성 | Stable Diffusion XL Base 1.0 (SDXL) |
| 이미지 캡셔닝 | LLaVA-1.5 7B |
| 초기 데이터셋 | UIUC Pascal Sentence Dataset |
| 초기 이미지–캡션 쌍 | 996개 |
| 세대별 합성 이미지 | 3,000장 |
| 재귀 학습 세대 | Gen 0–Gen 12 |

초기 데이터셋은 UIUC Pascal Sentence Dataset에서 사용한 이미지–캡션 쌍 996개로 구성되며, 각 이미지에는 캡션 1개가 대응한다.

이미지 생성에는 LLM으로 제작한 프롬프트 1,400개를 사용하였다. 원본 데이터셋의 캡션과 유사한 형식으로 작성한 프롬프트를 섞어 3,000개의 생성 입력을 구성했으므로 일부 프롬프트는 반복 사용되었다.

매 세대의 합성 이미지에는 LLaVA-1.5 7B가 새로운 캡션을 생성했고, 이 이미지–캡션 쌍을 다음 세대의 학습 데이터셋으로 구성하였다.

## 3. 실험 설계

주요 조작 변인은 **초기 학습 데이터셋에 주입하는 노이즈의 유형과 수준**이다.

라벨 노이즈와 가우시안 노이즈의 두 실험을 독립적으로 수행했으며, 각 조건은 별도의 재귀 학습을 시작하는 데 사용하였다.

### 3.1 라벨 노이즈

이미지–캡션 쌍 사이의 캡션을 무작위로 교환하여 노이즈를 주입하였다. 원래 이미지 픽셀과 캡션 문장은 유지하면서 이미지와 캡션의 대응 관계만 교란하였다.

| 조건 | 라벨 노이즈 비율 |
|---|---:|
| L0 | 0% |
| L5 | 5% |
| L10 | 10% |
| L15 | 15% |

표의 비율은 목표 오염 수준을 나타낸다. 두 항목씩 캡션을 교환하는 절차 때문에 실제로 교란된 이미지–캡션 쌍의 수는 목표와 약간 다를 수 있다.

### 3.2 가우시안 노이즈

초기 이미지의 RGB 픽셀값에 가우시안 노이즈를 직접 더하였다. 캡션 및 이미지–캡션 대응 관계는 유지하였다.

| 조건 | 표준편차 (σ) |
|---|---:|
| G0 | 0.00 |
| G003 | 0.03 |
| G010 | 0.10 |
| G030 | 0.30 |

정규화된 이미지 공간에서 평균 0과 지정된 표준편차를 갖는 노이즈를 생성하였다. 노이즈를 더한 뒤에는 픽셀값이 유효 범위를 벗어나지 않도록 제한하였다.

## 4. 재귀 학습 파이프라인

재귀 학습 과정은 다음과 같다.

1. **초기 학습(Gen 0):** 설정된 노이즈 조건의 초기 데이터셋으로 SDXL을 파인튜닝한다.
2. **이미지 생성:** 현재 SDXL 모델로 합성 이미지 3,000장을 생성한다.
3. **이미지 캡셔닝:** LLaVA-1.5 7B가 \`Describe this image in detail.\` 프롬프트를 이용하여 합성 이미지의 캡션을 생성한다.
4. **데이터셋 구성:** 생성 이미지와 해당 캡션을 다음 학습 데이터셋으로 묶는다.
5. **재귀 파인튜닝:** 이전 세대 모델의 가중치를 계승하면서 합성 데이터셋으로 다음 세대 SDXL을 파인튜닝한다.
6. **평가:** 생성 이미지를 평가하고 Gen 12까지 과정을 반복한다.

원본 데이터셋은 Gen 0 학습에만 사용한다. Gen 1부터는 직전 세대에서 생성한 합성 이미지와 캡션으로 학습한다.

이 과정에서 Gen 0부터 Gen 12까지 **13개 모델 세대**가 만들어지며, 재귀 학습 단계는 12회이다.

## 5. 학습 설정

실험에 적용한 주요 학습 설정은 다음과 같다.

| 항목 | 설정값 |
|---|---|
| 이미지 해상도 | 1024 × 1024 |
| 학습 방식 | Full Fine-Tuning (FFT) |
| 학습률 | 1e-6 |
| 배치 크기 | 1 |
| 그래디언트 누적 단계 | 4 |
| 유효 배치 크기 | 4 |
| 세대별 학습 단계 | 500 |
| 학습률 스케줄러 | Constant |
| 워밍업 단계 | 0 |
| 최적화 알고리즘 | 8-bit Adam |
| 학습 정밀도 | FP16 |
| 그래디언트 체크포인팅 | 활성화 |
| 난수 시드 | 42 |

실험 조건 간 비교를 위해 주요 학습 설정을 동일하게 유지하였다.

캡셔닝 모델은 LLaVA-1.5 7B (\`llava-hf/llava-1.5-7b-hf\`)를 사용하였다. 캡션 생성에는 NF4 4-bit 양자화, FP16 연산, 신규 토큰 최대 100개 및 결정적 디코딩(\`do_sample=False\`)을 적용하였다.

## 6. 평가 지표

생성 이미지의 품질·다양성 및 이미지–텍스트 의미적 정합성 변화를 분석하기 위해 세 가지 평가 방법을 활용하였다.

| 지표 | 설명 |
|---|---|
| Inception Score (IS) | 사전 학습된 Inception-v3의 클래스 예측을 이용하여 생성 이미지의 품질과 클래스 수준 다양성을 평가 |
| CLIP 기반 이미지–텍스트 코사인 유사도 | CLIP 임베딩으로 생성 이미지와 LLaVA가 작성한 캡션의 의미적 유사성을 측정 |
| PRDC | Precision, Recall, Density, Coverage를 통해 생성 데이터와 실제 데이터의 특징 분포 관계를 평가 |

### 6.1 Inception Score (IS)

사전 학습된 Inception-v3로 IS를 계산하였다. 이미지를 299 × 299로 변환하고 10개 분할(splits)을 사용해 평가하였다. 보고된 표준편차는 **분할 간 변동**을 의미하며, 독립적인 학습 반복 실험 간 변동을 뜻하지 않는다.

### 6.2 CLIP 기반 이미지–텍스트 코사인 유사도

\`openai/clip-vit-large-patch14\` 모델을 사용하여 각 생성 이미지와 해당 이미지에 대해 LLaVA가 작성한 캡션을 비교하였다.

**주의:** 이 지표는 이미지 생성 이후 작성된 캡션과 생성 이미지 사이의 유사도를 측정한다. 최초 이미지 생성 프롬프트와 이미지의 유사도를 평가하는 것이 아니며, 표준 CLIPScore 구현과도 동일하지 않다.

### 6.3 PRDC

PRDC는 다음 네 지표로 실제 데이터와 생성 데이터의 분포 관계를 평가한다.

- **Precision:** 생성 데이터가 실제 데이터의 분포 영역에 얼마나 포함되는지 측정한다.
- **Recall:** 생성 데이터가 실제 데이터의 특징 분포를 얼마나 포괄하는지 측정한다.
- **Density:** 실제 데이터 분포를 기준으로 생성 샘플의 국소적인 밀도를 평가한다.
- **Coverage:** 생성 샘플이 포괄하는 실제 데이터 샘플의 비율을 측정한다.

모든 조건과 세대에서 오염되지 않은 UIUC Pascal 원본 이미지 996장을 고정 기준 데이터로 사용하였다. Inception-v3에서 추출한 2,048차원 특징을 사용하고 최근접 이웃 수는 5로 설정하였다.

## 7. 연구 결과 요약

실험한 노이즈 조건 전반에서 재귀 학습이 진행될수록 평가 지표가 전반적으로 감소하는 양상이 관찰되었다.

### 7.1 라벨 노이즈

라벨 노이즈 수준에 따라 지표별 결과가 달랐다. Gen 12에서는 10% 조건의 Precision과 Density가 가장 높았고, 15% 조건의 Recall이 가장 높았다.

그러나 라벨 노이즈 비율과 모델 성능 사이에 일관된 개선이나 단조로운 관계는 관찰되지 않았다.

### 7.2 가우시안 노이즈

일부 가우시안 노이즈 조건은 초기 세대에서 무노이즈 조건보다 높은 Precision과 Density를 나타냈다.

Gen 12에서는 0.03 조건의 IS, Precision, Recall이 가장 높았고, 0.10 조건의 Density와 Coverage, 0.30 조건의 CLIP 유사도가 각각 가장 높았다.

이는 노이즈 수준이 평가 지표마다 다른 영향을 미칠 수 있음을 보여준다. 모든 지표에서 일관되게 가장 우수한 가우시안 노이즈 조건은 없었다.

### 7.3 종합 결과

두 실험 모두 재귀 학습이 진행되면서 IS와 PRDC가 전반적으로 감소하였다. CLIP 기반 이미지–텍스트 코사인 유사도도 초기 세대의 일시적인 상승 이후 후반 세대에서 감소하는 경향을 보였다.

초기 데이터 오염은 이종 멀티모달 재귀 학습의 성능 저하 양상에 영향을 줄 가능성이 있다. 그러나 특정 노이즈 유형이나 수준이 모델 붕괴를 일관되게 방지하거나 완화한다는 결론을 내릴 수는 없다.

**실험의 한계:** 각 실험 조건에서 독립적인 학습을 1회씩만 수행하였다. 따라서 관찰된 조건 간 차이는 통계적으로 확립된 효과가 아닌 실험상 경향으로 해석해야 한다.

## 8. 설치 및 실행

다음은 실험 환경을 설정하고 학습 파이프라인을 실행하는 기본 절차이다.

### 8.1 저장소 복제

\`\`\`bash
git clone https://github.com/Ylemon0618/multimodal-recursive-learning-varying-data-corruption.git
cd IASA_SA2026
\`\`\`

### 8.2 설정 스크립트 실행 권한 부여

\`\`\`bash
chmod +x setup.sh
\`\`\`

### 8.3 환경 설정

필요한 환경을 설치·설정하기 위해 스크립트를 실행한다.

\`\`\`bash
./setup.sh
\`\`\`

### 8.4 초기 데이터셋 준비

UIUC Pascal Sentence Dataset을 내려받고 이미지–캡션 쌍 996개를 준비한다.

프로젝트 구성에서 요구하는 경로에 이미지와 캡션 메타데이터를 배치한다. 구체적인 디렉터리 및 메타데이터 형식은 데이터 로딩 구현을 확인해야 한다.

### 8.5 초기 데이터셋에 노이즈 주입

다음 스크립트로 노이즈를 주입한다.

- **라벨 노이즈:** \`scripts/data/inject_label_noise.py\` — 이미지와 캡션의 대응 관계를 교란한다.
- **가우시안 노이즈:** \`scripts/data/inject_gaussian_noise.py\` — 초기 이미지 픽셀에 가우시안 노이즈를 더한다.

두 노이즈 유형은 실험 조건별로 독립적으로 적용한다. 실험 간 데이터가 덮어써지지 않도록 조건마다 별도 초기 데이터셋을 준비한다.

### 8.6 메타데이터 생성

학습에 필요한 메타데이터를 생성한다.

\`\`\`bash
python scripts/data/metadata_generator.py
\`\`\`

### 8.7 Full Fine-Tuning 실행

준비한 데이터셋과 해당 실험 설정으로 학습 스크립트를 실행한다.

\`\`\`bash
python scripts/training/fft.py
\`\`\`

정확한 데이터 경로, 스크립트 인수 및 설정 옵션은 실제 구현에 따라 다르므로 관련 스크립트와 설정 파일을 확인해야 한다.

### 8.8 재귀 학습

초기 학습이 끝나면 생성된 모델로 다음 세대를 위한 합성 이미지와 캡션을 생성한다.

이미지 생성, 캡셔닝, 데이터셋 구성, 파인튜닝을 Gen 12까지 반복한다.

세대별 구체적인 실행 명령은 저장소의 세대 관리 스크립트와 설정 파일을 따른다.

## 9. 저장소 구조

이 저장소는 모델 평가, 학습, 데이터 준비 및 노이즈 주입 모듈로 구성된다.

\`\`\`text
IASA_SA2026/
├── configs/
│   └── synthetic_image_prompts.json
├── data/
│   ├── initial/
│   └── processed/
├── outputs/
│   ├── checkpoints/
│   ├── generations/
│   ├── logs/
│   └── metrics/
├── scripts/
│   ├── data/
│   │   ├── data_sampler.py
│   │   ├── inject_gaussian_noise.py
│   │   ├── inject_label_noise.py
│   │   └── metadata_generator.py
│   └── training/
│       ├── fft.py
│       └── fft_pilot.sh
├── src/
│   ├── evaluation/
│   │   ├── clip.py
│   │   ├── is_score.py
│   │   └── prdc_eval.py
│   └── modules/
│       └── measures.py
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── setup.sh
\`\`\`

### 주요 디렉터리

| 디렉터리 | 설명 |
|---|---|
| \`src/modules/\` | 프로젝트에서 공유하는 모듈과 측정 관련 기능 |
| \`src/evaluation/\` | CLIP 유사도, IS, PRDC 평가 스크립트 |

### 주요 파일

| 파일 | 설명 |
|---|---|
| \`scripts/data/data_sampler.py\` | 실험 파이프라인에 사용할 데이터 샘플링 |
| \`scripts/training/fft.py\` | Full Fine-Tuning (FFT) 학습 구현 |
| \`scripts/training/fft_pilot.sh\` | 예비 파인튜닝 실행용 셸 스크립트 |
| \`scripts/data/inject_gaussian_noise.py\` | 초기 이미지에 가우시안 노이즈 주입 |
| \`scripts/data/inject_label_noise.py\` | 이미지와 캡션 간 불일치 생성 |
| \`scripts/data/metadata_generator.py\` | 학습용 메타데이터 생성 |
| \`requirements.txt\` | 프로젝트 Python 의존성 목록 |
| \`setup.sh\` | 실험 환경 설정 |
| \`configs/synthetic_image_prompts.json\` | 합성 이미지 생성용 프롬프트 |

### 평가 스크립트

\`src/evaluation/\`에는 본 연구에서 사용하는 평가 지표의 구현이 포함되어 있다.

- \`clip.py\`: CLIP 기반 이미지–텍스트 코사인 유사도 계산
- \`is.py\`: Inception Score (IS) 계산
- \`prdc.py\`: Precision, Recall, Density, Coverage (PRDC) 계산

\`src/modules/measures.py\`는 측정 관련 공통 기능을 포함한다.

## 10. 감사의 글 및 참고문헌

본 연구는 다음 모델, 데이터셋, 평가 방법을 활용한다.

1. Podell, D. et al. SDXL: Improving latent diffusion models for high-resolution image synthesis. Proc. ICLR (2024)
   https://arxiv.org/abs/2307.01952

2. Liu, H. et al. Improved baselines with visual instruction tuning. Proc. CVPR, 26296–26306 (2024)
   https://arxiv.org/abs/2304.08485

3. UIUC Pascal Sentence Dataset.  
   https://vision.cs.uiuc.edu/pascal-sentences/

4. Radford, A. et al. Learning transferable visual models from natural language supervision. Proc. ICML (PMLR) 139, 8748–8763 (2021)
   https://arxiv.org/abs/2103.00020

5. Naeem, M. F. et al. Reliable fidelity and diversity metrics for generative models. Proc. ICML (PMLR) 119, 7176–7185 (2020)
   https://arxiv.org/abs/2002.09797

6. Shumailov, I. et al. AI models collapse when trained on recursively generated data. Nature 631, 755–759 (2024)
   https://www.nature.com/articles/s41586-024-07566-y

연구보고서에서 다룬 추가 선행연구는 본문 참고문헌을 참조한다.

## 11. 현재 저장소 구조 및 세대 번호 체계

소스 트리는 \`scripts/data/\`, \`scripts/training/\`, \`src/evaluation/\`, \`src/modules/\`로 구성되어 있다. 합성 프롬프트 모음은 \`configs/synthetic_image_prompts.json\`에 저장된다. 저장소 루트에서 \`pip install -r requirements.txt\`로 의존성을 설치하고 \`bash setup.sh\`를 실행한 후 명령을 실행한다.

논문의 **Gen 0–12**는 \`models/gen_0\`부터 \`models/gen_12\`까지의 모델 체크포인트에 직접 대응한다. 첫 파인튜닝으로 Gen 0을 생성하고, Gen 1–12는 재귀적으로 생성한 이미지와 캡션을 이용한다. 13개 체크포인트를 학습하려면 \`GENERATIONS=13\`을 설정한다. Gen N에서 학습을 재개하려면 \`START_GEN=N\`을 사용하며, 직전 완료 체크포인트는 \`models/gen_(N-1)\`이다. 자동 재개 기능은 마지막으로 완료된 체크포인트 다음부터 시작한다. 과거 Gen 1–13 체계로 저장한 체크포인트는 재사용 전에 이름을 변경하거나 마이그레이션해야 한다.

실행 예시:

\`\`\`bash
GENERATIONS=13 python scripts/training/fft.py
START_GEN=6 GENERATIONS=13 python scripts/training/fft.py
python src/evaluation/is_score.py
python src/evaluation/clip.py
python src/evaluation/prdc_eval.py
\`\`\`

**재현성 관련 주의:** \`requirements.txt\`의 의존성 버전은 명시적인 스냅샷이며, 원래 CUDA 실험 환경을 검증하여 재현한 구성은 아니다. \`setup.sh\`가 Git 체크아웃 버전을 설치하면서 Diffusers 버전 고정을 덮어쓸 수 있다. 이 버전 구성에서 실험 결과가 동일하게 재현되는지는 검증되지 않았다.

## 12. 인용 방법

이 저장소를 연구에 활용한다면 다음과 같이 인용한다.

Kim, K., Kim, Y., & Ha, S. (2026).  
*Analyzing Model Collapse under Varying Initial Data Corruption Levels in Heterogeneous Multimodal Recursive Learning.*  
GitHub Repository: https://github.com/Ylemon0618/multimodal-recursive-learning-varying-data-corruption

논문이 학술대회나 학술지에 정식 게재되면 게재 정보를 반영하여 인용 정보를 갱신해야 한다.

## 13. 라이선스

이 저장소에는 MIT License가 적용된다.

MIT License 조건에 따라 소프트웨어를 사용, 복사, 수정, 병합, 게시, 배포, 재라이선스 및 판매할 수 있다.

자세한 내용은 [LICENSE](LICENSE) 파일을 확인한다.

**제3자 자료:** MIT License는 저장소의 원본 코드에만 적용된다. 연구에 사용된 외부 모델, 데이터셋, 라이브러리 및 기타 자료에는 각각의 라이선스와 이용 약관이 적용되므로 사용자가 이를 확인하고 준수해야 한다.

---

이 저장소는 학술 및 연구 목적으로 제공된다.

---

### 생성형 AI 활용 고지

OpenAI의 **ChatGPT**를 저장소 개발 및 문서 작성 보조 도구로 활용하였다. 여기에는 코드 변경 검토 및 제안, 저장소 구조 정리, 세대 번호 체계 수정, README 및 기타 프로젝트 문서의 일부 작성·편집 등이 포함된다.

연구 설계, 구현 결정, 실험 결과, 해석 및 공개 자료의 정확성에 대한 책임은 연구 저자에게 있다. AI가 제안한 내용은 독립적으로 검증된 코드나 실험 결과를 의미하지 않으며, ChatGPT가 실험을 직접 수행하거나 검증했다는 뜻도 아니다.

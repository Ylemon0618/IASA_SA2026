# Information Loss Between Images and Text in Recursive Loops: Identifying the Optimal Cross-Learning Threshold for Maintaining Data Diversity

#### Authors
* **Kim Kyuwan** (Incheon Academy of Science and Arts)
* **Kim Yoonhyeok** (Incheon Academy of Science and Arts)
* **Ha Seong-u** (Incheon Academy of Science and Arts)

## 1. Title & Abstract
* **Project Name**: Information Loss Between Images and Text in Recursive Loops: Identifying the Optimal Cross-Learning Threshold for Maintaining Data Diversity
* **Abstract**: This research establishes a heterogeneous multimodal recursive learning loop consisting of an image generation model (SDXL) and an image captioning model (LLaVA-1.5). By strictly controlling two primary variables—initial dataset error rate and image noise level—this project quantitatively investigates how recursive generation cycles affect information loss, vocabulary diversity contraction, and data distribution shifts. The ultimate goal is to identify the optimal cross-learning threshold required to mitigate and delay catastrophic model collapse in realistic, open-web data cycles.

---

## 2. Research Background & Objectives

### 📌 Problem Statement
With the explosive expansion of generative AI, an increasing portion of internet data is becoming synthetically generated. Recent studies predict a looming depletion of human-generated data relative to the massive training demands of advanced foundation models [1]. Consequently, future AI models will inevitably rely on auto-generated data, creating a closed recursive loop. Recent literature shows that recursive cycles suffer from **"model collapse"**, where data distributions monotonically converge and performance severely degrades due to cumulative statistical distortion [3]. Furthermore, input imperfections like image noise profoundly disrupt optimization boundaries [2], highlighting that data *quality* is just as critical as data *volume*.

### 🧪 Experimental Variables & Matrix
To systematic evaluate the boundaries of model collapse, we control two primary dimensions:

| Variable | Levels / Experimental Values | Operational Definition |
| :--- | :--- | :--- |
| **Initial Dataset Error Rate** | `0%`, `5%`, `10%`, `15%` | Mismatch ratio controlled by randomly swapping image-text pairs within the seed dataset. |
| **Image Noise Level** | `None`, `Low`, `High` | Intensity of deterministic digital noise applied to synthetic outputs prior to training. |

### ❓ Research Questions
1. How do data diversity and structural image quality shift across generations within an inter-model multimodal recursive learning environment?
2. Between the initial label error rate and the post-generation image noise level, which factor acts as a more dominant driver accelerating the onset of model collapse?
3. Does textual vocabulary richness (lexical diversity) and semantic accuracy decline statistically significantly as generation loops progress?

---

## 3. Pipeline Architecture
The system employs an auto-recycling pipeline that models the cross-talk between vision and language domains across discrete generations ($G_0 \rightarrow G_1 \rightarrow \dots \rightarrow G_n$).


```

[Initial Dataset (G0)] ──> [Full Fine-Tuning (FFT)] ──> [Updated SDXL Model]
│
▼
[Next Gen Dataset (G_n+1)] <── [LLaVA-1.5 Captioning] <── [Image Generation]

```

1. **Text-to-Image Synthesis**: The current generation's generative model synthesizes a bulk array of high-resolution images based on a randomized sampling pool.
2. **Image-to-Text Captioning**: A vision-language model (VLM) analyzes the generated images and outputs detailed descriptive captions, transforming visual features back into tokens.
3. **Recursive Re-training**: The newly paired synthetic dataset is fed back to update the base generation weights through recursive Full Fine-Tuning (FFT).

---

## 4. Getting Started & Installation

### 💻 Prerequisites
* **OS**: Linux (Ubuntu 20.04/22.04 LTS recommended)
* **Hardware**: CUDA-enabled GPU (Highly recommended: VRAM $\ge$ 24GB for SDXL/LLaVA operations)
* **Python Version**: $\ge$ 3.10

### 🚀 Installation
Clone the repository and install the required dependencies:
```bash
git clone [https://github.com/Ylemon0618/IASA_SA2026.git](https://github.com/Ylemon0618/IASA_SA2026.git)
cd IASA_SA2026
pip install -r requirements.txt

```

### ⚙️ Environment Variables

Configure your base data root and training parameters by creating a `.env` file in the root directory:

```env
DATASET_PATH="./dataset"
MODEL_PATH="stabilityai/stable-diffusion-xl-base-1.0"
GENERATIONS=10
START_GEN=0

```

---

## 5. Evaluation Metrics

To quantitatively map out the degradation profile and pinpoint the structural tipping point before autophagous model collapse occurs, the pipeline evaluates data across two major dimensions every generation:

### 📸 1. Visual Domain Analysis
* **Fréchet Inception Distance (FID) Score**: Evaluates global feature distribution distance between the generated outputs ($p_g$) and reference human datasets ($p_r$).
  $$d^2((m_r, \Sigma_r), (m_g, \Sigma_g)) = \|m_r - m_g\|_2^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2})$$
* **Per-Category FID Evaluation**: Utilizing `fid_category.py`, we apply caption keyword matching to calculate independent FID values per object class, preventing cross-category blending bias.
* **Inception Score (IS)**: Measures the quality and diversity of the generated images by analyzing the conditional distribution $p(y|x)$ using a pre-trained Inception-v3 network. It ensures that generated images contain clear, identifiable objects (low entropy) while covering a wide range of distinct categories (high entropy).
  $$\text{IS}(G) = \exp(\mathbb{E}_{x \sim p_g} [D_{KL}(p(y|x) \parallel p(y))]$$

### 🔤 2. Multimodal Semantic Alignment Analysis
* **CLIP Score**: Evaluates the semantic directional alignment between the generated images and their corresponding textual captions. By embedding both modalities into a shared latent space via a pre-trained CLIP model, it computes the cosine similarity to directly quantify cross-modal information loss and semantic drift across continuous recursive generations.
  $$\text{CLIP Score}(I, T) = \cos(\mathbf{e}_i, \mathbf{e}_t) = \frac{\mathbf{e}_i \cdot \mathbf{e}_t}{\|\mathbf{e}_i\| \|\mathbf{e}_t\|}$$

---

## 6. References

* **[1]** Pablo Villalobos, Anson Ho, Jaime Sevilla, Tamay Besiroglu, Lennart Heim, and Marius Hobbhahn. "Will we run out of data? Limits of LLM scaling based on human-generated data," *International Conference on Machine Learning (ICML)*, 2024.
* **[2]** S. Y. Lee, S. R. Heo, and W. J. Lee. "A Study on Improving Model Collapse caused by Artificial Intelligence Recursive Learning," *Journal of Software Forensics*, Vol. 20, No. 4, pp. 145-154, 2024.
* **[3]** Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Nicolas Papernot, Ross Anderson, and Yarin Gal. "AI models collapse when trained on recursively generated data," *Nature*, Vol. 631, pp. 755-760, 2024.

```

```

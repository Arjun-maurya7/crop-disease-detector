<p align="center">
  <img src="banner.png" alt="Krishika AI Banner" width="100%">
</p>

# 🌿 Krishika AI — Crop Disease Detector

> AI-powered plant disease diagnosis using VGG19 + SVM across 17 crops — built for real-world agricultural impact.

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-yellow.svg?style=flat-square)](https://huggingface.co/spaces/Arjun-Maurya/krishika-ai)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Gradio](https://img.shields.io/badge/Gradio-6.0%2B-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Crops](https://img.shields.io/badge/Crops_Supported-17-brightgreen?style=flat-square)

🔗 **Live Hugging Face Demo:** [https://huggingface.co/spaces/Arjun-Maurya/krishika-ai](https://huggingface.co/spaces/Arjun-Maurya/krishika-ai)

---

## 🧠 What is Krishika AI?

**Krishika AI** is an end-to-end deep learning system that detects diseases in crop leaves from a single photo. It combines the feature extraction power of **VGG19** (pretrained on ImageNet) with **per-crop SVM classifiers** trained on thousands of real leaf images — achieving up to **99.5% accuracy** across 17 crop types.

The Gradio-powered interface delivers instant, actionable diagnosis reports with:
- 🔬 Biological cause of the disease
- 💊 Recommended treatment / remedy
- 📉 Economic impact on yield
- 🚦 Severity level (Healthy / Monitor / Danger)

> Designed to put precision agriculture in every farmer's hands.

---

## 🌾 Supported Crops & Accuracy

| Crop | Classes | Accuracy |
|------|---------|----------|
| Lychee | 6 | 99.53% |
| Radish | 5 | 99.24% |
| Guava | 7 | 98.60% |
| Walnut | 5 | 97.28% |
| Apple | 3 | ~97% |
| Cherry | 5 | ~96% |
| Grape | 7 | ~95% |
| Potato | 3 | 95.87% |
| Rice | 4 | 92.70% |
| Tomato | 11 | 88.31% |
| Apricot | 3 | — |
| Beans | 4 | — |
| Brinjal | 6 | — |
| Cashew | 3 | — |
| Corn | 4 | — |
| Cucumber | 8 | — |
| Fig | 4 | — |

---

## 🏗️ Architecture

```
Input Image (Leaf Photo)
        │
        ▼
┌──────────────────┐
│   VGG19 (frozen) │  ← Pretrained on ImageNet
│  Feature Extractor│
│  + GAP Layer     │
└────────┬─────────┘
         │  512-dim feature vector
         ▼
┌──────────────────┐
│  Per-Crop SVM    │  ← Trained individually per plant
│  Classifier      │
└────────┬─────────┘
         │
         ▼
  Disease Prediction
  + Remedy Report
```

The pipeline follows a **two-stage transfer learning** approach:
1. **VGG19** acts as a frozen feature backbone — no fine-tuning, just powerful CNN features
2. **SVM** classifiers are trained per crop on the extracted features — fast, lightweight, and highly accurate

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow gradio scikit-learn opencv-python joblib numpy
```

### Project Structure

```
crop-disease-detector/
│
├── crop leaves/          # Zipped plant datasets (one .zip per crop)
├── data/                 # Auto-extracted datasets (generated)
├── output/               # Trained SVM models (.pkl files)
│   ├── tomato/
│   │   └── tomato_model.pkl
│   ├── rice/
│   │   └── rice_model.pkl
│   └── ...
│
├── evaluation_results/   # Per-crop confusion matrices & dashboard
├── Training.ipynb        # Model training pipeline
├── evaluate_models.py    # Per-crop accuracy, precision, recall & F1 evaluation
├── model_performance.html # Interactive model evaluation dashboard
├── app.py                # Production Gradio web app (Krishika AI)
├── Dockerfile            # Container deployment definition
└── README.md
```

### Step 1 — Train the Models

Open `Training.ipynb` in Jupyter and run all cells. Place your crop `.zip` files inside the `crop leaves/` directory first.

```bash
jupyter notebook Training.ipynb
```

Each crop will be trained and saved as `output/<crop>/<crop>_model.pkl`.

### Step 2 — Launch the Interface

```bash
python app.py
```

This will start the Krishika AI Gradio app locally on `http://localhost:7860`.

### Step 3 (Optional) — Run via Docker

```bash
docker build -t krishika-ai .
docker run -p 7860:7860 krishika-ai
```

### Step 4 (Optional) — Run Comprehensive Model Evaluation

```bash
python evaluate_models.py
```

---

## 🖥️ Interface Preview

The **Krishika AI** web UI features:
- 📂 Crop selector (auto-populated from available models)
- 📸 Image upload / webcam capture
- ⚡ One-click deep diagnosis
- 📋 Colour-coded HTML report (green / orange / red)

---

## 🔬 Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | VGG19 (ImageNet weights) |
| Input Size | 224 × 224 px |
| Feature Vector | 512-dim (after GAP) |
| Classifier | SVM (scikit-learn) |
| Max Samples/Class | 2,500 |
| Augmentation | Rotation, Flip, Zoom, Shear, Shift |
| Batch Size | 32 |

**Data Augmentation** is applied during feature extraction to handle class imbalance — especially useful for crops with fewer than 500 images per class (e.g., Fusarium Wilt in Tomato: 318 images).

---

## 📦 Dependencies

```txt
tensorflow>=2.10
gradio>=6.0
scikit-learn
opencv-python
joblib
numpy
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

- 🌱 Add more crops or disease classes
- 💊 Expand the remedy database in `interface.py`
- 🧪 Experiment with other backbones (EfficientNet, ResNet)
- 🌐 Add multilingual support (Hindi, Punjabi, etc.)

```bash
# Fork the repo, then:
git checkout -b feature/your-feature-name
git commit -m "Add: your feature"
git push origin feature/your-feature-name
# Open a Pull Request
```

---

## 📄 License

This project is licensed under the **MIT License** — free to use, modify, and distribute.

---

## 👨‍💻 Author

Built with ❤️ for Indian agriculture.  
If this project helped you, consider giving it a ⭐ on GitHub!

---

*Krishika AI — Smart Agriculture Diagnosis • Powered by VGG19 & SVM*

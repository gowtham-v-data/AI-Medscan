# AI MedScan — Medical Image Intelligence Platform 🏥🤖

<div align="center">

![AI MedScan](https://img.shields.io/badge/AI_MedScan-v2.0-00d4ff?style=for-the-badge&labelColor=050a18)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-43e97b?style=for-the-badge)

**Industry-grade AI-powered chest radiograph analysis using DenseNet121 architecture with Grad-CAM explainability**

[Features](#-features) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [API Docs](#-api-documentation) • [Research](#-research-references)

</div>

---

## 🌟 Features

| Feature | Description |
|---------|-------------|
| 🧠 **DenseNet121 Classification** | CheXNet-inspired multi-label classifier detecting 14 thoracic pathologies |
| 👁️ **Grad-CAM Explainability** | Visual attention maps showing diagnostic regions with anatomical localization |
| 📋 **Clinical Reporting** | Automated DICOM SR-style reports with ICD-10 codes and differential diagnoses |
| 🛡️ **Quality Assessment** | CLAHE-enhanced preprocessing with automated image quality scoring |
| ⚡ **Real-Time Processing** | Sub-2-second analysis with FastAPI async backend |
| 📊 **Risk Stratification** | Automated triage from CRITICAL to NORMAL with severity scoring |

## 🔬 Detectable Pathologies (14)

| # | Pathology | ICD-10 | Urgency |
|---|-----------|--------|---------|
| 1 | Atelectasis | J98.11 | Moderate |
| 2 | Cardiomegaly | I51.7 | Moderate |
| 3 | Consolidation | J18.9 | High |
| 4 | Edema | J81.0 | High |
| 5 | Effusion | J90 | Moderate |
| 6 | Emphysema | J43.9 | Moderate |
| 7 | Fibrosis | J84.10 | Moderate |
| 8 | Hernia | K44.9 | Low |
| 9 | Infiltration | R09.89 | Moderate |
| 10 | Mass | R91.8 | Critical |
| 11 | Nodule | R91.1 | High |
| 12 | Pleural Thickening | J92.9 | Low |
| 13 | Pneumonia | J18.9 | High |
| 14 | Pneumothorax | J93.9 | Critical |

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Input       │────▶│ Preprocess   │────▶│ DenseNet121  │────▶│ Grad-CAM     │
│  Chest X-Ray │     │ CLAHE+Norm   │     │ Classification│     │ Attention    │
│  (JPEG/PNG)  │     │ (224×224×3)  │     │ (14 classes)  │     │ Maps         │
└─────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                      │
                    ┌──────────────┐     ┌──────────────┐            │
                    │  Clinical    │◀────│  Risk        │◀───────────┘
                    │  Report      │     │  Assessment  │
                    │  (ICD-10)    │     │  (Triage)    │
                    └──────────────┘     └──────────────┘
```

### Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **ML Backbone** | TensorFlow + DenseNet121 | Feature extraction & classification |
| **Explainability** | Grad-CAM | Visual attention mapping |
| **Image Processing** | OpenCV + PIL | CLAHE enhancement, preprocessing |
| **Backend API** | FastAPI + Uvicorn | REST API + async processing |
| **Frontend** | HTML5 + CSS3 + JavaScript | Interactive analysis dashboard |
| **Visualization** | Chart.js | Confidence charts & quality radar |
| **Computation** | NumPy + SciPy | Numerical operations |

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip package manager

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/ai-medscan.git
cd ai-medscan

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# OR
.\venv\Scripts\activate         # Windows

# 3. Install dependencies
cd backend
pip install -r requirements.txt

# 4. Start the server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Access

- **Web UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc
- **Health Check**: http://localhost:8000/api/health

## 📡 API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/analyze` | Analyze a medical image |
| `GET` | `/api/health` | System health check |
| `GET` | `/api/model/info` | Model architecture info |
| `GET` | `/api/pathologies` | List all detectable pathologies |
| `GET` | `/api/analysis/{id}` | Retrieve previous analysis |
| `GET` | `/api/stats` | Platform statistics |

### Example: Analyze Image

```bash
curl -X POST "http://localhost:8000/api/analyze?enhance=true&colormap=jet" \
  -H "accept: application/json" \
  -F "file=@chest_xray.jpg"
```

### Response Structure

```json
{
  "analysis_id": "MEDSCAN-A1B2C3D4E5F6",
  "status": "completed",
  "processing_time_ms": 1245.3,
  "quality_assessment": {
    "quality_score": 82.5,
    "quality_grade": "Good"
  },
  "predictions": {
    "findings": [...],
    "assessment": {
      "risk_score": 45.2,
      "status": "BORDERLINE - Follow-up Suggested",
      "status_code": "borderline"
    }
  },
  "heatmaps": {
    "primary": {
      "overlay": "<base64>",
      "raw": "<base64>",
      "contours": "<base64>"
    }
  },
  "clinical_report": {...}
}
```

## 📚 Research References

1. **CheXNet**: Rajpurkar et al., "Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning" (arXiv:1711.05225, 2017)
2. **Grad-CAM**: Selvaraju et al., "Visual Explanations from Deep Networks via Gradient-based Localization" (ICCV 2017)
3. **CheXpert**: Irvin et al., "A Large Chest Radiograph Dataset with Uncertainty Labels" (AAAI 2019)
4. **DenseNet**: Huang et al., "Densely Connected Convolutional Networks" (CVPR 2017)

## 🎓 Training Pipeline

### Download Real NIH Data & Train

```bash
cd backend

# 1. Download real NIH Chest X-Ray14 images (via MedMNIST)
python -m app.training.download_real_data --output_dir ./real_data --max_images 3000

# 2. Train DenseNet121 with two-phase strategy
python -m app.training.train_model --data_dir ./real_data --epochs 15 --batch_size 16 --loss focal

# 3. Model auto-deploys to trained_models/best_model.keras
```

### Training Results (Real NIH Data)

| Metric | Value |
|--------|-------|
| **Dataset** | 4,200 real NIH chest X-rays |
| **Best Val AUC** | 0.664 |
| **Test Mean AUROC** | 0.609 |
| **Top Pathology** | Cardiomegaly (0.789 AUROC) |
| **Training Time** | ~6 hours (CPU) |

## 📁 Project Structure

```
ai-medscan/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI application
│   │   ├── models/
│   │   │   └── classifier.py          # DenseNet121 classifier
│   │   ├── services/
│   │   │   ├── gradcam.py             # Grad-CAM implementation
│   │   │   └── report_generator.py    # Clinical report generation
│   │   ├── utils/
│   │   │   └── preprocessing.py       # Image preprocessing pipeline
│   │   └── training/
│   │       ├── train_model.py         # DenseNet121 training engine
│   │       ├── dataset.py             # Multi-format dataset loader
│   │       ├── evaluate.py            # AUROC evaluation
│   │       ├── prepare_data.py        # Synthetic data generator
│   │       └── download_real_data.py  # Real NIH data downloader
│   ├── trained_models/                # Trained model weights (.keras)
│   ├── real_data/                     # Real NIH training images
│   └── requirements.txt
├── frontend/
│   ├── index.html                     # Main application page
│   ├── css/
│   │   └── styles.css                 # Medical-grade dark UI
│   └── js/
│       └── app.js                     # Frontend application logic
├── sample_images/                     # Sample chest X-rays
├── README.md
└── .gitignore
```

## ⚠️ Disclaimer

> **This application is for research and educational purposes ONLY.**
> It has NOT been validated for clinical use and should NOT be used for actual medical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice. The AI model may produce false positives or false negatives.

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.

---

<div align="center">

Built with ❤️ for advancing healthcare AI

**[⬆ Back to Top](#ai-medscan--medical-image-intelligence-platform-)**

</div>

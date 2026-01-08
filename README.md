# SpineAI-Multimodal

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-92.7%25-brightgreen.svg)

**Multimodal Deep Learning for Spinal Abnormality Detection**

[Paper](link) | [Dataset](link) | [Demo](link) | [Docs](link)

</div>

---

## 🎯 Overview

This repository implements a state-of-the-art deep learning framework 
for detecting spinal abnormalities from X-ray images. Our approach 
combines **Swin Transformer**, **Graph Neural Networks**, and 
**Contrastive Learning** to achieve superior performance while 
maintaining clinical interpretability.

### Key Features

- 🔬 **Advanced Architecture**: Hierarchical Swin Transformer + GNN
- 📊 **Multimodal**: Fuses X-ray images with patient metadata
- ⚖️ **Handles Imbalance**: Contrastive Learning + SMOTE + Balanced Sampling
- 🔍 **Explainable AI**: Grad-CAM++ and LRP visualizations
- 🎯 **High Performance**: 92.7% accuracy, 93.8% AUC-ROC
- 🏥 **Clinically Validated**: Tested on BUU-LSPINE dataset

### Performance Highlights

| Metric | Score |
|--------|-------|
| Accuracy | 92.7% |
| Precision | 91.5% |
| Recall | 91.0% |
| F1-Score | 91.2% |
| AUC-ROC | 93.8% |
| MCC | 88.9% |

---


## 🚀 Quick Start
```bash
# Clone repository
git clone https://github.com/aiosmartlab/SpineAI-Multimodal.git
cd SpineAI-Multimodal

# Install dependencies
pip install -r requirements.txt

# Train model
python main.py --config config.yaml

# Run inference
python inference.py --image path/to/xray.jpg --age 45 --gender male
```

---
This article is archived on Zenodo.
DOI: 10.5281/zenodo.18182945 




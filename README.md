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
git clone https://github.com/your-username/SpineAI-Multimodal.git
cd SpineAI-Multimodal

# Install dependencies
pip install -r requirements.txt

# Train model
python main.py --config config.yaml

# Run inference
python inference.py --image path/to/xray.jpg --age 45 --gender male
```

---

## 📖 Documentation

- [Installation Guide](docs/installation.md)
- [Dataset Preparation](docs/dataset.md)
- [Training Guide](docs/training.md)
- [API Reference](docs/api.md)

---

## 🏗️ Architecture

Our framework consists of three main components:

1. **Swin Transformer**: Hierarchical feature extraction from X-ray images
2. **Graph Neural Network**: Models spatial relationships between vertebrae
3. **Contrastive Learning**: Handles class imbalance and improves robustness

![Architecture Diagram](assets/architecture.png)

---

## 📊 Results

### Classification Performance

![Confusion Matrix](assets/confusion_matrix.png)

### Explainability

![Grad-CAM Example](assets/gradcam_example.png)

---

## 🔬 Citation

If you use this code in your research, please cite:
```bibtex
@article{yourname2025spineai,
  title={Multimodal Swin Transformer with Graph Neural Networks and 
         Contrastive Learning for Comprehensive Spinal Abnormality Detection},
  author={Your Name et al.},
  journal={Journal Name},
  year={2025}
}
```

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) 
for guidelines.

---

## 📧 Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **Institution**: Your University/Hospital

---

## 🙏 Acknowledgements

- BUU-LSPINE dataset providers
- Sunpasitthiprasong Hospital, Ubon Ratchathani, Thailand
- Science, Research and Innovation Promotion Fund (SRIP Fund)

</div>
```

---

## 🎨 GitHub Topics/Tags ที่ควรใส่
```
deep-learning
medical-imaging
computer-vision
spinal-detection
swin-transformer
graph-neural-networks
contrastive-learning
explainable-ai
medical-ai
pytorch
radiology
healthcare-ai
medical-diagnosis
x-ray-analysis
vertebrae-detection
clinical-ai
```

---

## 💡 Tips สำหรับ Description ที่ดี

### ✅ ควรมี:
1. **Purpose** - ทำอะไร
2. **Technology** - ใช้เทคโนโลยีอะไร
3. **Performance** - ผลลัพธ์เป็นอย่างไร
4. **Unique Value** - จุดเด่นคืออะไร

### ✅ Keywords ที่ควรใส่:
- Spinal abnormality
- Detection
- Deep learning / AI
- Swin Transformer
- Graph Neural Network
- Contrastive Learning
- Explainable AI
- Medical imaging
- X-ray
- PyTorch

### ❌ ควรหลีกเลี่ยง:
- ประโยคยาวเกินไป
- ศัพท์เทคนิคมากเกินไปโดยไม่อธิบาย
- ไม่มี keywords สำคัญ
- ไม่บอกว่าทำอะไร

---

## 🎯 คำแนะนำสุดท้าย

**สำหรับ Description สั้นใน About Section:**
```
AI-powered spinal abnormality detection using Swin Transformer, 
Graph Neural Networks, and Contrastive Learning. Achieves 92.7% 
accuracy with explainable predictions for clinical use.

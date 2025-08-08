# Multi-Modal Multi-Task Federated Foundation Model for Embodied AI:

This repository contains the official implementation for **Multi-Modal Multi-Task (M3T) Federated Foundation Models for Embodied AI: Potentials and Challenges for Edge Integration**.


Key features:

- Multi-modal support via pretrained ViLT backbone
- Personalized Multi-task classification heads
- Adapter-based personalization with low memory usage
- FL orchestration with peer-to-peer (P2P) aggregation

## 🌐 Network Model

The M3T FFM system's network model is inspired by and extends the following works:

- (https://ieeexplore.ieee.org/abstract/document/9705093): Multi-Stage Hybrid Federated Learning Over Large-Scale D2D-Enabled Fog Networks
- (https://ieeexplore.ieee.org/document/10304380): Delay-Aware Hierarchical Federated Learning
- (https://arxiv.org/abs/2404.06324): Dynamic D2D-Assisted Federated Learning over O-RAN: Performance Analysis, MAC Scheduler, and Asymmetric User Selection
- (https://ieeexplore.ieee.org/document/9148862): Client-Edge-Cloud Hierarchical Federated Learning

## 📁 Project Structure

```
HFFM/
├── core/
│   ├── datasets.py              # Dataset loading and preprocessing
│   ├── models.py                # Adapter-based model definitions
│   ├── network.py               # Communication & aggregation logic
│   └── utils.py                 # Helper functions
│
├── datasets/
│   ├── dataset_generator_*.py   # Scripts to generate balanced datasets
│   ├── *_vocab_balanced.py      # Balanced vocab files for tasks (ART, GQA, VizWiz)
│
├── methods/
│   ├── main_ffm.py        # Entry point for hierarchical FL
│   ├── main_local.py            # Entry point for local-only training
│   ├── table_generator.py       # Summarize evaluation metrics
│   └── results/                 # Folder to store results and logs
```

## 📦 Installation

```bash
git clone https://github.com/payamsiabd/M3T-FFM-EmbodiedAI.git
```

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Ensure PyTorch with GPU support is installed and datasets are available locally.

### 2. Prepare Datasets
The experiments in this project are conducted on two Visual Question Answering (VQA) datasets:

- **VizWiz**: [VizWiz Dataset](https://vizwiz.org/tasks-and-datasets/vqa/)
- **GQA**: [GQA Dataset](https://cs.stanford.edu/people/dorarad/gqa/download.html)
- **Anotation files**: [Anotations](https://drive.google.com/drive/folders/1wlx22Y8KGPmrRFELj2su8CSU5dM0Iu_1?usp=drive_link)
  
Preprocess datasets:

```bash
python datasets/dataset_generator_gqa_balanced.py
python datasets/dataset_generator_vizwiz_balanced.py
python datasets/gqa_vocab_balanced.py
python datasets/vizwiz_vocab_balanced.py
```

### 3. Run Training

#### Local FL (no aggregation):
```bash
python methods/main_local.py
```

#### M3T FFL:
```bash
python methods/main_ffm.py
```

## 📄 License

This project is released under the MIT License.

## 👤 Author

**Payam Abdisarabshali** – Ph.D. Student, Electrical Engineering, The State University of New York at Buffalo

For questions or collaborations, feel free to contact via GitHub or [Google Scholar](https://scholar.google.com/citations?user=ksQpR00AAAAJ&hl=en).


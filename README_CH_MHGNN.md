# Cross-Heterogeneous Meta-Heterogeneous Graph Neural Network  
*A Few-Shot Learning Framework for Disease Prediction*

---

## Overview

This repository contains the implementation of **CH-MHGNN**, a graph-based meta-learning framework for few-shot disease prediction in biomedical domains.  
Unlike traditional models that learn from a single homogeneous dataset, CH-MHGNN is designed to **learn from multiple heterogeneous graphs**—each representing a different biomedical domain (clinical, molecular, pharmacological, etc.)—and transfer knowledge across them.

In simpler terms: the model learns *how to learn* disease patterns, even when labeled data are scarce.

---

##  Dataset Description

The dataset used here consists of **five heterogeneous graphs**, each modeling a distinct biomedical domain:

| Graph ID| Domain                     | Node Types                  | Edge Types                  |
|---------|----------------------------|-----------------------------|-----------------------------|
| G1      | Patient–Symptom–Disease    | patient, symptom, disease   | has_symptom, diagnosed_with |
| G2      | Gene–Protein–Pathway       | gene, protein, pathway      | encodes, involved_in        |
| G3      | Drug–Target–Disease        | drug, target, disease       | binds_to, treats            |
| G4      | Treatment–Outcome–Disease  | treatment, outcome, disease | leads_to, associated_with   |
| G5      | Microbe–Metabolite–Disease | microbe, metabolite, disease| produces, affects           |

Each node is associated with an **8-dimensional feature vector**, and graphs contain around **300 nodes** and **800 edges** each.  
If the dataset file (`cross_hetero_dataset.json`) is not found, it will be **auto-generated** synthetically during runtime.

---

## Training Workflow

The training follows a **meta-learning pipeline** inspired by MAML and Prototypical Networks.

### 1. Environment Setup
```bash
pip install dgl torch torchvision torchaudio tqdm
```

### 2. Open in Google Colab
You can run the full training notebook:
```
ch_mhgnn.ipynb
```
It will:
- Load or generate the heterogeneous graphs  
- Build DGL-based graph objects  
- Initialize the CH-MHGNN encoder  
- Train the model using few-shot episodes (support/query sets)  
- Evaluate cross-domain generalization on unseen graphs

### 3. Training Details
- **Inner loop:** few-shot adaptation on the support set  
- **Outer loop:** meta-optimization using the query set  
- **Loss:** classification + cross-graph alignment (for transferable learning)  
- **Optimizer:** AdamW  
- **Framework:** PyTorch + DGL  

---

## Few-Shot Setup

For each domain graph:
- **Support set:** K = 3 labeled samples per class  
- **Query set:** 15–20 unlabeled samples for evaluation  
- **Classes per graph:** 5 (pseudo-labels for synthetic data)

Splits:
- Meta-Train → G1, G2, G3  
- Meta-Validation → G4  
- Meta-Test → G5  

This structure allows the model to generalize to new biomedical domains or unseen disease types.

---

## Output

During training:
```
[Epoch 5]   Train Loss: 0.642 | Train Acc: 0.77 | Val Loss: 0.613 | Val Acc: 0.81
[META-TEST] Loss: 0.579       | Acc: 0.84
```
These metrics report **few-shot classification accuracy** averaged over multiple meta-tasks.

---

##  Customization

You can modify:
- `K_SHOT` – number of labeled samples per class  
- `EPOCHS`, `LR`, and hidden dimensions for performance tuning  
- Replace the random feature vectors with **real biomedical embeddings** (e.g., from Gene2Vec, ProtBERT, or UMLS).  


### Notes
This repository is intended for educational and experimental research use.  
For real biomedical applications, please replace the synthetic dataset with validated biomedical graph data.

---

# 🤖 AI Assistant Usage in Student Life — BPNN Optimizer Comparison
### ISP560 — Machine Learning | UiTM Shah Alam

> A Backpropagation Neural Network (BPNN) classification model that predicts whether a student will reuse an AI assistant, comparing the performance of 7 different optimizers on a 10,000-record dataset.

---

## 🎯 Objectives

1. Build a BPNN model to classify whether a student will reuse an AI assistant
2. Train and evaluate the model using multiple optimizers and activation functions
3. Compare optimizer performance to identify the most accurate model

---

## 🗂️ Dataset

| Info | Details |
|---|---|
| **Source** | Kaggle — [AI Assistant Usage in Student Life (Synthetic)](https://www.kaggle.com/datasets/ayeshasal89/ai-assistant-usage-in-student-life) by Ayesha Saleem |
| **Records** | 10,000 synthetic entries |
| **Train / Test** | 8,000 / 2,000 |
| **Target** | `UsedAgain` (0 = Will Not Use Again, 1 = Will Use Again) |

**Key Features:** `StudentLevel`, `Discipline`, `SessionLength`, `TotalPrompts`, `TaskType`, `AI_AssistanceLevel`, `SatisfactionRating`, `FinalOutcome`

---

## 🏗️ Model Architecture (BPNN)

```
Input Layer  →  Dense(64, ReLU)  →  Dropout(0.3)
             →  Dense(32, ReLU)  →  Dropout(0.3)
             →  Dense(1, Sigmoid)
```

| Parameter | Value |
|---|---|
| Loss Function | Focal Loss (γ=2, α=0.85) |
| Epochs | 50 |
| Batch Size | 32 |
| Class Weights | {0: 1.70, 1: 0.70} |
| Imbalance Handling | SMOTE |

---

## ⚙️ Optimizers Compared

| Optimizer | Description |
|---|---|
| **Adam** | Adaptive learning rate — combines momentum + RMSprop |
| **AdamW** | Adam with weight decay regularization |
| **RMSprop** | Adaptive learning rate for RNNs / noisy data |
| **Adagrad** | Adapts learning rate per parameter |
| **Nadam** | Adam + Nesterov momentum |
| **Adadelta** | Extension of Adagrad, avoids learning rate decay |
| **SGD** | Classic stochastic gradient descent |

---

## 👥 Team Contributions

Each member ran experiments with different optimizer combinations:

| Member | Optimizers Tested |
|---|---|
| Muhammad Imran Bin Roszaide | Adam, AdamW, RMSprop (ReLU) |
| Muhammad Khairul Izzat Bin Khalid | Adam, AdamW, RMSprop (ELU) |
| Nazif Muhsin Bin Mohd Zahir | Adagrad, Nadam, Adadelta (ReLU) |
| Nik Mohamad Aniq Irfan Bin Nik Mahadi | Adagrad, Nadam, Adadelta (ELU) |
| Muhammad Ameer Mubaraq Bin Adnan | SGD, Nadam (ReLU & ELU) |

---

## 🚀 How to Run

**Requirements:**
```bash
pip install tensorflow scikit-learn imbalanced-learn pandas numpy matplotlib
```

**Run:**
```bash
python ai_assistant_bpnn.py
```

Make sure `ai_assistant_usage_student_life.csv` is in the same directory.

**Output files generated:**
- `confusion_matrix.png` — Best model confusion matrix
- `training_curves.png` — Accuracy & loss curves for all optimizers
- `accuracy_comparison.png` — Bar chart comparing all optimizer accuracies

---

## 📁 Project Files

| File | Description |
|---|---|
| `ai_assistant_bpnn.py` | Combined main script (all 5 members' experiments) |
| `ai_assistant_usage_student_life.csv` | Dataset |
| `ISP560_MINI_PROJECT_REPORT.pdf` | Full project report |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=black)

---

## 👤 My Profile

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/muhammad-khairul-izzat-khalid-8086622a1/)
[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:khairulizzat0412@gmail.com)

**Lecturer:** Sir Mohd Razif Bin Shamsuddin

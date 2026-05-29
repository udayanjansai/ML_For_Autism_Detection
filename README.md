# 🧠 ML for Autism Spectrum Disorder (ASD) Detection

A machine learning and deep learning project for early detection of Autism Spectrum Disorder (ASD) using ensemble classifiers and a TabTransformer deep learning model across multiple age groups.

---

## 📌 Project Overview

This project combines four ASD screening datasets (Child, Adult, Adolescent, and Toddler) into a unified pipeline that applies rigorous preprocessing, balanced resampling, and trains five ensemble ML models alongside a TabTransformer deep learning model. The goal is to build a robust, generalizable ASD classifier that works across age groups.

---

## 📂 Repository Structure

```
ML_For_Autism_Detection/
│
├── Autism_Child_Data.csv          # ASD screening data for children
├── Autism_Adult_Data.csv          # ASD screening data for adults
├── Autism_Adolescent_Data.csv     # ASD screening data for adolescents
├── Autism_Toddler_Data.csv        # ASD screening data for toddlers
├── ML_for_ASD.ipynb               # Main Jupyter Notebook (full pipeline)
└── README.md                      # Project documentation
```

---

## 📊 Datasets

| Dataset     | Source Group  | Target Column        |
|-------------|--------------|----------------------|
| Child        | Children      | `Class/ASD`          |
| Adult        | Adults        | `Class/ASD`          |
| Adolescent   | Adolescents   | `Class/ASD`          |
| Toddler      | Toddlers      | `Class/ASD Traits`   |

All four datasets are harmonized to a **common 16-feature schema**:

`A1_Score` through `A10_Score`, `age`, `gender`, `ethnicity`, `jundice`, `austim`, `Class/ASD`

---

## ⚙️ Pipeline

### 1. Data Loading & EDA
- Load all four CSV files
- Inspect shapes, column names, and class distributions
- Visualize class distribution across datasets

### 2. Preprocessing
- Rename and align columns across datasets (especially Toddler)
- Convert `Age_Mons → age` (in years) for Toddler dataset
- Standardize target labels to `YES / NO → 1 / 0`
- Handle missing values and `?` entries
- Label encode categorical features (`gender`, `ethnicity`, `jundice`, `austim`)
- Visualize age distribution and feature correlation heatmap

### 3. Merging & Resampling
- Concatenate all four datasets with a `source` tag
- Stratified 80/20 train-test split
- Per-source bootstrap resampling:
  - **ML models**: 500 samples per source → 2,000 training rows
  - **DL model**: 2,500 samples per source → 10,000 training rows

### 4. ML Model Training
Five ensemble classifiers are trained and evaluated:

| Model          | Strategy                          |
|----------------|-----------------------------------|
| Random Forest  | 200 trees, default depth          |
| Extra Trees    | 200 trees, default depth          |
| AdaBoost       | 200 stumps, learning rate = 0.5   |
| XGBoost        | GridSearchCV tuned (5-fold CV)    |
| LightGBM       | GridSearchCV tuned (5-fold CV)    |

### 5. Deep Learning — TabTransformer
- Built with [`pytorch-tabular`](https://github.com/manujosephv/pytorch_tabular)
- Architecture: 8 attention heads, 6 transformer blocks, dropout = 0.1
- Optimizer: Adam with ReduceLROnPlateau scheduler
- 50 epochs with early stopping on validation loss
- Evaluated with 5-fold cross-validation

---

## 📈 Visualizations Generated

| File | Description |
|------|-------------|
| `fig1_class_distribution.png` | Bar charts of ASD vs No-ASD per dataset |
| `fig2_age_distribution.png` | KDE plot of age distribution by class |
| `fig3_correlation_heatmap.png` | Feature correlation heatmap |
| `fig_original_vs_resampled_ml_dl.png` | Original vs resampled training sizes |
| `fig_rf_et_cm.png` | Confusion matrices — RF & Extra Trees |
| `fig6_boosting_cm.png` | Confusion matrices — AdaBoost, XGBoost, LightGBM |
| `cm_tabtransformer.png` | Confusion matrix — TabTransformer |
| `fig8_model_comparison.png` | Accuracy & F1 comparison across all models |
| `fig9_cv_comparison.png` | 5-fold CV accuracy with error bars |
| `best_params_heatmap.png` | Best hyperparameter heatmap across models |

---

## 🧪 Evaluation Metrics

Each model is evaluated on:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC** (TabTransformer)
- **5-Fold Cross-Validation** (mean ± std)

---

## 🛠️ Tech Stack

| Category         | Tools / Libraries                                   |
|------------------|-----------------------------------------------------|
| Language         | Python 3.x                                          |
| Environment      | Google Colab                                        |
| Data             | pandas, numpy                                       |
| Visualization    | matplotlib, seaborn                                 |
| ML Models        | scikit-learn, xgboost, lightgbm                     |
| Deep Learning    | pytorch-tabular (TabTransformer)                    |
| Preprocessing    | LabelEncoder, train_test_split, resample            |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/udayanjansai/ML_For_Autism_Detection.git
cd ML_For_Autism_Detection
```

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm pytorch-tabular
```

### 3. Run the Notebook

Open `ML_for_ASD.ipynb` in **Google Colab** (recommended) or Jupyter Notebook.

> **Note:** If running on Colab, mount your Google Drive and place all four CSV files at:
> `/content/drive/MyDrive/datasets/`

---

## 📋 Features Used

| Feature       | Type        | Description                              |
|---------------|-------------|------------------------------------------|
| A1–A10 Score  | Numerical   | AQ-10 screening questionnaire responses  |
| age           | Numerical   | Age in years                             |
| gender        | Categorical | Male / Female                            |
| ethnicity     | Categorical | Ethnic background                        |
| jundice       | Categorical | Jaundice at birth (yes/no)               |
| austim        | Categorical | Family member with ASD (yes/no)          |
| Class/ASD     | Target      | ASD diagnosis (1 = Yes, 0 = No)          |

---

## 👤 Author

**Uday Anjan Sai**
B.Tech – Computer Science & Design (CSD-C)
Sri Indu College of Engineering and Technology
Roll No: 23B81A67J4

---

## 📄 License

This project is intended for academic and research purposes.

---

## 🙏 Acknowledgements

- UCI Machine Learning Repository — ASD Screening Datasets
- [`pytorch-tabular`](https://github.com/manujosephv/pytorch_tabular) by Manu Joseph
- Scikit-learn, XGBoost, and LightGBM communities

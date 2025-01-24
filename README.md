# Predicting Wine Quality Using Machine Learning

This project focuses on predicting the quality of red wine (`quality`) based on chemical properties. The analysis spans exploratory data analysis, feature engineering, and machine learning model development. The project evaluates multiple models, selects the best-performing ones, and extracts actionable insights for understanding wine quality.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset Description](#dataset-description)
3. [Workflow](#workflow)
4. [Results](#results)
5. [How to Run the Project](#how-to-run-the-project)
6. [Future Work](#future-work)
7. [Acknowledgments](#acknowledgments)
8. [Contact](#contact)

---

## Introduction

Wine quality prediction is a challenging multi-class classification problem with applications in the wine industry. This project leverages machine learning to predict wine quality based on physicochemical properties such as acidity, alcohol content, and sugar levels.

The analysis identifies the most influential features and builds interpretable, high-performing models to predict wine quality while addressing challenges like class imbalance and feature representation.

---

## Dataset Description

The dataset contains data for red wines from the Portuguese "Vinho Verde" region. It includes physicochemical features and a quality score assigned by wine tasters. The target variable (`quality`) ranges from 3 to 8, representing different levels of wine quality.

**Key Statistics**:
- **Total records**: 1,599
- **Number of features**: 12 (11 input features + 1 target variable)
- **Feature examples**:
  - `fixed_acidity`, `volatile_acidity`, `alcohol`
- **Target variable**: `quality` (multiclass: 3–8)

---

## Workflow

1. **Problem Understanding**:
   - The goal was to predict wine quality based on physicochemical properties using multi-class classification.

2. **Exploratory Data Analysis (EDA)**:
   - Investigated feature distributions and relationships with wine quality.
   - Identified Alcohol, Sulphates, and Volatile Acidity as key predictors.

3. **Data Preprocessing**:
   - Addressed class imbalance, explored scaling impacts, and ensured data quality.
   - Standardized data selectively for models like Support Vector Machines (SVM).

4. **Baseline Model Performance**:
   - Evaluated a variety of models, including Logistic Regression, Random Forest, and Neural Networks, to establish baseline metrics.

5. **Hyperparameter Tuning**:
   - Optimized key models, including Random Forest and Gradient Boosting algorithms, to enhance performance.

6. **Feature Importance Analysis**:
   - Used the Random Forest model to rank feature importance and identify the top contributors to wine quality.

7. **Final Evaluation**:
   - Validated top models on the test set to confirm generalizability.

---

## Results

**Top-Performing Models**:
1. **Random Forest**:
   - Untuned Accuracy: 62.5% (highest overall performance).
   - Tuning resulted in minor performance changes, highlighting robust default behavior.
2. **Extra Trees**:
   - Untuned Accuracy: 62.1%, making it another strong contender.
3. **LightGBM**:
   - Untuned Accuracy: 61.0%, though tuning led to a slight decline.

**Feature Importance**:
- The top predictors of wine quality identified by the Random Forest model were:
  1. **Alcohol**: 15.03% importance.
  2. **Sulphates**: 11.29% importance.
  3. **Volatile Acidity**: 10.54% importance.

**Key Observations**:
- Distance-based models (e.g., SVM) required scaling for optimal performance, whereas tree-based models performed well without it.
- Class imbalance limited the accuracy for minority classes (e.g., quality scores of 3 and 8).

---

## How to Run the Project

To reproduce this analysis, follow the steps below:

```bash
# Step 1: Clone the repository
git clone <https://github.com/Devin-Shrode/Wine-Quality>

# Step 2: Navigate to the project directory
cd Wine-Quality

# Step 3: Install the required libraries
pip install -r requirements.txt

# Step 4: Open the Jupyter Notebook
jupyter notebook Wine_Quality_Prediction_Final.ipynb

# Step 5: Run the notebook cells sequentially to reproduce the analysis and results.

```

---

## Future Work

1. **Expand Dataset**:
   - Include additional wine types (e.g., white wines) and data from other regions for broader applicability.
2. **Address Class Imbalance**:
   - Implement advanced techniques like SMOTE or ensemble methods tailored for imbalanced data.
3. **Explore Additional Models**:
   - Investigate the use of AutoML frameworks or deep learning models for enhanced predictive performance.

---

## Acknowledgments

- **Dataset**: Sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality).
- **Tools**: Analysis conducted using Python and libraries such as Pandas, Scikit-learn, and Matplotlib.

---

## Contact

For questions or collaboration, feel free to reach out:
- **Email**: devin.shrode@proton.me
- **LinkedIn**: [linkedin.com/in/DevinShrode](https://www.linkedin.com/in/DevinShrode)

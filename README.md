# Predicting the Quality of Red Wine Using Machine Learning

This project focuses on predicting the quality of red wine based on various chemical properties. Through **exploratory data analysis (EDA), feature selection, machine learning model comparison, and hyperparameter tuning**, we identify the best-performing model for wine classification. Additionally, we deploy the final model using **FastAPI** and **Docker** for real-world inference.

---

## **Table of Contents**
1. [Introduction](#introduction)  
2. [Dataset Description](#dataset-description)  
3. [Project Workflow](#project-workflow)  
4. [Results](#results)  
5. [Installation & Running the Project](#installation--running-the-project)  
6. [Future Enhancements](#future-enhancements)  
7. [Acknowledgments](#acknowledgments)  
8. [Contact](#contact)  

---

## **Introduction**
Wine quality is influenced by a combination of **chemical properties** such as acidity, sugar content, and alcohol levels. This project leverages machine learning to build a classification model that predicts wine quality based on these characteristics.

### **Objectives**
- Perform **data preprocessing and feature engineering** to enhance model performance.
- Explore different **machine learning algorithms** and select the most effective model.
- Deploy the trained model as a **FastAPI web service** for real-time predictions.

---

## **Dataset Description**
The dataset used for this project comes from the **UCI Machine Learning Repository** and contains **1,599 instances of red wine** samples, each with **11 physicochemical attributes** and a **quality rating** (3–8). The key attributes include:

- **Fixed Acidity**: The amount of non-volatile acids.
- **Volatile Acidity**: The amount of acetic acid (excess leads to an unpleasant vinegar taste).
- **Citric Acid**: Enhances freshness and taste.
- **Residual Sugar**: The sugar remaining after fermentation.
- **Chlorides**: Salt content affecting wine preservation.
- **Free Sulfur Dioxide** & **Total Sulfur Dioxide**: Important preservatives affecting freshness.
- **Density**: Determines alcohol and sugar content balance.
- **pH**: Affects taste and preservation.
- **Sulphates**: Contributes to aroma and microbial stability.
- **Alcohol**: Influences body and flavor.

The **target variable** represents wine quality, categorized as **low (3–4), medium (5–6), and high (7–8).**

---

## **Project Workflow**

### **1. Exploratory Data Analysis (EDA)**
- Analyzed feature distributions and correlations to detect key relationships.
- Identified **alcohol, volatile acidity, and sulphates** as strong quality indicators.

### **2. Data Preprocessing**
- Handled missing values (if any) and **scaled numerical features where necessary**.
- Verified that tree-based models (e.g., **Random Forest, LightGBM**) did not require feature scaling.

### **3. Model Selection & Hyperparameter Tuning**
- Compared multiple models:
  - **Baseline Models**: Logistic Regression, k-Nearest Neighbors.
  - **Ensemble Models**: Random Forest, AdaBoost, Extra Trees.
  - **Gradient Boosting Models**: XGBoost, LightGBM, CatBoost.
- Tuned hyperparameters using **GridSearchCV & RandomizedSearchCV**.
- **LightGBM emerged as the best model** in terms of accuracy and robustness.

### **4. Feature Importance Analysis**
- Compared feature importance rankings across models.
- Used **SHAP values** to validate model interpretation and investigate misclassifications.

### **5. Model Deployment**
- Built a **FastAPI application** to serve predictions via an API.
- Containerized the model using **Docker** for scalability and portability.

---

## **Results**

### **Best Model: LightGBM**
| Metric | Score |
|--------|-------|
| **Accuracy** | **85.1%** |
| **Weighted F1-Score** | **0.848** |
| **ROC-AUC Score** | **0.972** |

### **Key Findings**
- **Alcohol is the strongest predictor** of wine quality.
- **Volatile acidity negatively correlates** with quality, confirming its impact on taste.
- **SHAP analysis highlighted feature interactions**, revealing why certain misclassifications occurred.

---

## **Installation & Running the Project**

### **1. Clone the Repository**
``` bash
git clone https://github.com/Devin-Shrode/Wine-Quality
cd Wine-Quality
```

### **2. Set Up Virtual Environment**
``` bash
python -m venv wine_env
source wine_env/bin/activate  # For macOS/Linux
wine_env\Scripts\activate     # For Windows
```

### **3. Install Dependencies**
``` bash
pip install -r requirements.txt
```

### **4. Run the FastAPI Server**
``` bash
uvicorn wine_api:app --reload
```

### **5. Test the API**
- Open a browser and navigate to: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**
- Use **Postman** or **cURL** to send a POST request:
``` bash
curl -X 'POST' 'http://127.0.0.1:8000/predict' -H 'Content-Type: application/json' -d '{"features": [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]}'
```

- Expected response:
``` json
{"predicted_quality": 5}
```

---

## **Future Enhancements**
While the current model performs well, potential improvements include:
1. **Expanding the dataset** to include more wine varieties.
2. **Feature engineering** to create synthetic variables that capture more complex relationships.
3. **Deploying on the cloud** via **AWS Lambda or Google Cloud Run** for real-time use.
4. **Building a front-end application** to allow users to upload wine data and receive quality predictions visually.

---

## **Acknowledgments**
- **Dataset**: Sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality).
- **Libraries Used**: **Scikit-learn, LightGBM, SHAP, Pandas, Matplotlib, FastAPI, Docker**.
- **Deployment Tools**: **FastAPI & Docker for scalable model hosting**.

---

## **Contact**
For any questions or collaboration opportunities, reach out at:
- **Email**: devin.shrode@proton.me  
- **LinkedIn**: [linkedin.com/in/DevinShrode](https://www.linkedin.com/in/DevinShrode)  
- **GitHub**: [github.com/Devin-Shrode/Wine-Quality](https://github.com/Devin-Shrode/Wine-Quality)  

---

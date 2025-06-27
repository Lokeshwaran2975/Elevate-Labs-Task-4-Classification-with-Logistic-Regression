# Elevate-Labs-Task-4
# Classification with Logistic Regression 

This project is part of a machine learning internship task focused on building a binary classifier using Logistic Regression. The goal is to classify tumors as **Malignant (M)** or **Benign (B)** based on the **Breast Cancer Wisconsin Diagnostic Dataset**.

## 📌 Objective

Build a binary classification model using Logistic Regression that can predict whether a tumor is malignant or benign based on features extracted from digitized images of breast masses.


## 🛠️ Tools and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn (sklearn)


## 📂 Dataset

- **Source**: Breast Cancer Wisconsin Diagnostic Dataset
- **Target Variable**: `diagnosis` (`M` = Malignant, `B` = Benign)
- **Features**: 30 numerical features computed from images (e.g., radius_mean, texture_mean, etc.)


## ✅ Steps Performed

### 1. Data Preprocessing
- Dropped unnecessary columns (`id`, `Unnamed: 32`)
- Encoded target labels (`M` → 1, `B` → 0)
- Scaled features using `StandardScaler`

### 2. Model Building
- Split data into training and testing sets (80-20 split)
- Trained a Logistic Regression model on training data

### 3. Model Evaluation
- Evaluated using:
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-score)
  - ROC Curve and AUC Score

### 4. Threshold Tuning
- Demonstrated how changing the decision threshold affects model predictions
- Explained use of the **Sigmoid Function** in Logistic Regression


## 📊 Results

- **Accuracy**: ~97%
- **AUC Score**: High, indicating strong separability between classes
- ROC Curve and confusion matrix visualizations confirm strong performance


## 📈 Evaluation Metrics

| Metric     | Description                                      |
|------------|--------------------------------------------------|
| Accuracy   | Proportion of total correct predictions          |
| Precision  | True Positives / (True Positives + False Positives) |
| Recall     | True Positives / (True Positives + False Negatives) |
| F1-Score   | Harmonic mean of Precision and Recall            |
| AUC Score  | Area under the ROC Curve                         |


## 📌 Sigmoid Function in Logistic Regression

The logistic (sigmoid) function outputs a probability between 0 and 1:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

Logistic Regression applies this to the linear combination of features to make binary predictions.


## 📁 Files in the Repository

- `Task 4 Classification with Logistic Regression.ipynb` — Jupyter Notebook with code and visualizations
- `README.md` — Project overview and documentation
- `breast_cancer_data.csv` — Dataset used (if permitted)
- `Confusion Matrix.png`

    ![Confusion matrix](https://github.com/user-attachments/assets/47c45140-cd19-4383-bcc2-d335c0d4db32)
- `ROC Curve.png`

  ![ROC curve](https://github.com/user-attachments/assets/ce43402b-289a-4e26-b89a-a4d130ecdb5d)




## 🚀 How to Run

1. Clone the repository
2. Install dependencies:  
   `pip install pandas numpy matplotlib seaborn scikit-learn`
3. Run the Jupyter notebook or Python script
---

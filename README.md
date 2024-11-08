# Heart Attack Analysis Using Machine Learning

This project analyzes a dataset related to heart attack risks and performs predictions using various machine learning techniques. The goal is to identify which models are most effective for age prediction and classification of heart attack risk.

## Authors
- Roy Boker
- Maks Krutinsky

## Project Overview
This project explores multiple machine learning models, including Decision Trees, Linear Regression, and Random Forests, to determine the most accurate model for predicting age and classifying heart attack risk.

### Dataset
The dataset used for this project contains the following features:

- **Age**: Age of the individual.
- **Gender**: Gender of the individual.
- **cp**: Chest pain type (values 1-3).
- **trbps**: Resting blood pressure.
- **chol**: Cholesterol level.
- **fbs**: Fasting blood sugar.
- **thalach**: Maximum heart rate achieved.
- **oldpeak**: ST depression induced by exercise.
- **thall**: Thalassemia results.
- **exng**: Exercise-induced angina.
- **output**: Target variable indicating heart attack risk (binary).

## Problem Statement
The main focus of this project was to:

1. Predict age using regression analysis.
2. Classify the target variable, `output`, which indicates heart attack risk (yes/no).

## Methodology

### Preprocessing
- Checked for missing values and addressed them.
- Filtered out features with low correlation.
- Split the dataset into training and test sets (80%-20%).

### Age Prediction
**Regression Models Evaluated**:
- **Decision Tree Regressor**: With and without normalization.
- **Linear Regression**: Compared performance metrics before and after normalization.

### Discretization and Classification
The age column was divided into three categories:
- **29-40**
- **41-55**
- **56-77**

**Classifiers Evaluated**:
- **Decision Tree Classifier**
- **Linear Classifier**
- **Naive Bayes Classifier**
- **Random Forest Classifier**

## Results Overview

### Regression Analysis
**Decision Tree Regressor**:
- **With Normalization**: MSE = 65.37, R^2 = 0.11
- **Without Normalization**: MSE = 66.13, R^2 = 0.10

### Classification Results
**Decision Tree Classifier**:
- **Test Accuracy**: 70.49%
- **Train Accuracy**: 65.70%

**Linear Classifier**:
- **Test Accuracy**: 67.21%
- **Train Accuracy**: 66.52%

**Naive Bayes Classifier**:
- **Test Accuracy**: 48.35%
- **Train Accuracy**: 42.26%

**Random Forest Classifier**:
- **Test Accuracy**: 63.93%
- **Train Accuracy**: 99.58%

### Heart Attack Risk Prediction
For predicting whether an individual is at risk of a heart attack (binary classification), we evaluated various classifiers using accuracy and ROC AUC as performance metrics:

**Models Used**:
- **Decision Tree Classifier** (using both Gini and Entropy criteria)
- **Logistic Regression**
- **k-Nearest Neighbors (kNN)**
- **Random Forest Classifier**

**Evaluation Metrics**:
- **Accuracy**: Proportion of correctly classified instances.
- **ROC AUC**: Measures the area under the ROC curve, indicating the model's ability to distinguish between classes.

**Results**:
- **Decision Tree Classifier (Gini & Entropy)**: Accuracy = 70.49%, ROC AUC = 0.76
- **Logistic Regression**: Accuracy = 77.05%, ROC AUC = 0.84
- **k-Nearest Neighbors (kNN)**: Accuracy = 57.38%, ROC AUC = 0.54
- **Random Forest Classifier**: Accuracy = 75.41%, ROC AUC = 0.84

**Key Insights**:
- The **Logistic Regression model** provided the highest accuracy and ROC AUC, indicating strong performance for this binary classification task.
- **Random Forest** also performed well, showcasing its robustness in predictive tasks.
- The **kNN model** had limited accuracy, suggesting potential issues with feature scaling or parameter selection.
- **Decision Tree models** showed reasonable accuracy but were prone to overfitting, particularly when depth was not restricted.

## Conclusion
The analysis showed that the **Random Forest model** had a strong and balanced performance, demonstrating good generalization on unseen data. The **Linear Classifier** also performed well, especially for Class 1 predictions. **k-Nearest Neighbors (kNN)** and **Decision Tree models** faced challenges, with kNN showing limited accuracy and Decision Trees suffering from overfitting.

**Normalization** played a significant role in improving model performance, especially for the Decision Tree model.

---

## How to Run the Project
1. Load the dataset and perform the necessary preprocessing steps.
2. Train and evaluate models using the code provided in the project.
3. Compare the metrics and analyze the models' performance based on the output.

**Note**: For full code and implementation details, refer to the project notebook.

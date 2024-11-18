# Bank Customer Churn Prediction - Machine Learning Model

## Overview

The objective of this project is to predict whether a bank customer will churn (i.e., leave the bank) based on various features such as credit score, geography, age, balance, and more. This involves a series of data preprocessing steps, including data cleaning, encoding categorical variables, exploratory data analysis (EDA), handling class imbalance, and training various machine learning models to make predictions.

---

## Table of Contents

- [Objective](#objective)
- [Steps](#steps)
  - [1. Data Import and Exploration](#1-data-import-and-exploration)
  - [2. Data Cleaning](#2-data-cleaning)
  - [3. Encoding Categorical Data](#3-encoding-categorical-data)
  - [4. Exploratory Data Analysis (EDA)](#4-exploratory-data-analysis-eda)
  - [5. Handling Imbalanced Dataset](#5-handling-imbalanced-dataset)
  - [6. Data Splitting](#6-data-splitting)
  - [7. Model Training and Evaluation](#7-model-training-and-evaluation)
  - [8. Final Model Selection](#8-final-model-selection)
- [Conclusion](#conclusion)

---

## Objective

To predict whether a bank customer will churn based on factors like their credit score, geography, age, and balance. The goal is to provide a reliable prediction model to help the bank take preemptive actions to retain valuable customers.

---

## Steps

### 1. Data Import and Exploration

The first step is to import the necessary libraries and load the dataset for exploration:

- **Libraries**: Import the essential libraries: `pandas` for data manipulation, `seaborn` and `matplotlib` for visualizations.
- **Loading Data**: Load the dataset from a CSV file using the `pd.read_csv()` function.
- **Initial Exploration**:
  - Display the first few rows using `head()` to get an initial understanding of the data.
  - Display the last few rows using `tail()` to ensure there are no discrepancies.
  - Use `info()` to inspect the structure of the dataset, including column types, non-null values, and memory usage.
  - Use `describe()` to get a statistical summary of numerical columns (e.g., mean, median, standard deviation, etc.).

### 2. Data Cleaning

Data cleaning ensures the dataset is free from inconsistencies and irrelevant data:

- **Missing Values**: Check for missing values using `isnull().sum()`. If any columns have missing data, determine whether to drop or impute them.
- **Drop Irrelevant Columns**: Remove columns that don't contribute to the analysis, such as `RowNumber`, `CustomerID`, and `Surname`, using the `drop()` method. These columns typically don’t have predictive value for the model.

### 3. Encoding Categorical Data

Machine learning models require numerical input, so categorical data needs to be converted:

- **Geography and Gender**: Convert categorical columns like `Geography` and `Gender` into numerical format.
  - Use one-hot encoding for these features, turning each category into separate binary (0 or 1) columns using `pd.get_dummies()`. Set `drop_first=True` to avoid multicollinearity.

### 4. Exploratory Data Analysis (EDA)

EDA is a crucial step to understand the dataset and visualize any potential relationships:

- **Target Variable Distribution**: Analyze the distribution of the target variable (`Exited`) using `value_counts()`. This shows how many customers churned versus how many did not.
- **Class Imbalance**: Visualize the class imbalance using `sns.countplot()` to display the count of churned versus non-churned customers. If the data is imbalanced, the next step will address this issue.

### 5. Handling Imbalanced Dataset

Imbalanced data can lead to biased predictions, with the model favoring the majority class. To handle this:

- **SMOTE (Synthetic Minority Oversampling Technique)**: This technique generates synthetic samples for the minority class (churned customers) to balance the dataset.
  - Import `SMOTE` from the `imblearn.over_sampling` module.
  - Apply SMOTE to the dataset, creating a balanced set of features `X_smote` and target values `Y_smote`.
  - Use `Y_smote.value_counts()` to confirm that the classes are now balanced.

### 6. Data Splitting

After data preprocessing, split the dataset into training and testing sets to evaluate the model’s performance:

- **Training and Testing Split**: Use `train_test_split` from `sklearn.model_selection` to divide the dataset into training (80%) and testing (20%) sets. This allows the model to be trained on one subset of the data and evaluated on a separate unseen subset.

### 7. Model Training and Evaluation

Multiple machine learning models are trained and evaluated to find the best performer:

- **Logistic Regression**: A simple model for binary classification. Evaluate it using metrics like accuracy, precision, recall, and F1-score.
  - Example: Logistic Regression gives an F1 score of around 79%.
- **Support Vector Classifier (SVC)**: A powerful classifier that can handle both linear and non-linear data. The accuracy is improved to around 85%.
- **K-Nearest Neighbors (KNN)**: A non-parametric model that makes predictions based on the majority class of its nearest neighbors. Accuracy here is around 83%.
- **Decision Tree**: A model that splits the data based on feature values. Accuracy is around 79%.
- **Random Forest**: An ensemble of decision trees. This model gives the best performance, with an accuracy of around 87% and a precision of 86%.

### 8. Final Model Selection

After evaluating all models, **Random Forest** is selected as the final model because it provides the best balance between accuracy, precision, and other metrics.

- The Random Forest model consistently outperformed the others, demonstrating better predictive power in classifying customers who will churn versus those who will stay.

---

## Conclusion

- **Class Imbalance**: We effectively addressed the issue of class imbalance using the **SMOTE** technique, ensuring that both churned and non-churned customers were equally represented in the dataset.
- **Best Model**: The **Random Forest** classifier was chosen as the final model due to its high accuracy and precision, making it ideal for this churn prediction task.

This project showcases how data preprocessing, feature engineering, and the careful selection of machine learning models can create a robust prediction model for business use cases like churn prediction. 


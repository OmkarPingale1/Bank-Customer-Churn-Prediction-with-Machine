Bank Customer Churn Prediction

**Objective**  
Predict if a bank customer will churn based on features such as credit score, geography, age, balance, and more.

**Steps**

**1. Data Import and Exploration**
- Import libraries: `pandas`, `seaborn`, `matplotlib`.  
- Load the dataset using `pd.read_csv()`.  
- Display the first few rows using `head()` and last few rows using `tail()`.  
- Check the dataset’s structure with `info()` and `describe()`.  

**2. Data Cleaning**
- Use `isnull().sum()` to check for missing values.  
- Drop irrelevant columns such as `RowNumber`, `CustomerID`, and `Surname` using `drop()`.  

**3. Encoding Categorical Data**  
- Convert categorical variables like `Geography` and `Gender` into numerical format using `pd.get_dummies()` with `drop_first=True`.  

**4. Exploratory Data Analysis (EDA)**  
- Analyze the target variable (`Exited`) distribution with `value_counts()`.  
- Visualize class imbalance using `sns.countplot(data=df, x='Exited')`.  

**5. Handling Imbalanced Dataset**  
- Recognize that the dataset is imbalanced based on the target variable’s distribution.  
- Use SMOTE (Synthetic Minority Oversampling Technique) to balance classes.  
  - Import `SMOTE` from `imblearn.over_sampling`.  
  - Resample the data using `X_smote, Y_smote = SMOTE().fit_resample(X, Y)`.  
  - Check balanced distribution with `Y_smote.value_counts()`.  

**6. Data Splitting**  
- Split the dataset into training and test sets using `train_test_split` from `sklearn.model_selection`.  

**7. Model Training and Evaluation**
- Train multiple models: Logistic Regression, SVC, KNN, Decision Tree, and Random Forest.  
- Evaluate models using accuracy, precision, recall, and F1 Score:  
- Logistic Regression: Moderate performance with F1 Score ~ 79%.
- Support Vector Classifier (SVC): Improved accuracy ~ 85%.  
- K-Nearest Neighbors (KNN): Accuracy ~ 83%.  
- Decision Tree: Accuracy ~ 79%.  
- Random Forest: Best performance with accuracy ~ 87%, precision ~ 86%.  

**8. Final Model Selection**  
- Random Forest selected as the final model for its balance between accuracy and precision.  

---

**Conclusion**
- Addressed class imbalance effectively using SMOTE.  
- Random Forest outperformed other models with optimal metrics.  

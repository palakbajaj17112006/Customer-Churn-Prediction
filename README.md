# Customer Churn Prediction using Machine Learning

A machine learning project for predicting whether a customer is likely to **churn (leave a service)** based on demographic information, account details, services used, contract type, payment method, tenure, and charges.

The project compares multiple classification algorithms and evaluates them using **Accuracy, ROC-AUC, Classification Reports, Confusion Matrices, and 5-Fold Cross-Validation**. It also performs hyperparameter tuning using `GridSearchCV` and uses Random Forest feature importance to identify the most influential features.

---

## 📌 Project Overview

Customer churn is an important business problem for subscription-based companies. Identifying customers who are likely to leave allows businesses to take preventive actions such as:

* Offering personalized discounts
* Improving customer support
* Providing better service plans
* Targeting high-risk customers with retention campaigns
* Understanding factors associated with customer churn

This project develops and compares several machine learning classification models to predict customer churn.

---

## 🎯 Objectives

The main objectives of this project are:

1. Explore and understand the customer churn dataset.
2. Perform data cleaning and preprocessing.
3. Analyze relationships between customer attributes and churn.
4. Convert categorical variables into numerical features.
5. Train multiple machine learning classification models.
6. Compare model performance using Accuracy and ROC-AUC.
7. Perform 5-Fold Cross-Validation.
8. Perform hyperparameter tuning using `GridSearchCV`.
9. Identify important features using Random Forest.
10. Generate churn probabilities for customers.
11. Categorize customers into Low, Medium, and High churn-risk groups.

---

## 📂 Project Structure

```text
Customer-Churn-Prediction/
│
├── ml(4).ipynb
├── churn_dataset.csv
├── README.md
├── requirements.txt
└── .gitignore
```

### Files

| File                | Description                                                                          |
| ------------------- | ------------------------------------------------------------------------------------ |
| `ml(4).ipynb`       | Main Jupyter Notebook containing the complete analysis and machine learning workflow |
| `churn_dataset.csv` | Customer churn dataset used by the notebook                                          |
| `README.md`         | Project documentation                                                                |
| `requirements.txt`  | Python dependencies required to run the project                                      |
| `.gitignore`        | Files and folders excluded from Git                                                  |

---

# 📊 Dataset

The notebook loads the dataset using:

```python
df = pd.read_csv("churn_dataset.csv")
```

The dataset contains:

* **7,043 customer records**
* **21 columns initially**
* Customer demographic information
* Services subscribed to
* Contract information
* Payment information
* Monthly and total charges
* Churn status

### Dataset Columns

```text
customerID
gender
SeniorCitizen
Partner
Dependents
tenure
PhoneService
MultipleLines
InternetService
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
Contract
PaperlessBilling
PaymentMethod
MonthlyCharges
TotalCharges
Churn
```

---

# 🔍 Exploratory Data Analysis

The notebook performs several exploratory data analysis steps.

## Dataset Shape

The original dataset contains:

```text
7043 rows × 21 columns
```

After removing the customer ID:

```text
7043 rows × 20 columns
```

The target variable is:

```text
Churn
```

---

## Target Variable

The original churn values are:

```text
No
Yes
```

They are converted into binary values:

```text
No  → 0
Yes → 1
```

The resulting distribution is:

| Churn        | Customers |
| ------------ | --------: |
| 0 - No Churn |      5174 |
| 1 - Churn    |      1869 |

This shows that the dataset is imbalanced, with more customers staying than leaving.

---

# 🧹 Data Preprocessing

Several preprocessing operations are performed.

## 1. Customer ID Removal

`customerID` is removed because it is an identifier and does not provide useful predictive information.

```python
df = df.drop("customerID", axis=1)
```

---

## 2. Total Charges Conversion

`TotalCharges` is initially stored as a string.

It is converted into a numeric column using:

```python
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
```

This creates missing values where conversion is not possible.

The notebook identifies **11 missing values** in `TotalCharges`.

---

## 3. Duplicate Check

Duplicate rows are checked:

```python
df.duplicated().sum()
```

The dataset contains:

```text
0 duplicate rows
```

---

## 4. Target Encoding

The target variable is converted into binary values:

```python
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})
```

---

## 5. One-Hot Encoding

Categorical variables are converted into numerical variables using:

```python
pd.get_dummies(
    X,
    drop_first=True
)
```

Before encoding:

```text
7043 × 19
```

After one-hot encoding:

```text
7043 × 30
```

The resulting feature matrix contains **30 features**.

---

## 6. Train-Test Split

The data is divided into training and testing sets using:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

Initial split:

```text
Training: 5634 samples
Testing:  1409 samples
```

The notebook subsequently removes rows affected by missing values before final model evaluation, resulting in:

```text
Training: 5626 samples
Testing:  1406 samples
```

---

## 7. Feature Scaling

`StandardScaler` is used to standardize the features:

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Scaling is particularly useful for models such as:

* Logistic Regression
* KNN
* SVM

---

# 🤖 Machine Learning Models

The notebook compares five classification algorithms.

## 1. Logistic Regression

```python
LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

Logistic Regression provides the best basic-model accuracy in the notebook.

---

## 2. Decision Tree

```python
DecisionTreeClassifier(
    random_state=42
)
```

Decision Trees split the dataset based on feature values to make classification decisions.

---

## 3. K-Nearest Neighbors

```python
KNeighborsClassifier(
    n_neighbors=5
)
```

KNN predicts the class of a customer based on the classes of nearby observations.

---

## 4. Gaussian Naive Bayes

```python
GaussianNB()
```

Naive Bayes uses probability-based classification under the assumption of conditional independence between features.

---

## 5. Support Vector Machine

```python
SVC(
    probability=True,
    random_state=42
)
```

SVM attempts to find an effective decision boundary between churn and non-churn customers.

---

# 📈 Model Evaluation

The models are evaluated using:

### Accuracy

Measures the percentage of correctly classified customers.

### ROC-AUC

Measures how well the model distinguishes between churn and non-churn customers.

### Classification Report

Includes:

* Precision
* Recall
* F1-score
* Support

### Confusion Matrix

Shows:

* True Negatives
* False Positives
* False Negatives
* True Positives

---

# 🏆 Model Results

The notebook produced the following results:

| Model                   |   Accuracy |    ROC-AUC |
| ----------------------- | ---------: | ---------: |
| **Logistic Regression** | **0.8065** | **0.8414** |
| SVM                     |     0.7923 |     0.7960 |
| KNN                     |     0.7468 |     0.7709 |
| Decision Tree           |     0.7326 |     0.6503 |
| Naive Bayes             |     0.6551 |     0.8089 |

### Best Basic Model

Based on test-set accuracy and ROC-AUC:

**Logistic Regression**

```text
Accuracy: 80.65%
ROC-AUC:  84.14%
```

---

# 🔄 5-Fold Cross-Validation

The project also evaluates model stability using 5-Fold Cross-Validation.

| Model               | Mean CV Accuracy | Standard Deviation |
| ------------------- | ---------------: | -----------------: |
| Logistic Regression |           0.8038 |             0.0134 |
| Decision Tree       |           0.7254 |             0.0026 |
| KNN                 |           0.7624 |             0.0109 |
| Naive Bayes         |           0.6623 |             0.0206 |
| SVM                 |           0.8039 |             0.0138 |

The cross-validation results show that **Logistic Regression and SVM perform very similarly**, with SVM having a slightly higher mean CV accuracy.

---

# ⚙️ Hyperparameter Tuning

`GridSearchCV` with 5-fold cross-validation is used to search for better hyperparameters.

## Logistic Regression

Best parameters:

```text
C = 100
solver = liblinear
```

Best CV Accuracy:

```text
0.8050
```

---

## Decision Tree

Best parameters:

```text
criterion = gini
max_depth = 3
min_samples_split = 2
```

Best CV Accuracy:

```text
0.7872
```

---

## KNN

Best parameters:

```text
metric = manhattan
n_neighbors = 15
weights = uniform
```

Best CV Accuracy:

```text
0.7879
```

---

## SVM

Best parameters:

```text
C = 1
kernel = rbf
gamma = scale
```

Best CV Accuracy:

```text
0.8039
```

---

# 🌲 Random Forest Analysis

The notebook also trains a Random Forest classifier:

```python
RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
```

The Random Forest model achieves:

```text
Accuracy: 78.88%
ROC-AUC:  82.65%
```

The Random Forest is then used to calculate feature importance.

---

# 🔑 Feature Importance

The top features identified by Random Forest are:

| Rank | Feature                        | Importance |
| ---: | ------------------------------ | ---------: |
|    1 | TotalCharges                   |     0.1949 |
|    2 | MonthlyCharges                 |     0.1694 |
|    3 | tenure                         |     0.1670 |
|    4 | InternetService_Fiber optic    |     0.0391 |
|    5 | PaymentMethod_Electronic check |     0.0384 |
|    6 | Contract_Two year              |     0.0319 |
|    7 | gender_Male                    |     0.0278 |
|    8 | OnlineSecurity_Yes             |     0.0266 |
|    9 | PaperlessBilling_Yes           |     0.0252 |
|   10 | Partner_Yes                    |     0.0243 |

The three strongest features according to the Random Forest analysis are:

```text
TotalCharges
MonthlyCharges
tenure
```

---

# ⚠️ Customer Churn Risk Classification

The project goes beyond binary predictions by assigning customers to three risk categories based on their predicted churn probability.

```python
def classify_risk(probability):

    if probability < 0.30:
        return "Low Risk"

    elif probability < 0.70:
        return "Medium Risk"

    else:
        return "High Risk"
```

### Risk Categories

| Churn Probability | Risk Level  |
| ----------------: | ----------- |
|          `< 0.30` | Low Risk    |
|    `0.30 – <0.70` | Medium Risk |
|          `≥ 0.70` | High Risk   |

The final prediction table contains:

```text
Actual_Churn
Predicted_Churn
Churn_Probability
Risk_Level
```

This makes the model output more useful for customer-retention applications.

---

# 📊 Visualizations

The notebook generates several visualizations, including:

* Customer churn distribution
* Numerical features vs churn
* Correlation matrix
* Churn percentage by categorical features
* Confusion matrices
* Model accuracy comparison
* Random Forest feature importance
* Customer churn-risk distribution

---

# 🛠️ Technologies Used

The project is implemented in Python using:

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Jupyter Notebook**

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Customer-Churn-Prediction.git
```

```bash
cd Customer-Churn-Prediction
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
ml(4).ipynb
```

Make sure `churn_dataset.csv` is located in the same directory as the notebook.

Then run the notebook cells sequentially.

---

# 📌 Expected Workflow

The notebook follows this pipeline:

```text
Dataset
   ↓
Data Loading
   ↓
Data Inspection
   ↓
Exploratory Data Analysis
   ↓
Data Cleaning
   ↓
Missing Value Handling
   ↓
Target Encoding
   ↓
One-Hot Encoding
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Cross-Validation
   ↓
Hyperparameter Tuning
   ↓
Random Forest Feature Importance
   ↓
Churn Probability
   ↓
Risk Classification
```

---

# 💡 Key Findings

The analysis demonstrates that:

1. Logistic Regression achieved the highest basic test accuracy among the five initially evaluated models.
2. Logistic Regression also achieved the highest basic ROC-AUC.
3. Logistic Regression and SVM showed very similar cross-validation performance.
4. TotalCharges, MonthlyCharges, and tenure were the most important features according to the Random Forest model.
5. The model can produce a churn probability rather than only a binary prediction.
6. Customers can be categorized into Low, Medium, and High churn-risk groups.

---

# ⚠️ Limitations

This project is primarily a machine learning analysis notebook and has some limitations:

* The dataset is imbalanced between churn and non-churn customers.
* The model evaluation focuses mainly on accuracy and ROC-AUC.
* No dedicated deployment API or web interface is included.
* The current notebook does not save the trained model as a serialized `.pkl` or `.joblib` file.
* Risk thresholds of 0.30 and 0.70 are manually defined.
* The notebook uses a single train-test split in addition to cross-validation.

---

# 🔮 Future Improvements

Possible improvements include:

* Handle class imbalance using class weights or resampling techniques.
* Use Precision-Recall AUC in addition to ROC-AUC.
* Perform more extensive hyperparameter tuning.
* Use an end-to-end preprocessing pipeline.
* Save the trained model using `joblib`.
* Develop a Streamlit dashboard.
* Build a Flask/FastAPI prediction API.
* Deploy the model as a web application.
* Add customer-level explanations using SHAP.
* Optimize the model specifically for detecting churn customers rather than maximizing overall accuracy.

---

# 👨‍💻 Author

**Palak Bajaj**

Computer Science Engineering Student
VIT-AP University

---

# 📄 License

This project is intended for educational and learning purposes.

If you use or modify this project, please provide appropriate attribution.

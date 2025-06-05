# task7
# 🧠 SVM Classifier for Breast Cancer Detection

## 🎯 Objective
Classify tumors as malignant (M) or benign (B) using Support Vector Machines (SVM) with both Linear and RBF kernels.

---

## 📁 Dataset
- Source: Breast Cancer Dataset (CSV)
- Target Variable: `diagnosis` — encoded as 1 (Malignant), 0 (Benign)
- Features: 30 numerical attributes (e.g., `radius_mean`, `texture_mean`, etc.)

---

## 🔟 Step-by-Step Workflow

### 1. Import Libraries
Essential libraries like NumPy, Pandas, Matplotlib, Scikit-learn, and Mlxtend were imported to support data processing, modeling, evaluation, and visualization.

-->pandas & numpy: Data handling

-->matplotlib & seaborn: Plotting

-->train_test_split: Splits data into training and testing

-->SVC: The Support Vector Classifier from scikit-learn

-->StandardScaler: Scales feature values to a standard range

-->GridSearchCV: For hyperparameter tuning

### 2. Load & Explore Dataset
Loaded the CSV file using `pandas.read_csv()` and explored its shape and structure to understand the features and labels.

### 3. Preprocess Data
Removed the ID column, encoded the target values (`M` as 1 and `B` as 0), and checked for any missing values.

✅ Explanation:

Removes unnecessary columns

Converts target column diagnosis to binary (1 = Malignant, 0 = Benign)

Scales features so that each has mean = 0 and std = 1 (important for SVM)

### 4. Feature Scaling
Split the dataset into features (X) and target (y), and used `StandardScaler` to normalize the feature values for optimal SVM performance.

### 5. Train-Test Split
Split the data into training and testing sets (80% training, 20% testing) using `train_test_split()` to evaluate model performance reliably.

✅ Explanation:

Splits dataset into:

70% for training

30% for testing

### 6. Train SVM with Linear Kernel
Trained an SVM with a linear kernel and evaluated it using a confusion matrix and classification report.

### 7. Train SVM with RBF Kernel
Trained an SVM with an RBF (non-linear) kernel to allow for curved decision boundaries, and evaluated its performance.

✅ Explanation:

confusion_matrix: How many predictions were correct/incorrect

classification_report: Includes precision, recall, F1-score for each class

### 8. Hyperparameter Tuning (GridSearchCV)
Used `GridSearchCV` to test multiple values of `C` and `gamma` to find the best-performing configuration using 5-fold cross-validation.

✅ Explanation:

C: Penalty for misclassification

gamma: Controls curvature of decision boundary

GridSearchCV: Tests all combinations of parameters using 5-fold cross-validation

### 9. Cross-Validation
Performed 5-fold cross-validation on the best RBF SVM model to ensure consistency and generalizability across different splits.

✅ Explanation:

Performs 5-fold cross-validation:

Checks model performance across different data splits

Returns mean accuracy

### 10. Visualize Decision Boundary
Used only the first 2 features to train a 2D SVM and plotted its decision boundary using `plot_decision_regions` from `mlxtend`.

✅ Explanation:

Visualizes the boundary between classes using only two features.

Useful to understand how SVM separates classes using the kernel trick.

---

## 📈 Results and Interpretation

### 🔹 Decision Boundary Plot (RBF Kernel, 2 Features)
This plot shows how the SVM with an RBF kernel separates classes using just the first two features.

![image](https://github.com/user-attachments/assets/529fa31e-9018-4f2a-b767-c5fb72731830)


- The **blue** and **orange** regions represent the areas predicted as Benign (0) and Malignant (1) respectively.
- The **boundary** between them is non-linear, showcasing the power of the RBF kernel to handle complex patterns.
- Most points are well-separated with very few misclassifications near the decision edge.

---

### 🔹 Linear Kernel Performance

**Confusion Matrix:**


#### 📊 Classification Report

| 🏷 Class | 🎯 Precision | 🔁 Recall | 🧮 F1-Score | 🔢 Support |
|---------|--------------|-----------|------------|------------|
|    0    |     0.97     |   0.96    |    0.96     |     71     |
|    1    |     0.93     |   0.95    |    0.94     |     43     |

✅ **Accuracy**: **96%**

🔍 **Interpretation**:
- The **linear kernel** works well with a straight-line boundary.
- High accuracy in identifying both benign and malignant tumors.
- Some malignant cases misclassified due to linear limitations.

---

## 🌀 What is RBF?

**RBF (Radial Basis Function)** is a commonly used **non-linear kernel** in Support Vector Machines (SVM). It helps the model create **curved decision boundaries**, making it ideal for datasets where classes are not linearly separable.

### ✅ Key Points:
- Transforms data into higher dimensions
- Captures complex patterns
- Requires tuning of `C` and `gamma` parameters
- Performs better than linear kernel on non-linear data


### 🌀 🔹 RBF Kernel SVM


#### 📊 Classification Report

| 🏷 Class | 🎯 Precision | 🔁 Recall | 🧮 F1-Score | 🔢 Support |
|---------|--------------|-----------|------------|------------|
|    0    |     0.97     |   0.99    |    0.98     |     71     |
|    1    |     0.98     |   0.95    |    0.96     |     43     |

✅ **Accuracy**: **97%**

🔍 **Interpretation**:
- The **RBF kernel** models a curved boundary, allowing it to handle more complex patterns.
- Excellent precision and recall — especially on malignant tumors.
- Overall improvement over the linear kernel with better generalization.

---

### ⚔️ Linear vs RBF Summary

| Feature                         | 🔹 Linear Kernel | 🌀 RBF Kernel |
|----------------------------------|------------------|---------------|
| 🔳 Decision Boundary             | Straight Line    | Curved Surface |
| 🎯 Accuracy                      | 96%              | **97%** ✅     |
| 🧠 Handles Non-Linear Data       | ❌ No             | ✅ Yes         |
| 📈 Best For                      | Simple Data      | Complex Data   |

🏆 **Conclusion**: The **RBF kernel** outperformed the linear kernel and is better suited for this binary classification problem.

---



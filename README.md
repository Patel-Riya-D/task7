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

### 2. Load & Explore Dataset
Loaded the CSV file using `pandas.read_csv()` and explored its shape and structure to understand the features and labels.

### 3. Preprocess Data
Removed the ID column, encoded the target values (`M` as 1 and `B` as 0), and checked for any missing values.

### 4. Feature Scaling
Split the dataset into features (X) and target (y), and used `StandardScaler` to normalize the feature values for optimal SVM performance.

### 5. Train-Test Split
Split the data into training and testing sets (80% training, 20% testing) using `train_test_split()` to evaluate model performance reliably.

### 6. Train SVM with Linear Kernel
Trained an SVM with a linear kernel and evaluated it using a confusion matrix and classification report.

### 7. Train SVM with RBF Kernel
Trained an SVM with an RBF (non-linear) kernel to allow for curved decision boundaries, and evaluated its performance.

### 8. Hyperparameter Tuning (GridSearchCV)
Used `GridSearchCV` to test multiple values of `C` and `gamma` to find the best-performing configuration using 5-fold cross-validation.

### 9. Cross-Validation
Performed 5-fold cross-validation on the best RBF SVM model to ensure consistency and generalizability across different splits.

### 10. Visualize Decision Boundary
Used only the first 2 features to train a 2D SVM and plotted its decision boundary using `plot_decision_regions` from `mlxtend`.

---

## 📈 Results and Interpretation

### 🔹 Decision Boundary Plot (RBF Kernel, 2 Features)
This plot shows how the SVM with an RBF kernel separates classes using just the first two features.



- The **blue** and **orange** regions represent the areas predicted as Benign (0) and Malignant (1) respectively.
- The **boundary** between them is non-linear, showcasing the power of the RBF kernel to handle complex patterns.
- Most points are well-separated with very few misclassifications near the decision edge.

---

### 🔹 Linear Kernel Performance

**Confusion Matrix:**


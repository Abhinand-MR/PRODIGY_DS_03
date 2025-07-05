# PRODIGY_DS_03
# Task 03 – Decision Tree Classifier (Bank Marketing Dataset)

## 🎯 Objective
Build a Decision Tree Classifier to predict whether a customer will subscribe to a term deposit based on demographic and behavioral features using the Bank Marketing dataset.

## 📎 Dataset
- Source: [UCI Machine Learning Repository – Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- The dataset includes features such as age, job, marital status, education, contact method, previous outcome, and target variable `y` (yes/no).

## 🧰 Tools & Libraries Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## ⚙️ Task Workflow

1. **Data Loading**
   - Imported the dataset using `pandas.read_csv()`.

2. **Data Preprocessing**
   - Checked for missing values and data types.
   - Applied label encoding and one-hot encoding for categorical variables.
   - Split the dataset into feature set (X) and target variable (y).

3. **Model Building**
   - Used `train_test_split` to divide data into training and testing sets.
   - Built a Decision Tree Classifier using `sklearn.tree.DecisionTreeClassifier`.
   - Trained the model and made predictions on the test set.

4. **Model Evaluation**
   - Evaluated using:
     - Accuracy Score
     - Confusion Matrix
     - Classification Report (Precision, Recall, F1-Score)
   - Visualized the confusion matrix and tree structure (optional).

5. **Visualization**
   - Feature importance bar chart
   - Decision tree plot (optional using `plot_tree`)

## 📊 Output

- A trained Decision Tree model to predict customer responses.
- Accuracy score and evaluation metrics.
- Visualizations showing model decisions and important features.

## 📷 Sample Visuals
*(Optional: Add decision_tree_plot.png, confusion_matrix.png, feature_importance.png)*

## ✅ Learnings

- How to apply supervised machine learning on marketing data.
- Understood encoding strategies for categorical features.
- Learned to evaluate classification models using multiple metrics.
- Gained insights into the interpretability of Decision Trees.

## 📂 Folder Structure


# Model Lens: Classification Model Comparison

## 1. Problem statement

In this assignment, we take one dataset and train five different classification models on it. The aim is to find out which model works best for this dataset.

A model is given a set of measurements and it has to predict which class the record belongs to. We measure how well each model does this using six evaluation metrics. We then compare the five models and choose a winner.

We also built a Streamlit web application. In this application, the user can upload a test file, select any one of the five models, and see the results of that model on the uploaded data.

## 2. Dataset description

We used the **Breast Cancer Wisconsin Diagnostic dataset**. This dataset is available in the UCI Machine Learning Repository and is also included in the scikit-learn library.

The dataset has the following properties:

- Number of records: 569
- Number of features: 30
- Type of problem: binary classification

The target column has two values:

- `0` = malignant (cancerous)
- `1` = benign (not cancerous)

The 30 features are measurements taken from images of breast cells. They describe values such as radius, texture, perimeter, area, and smoothness of the cells.



## 3. GitHub repository link

The complete source code is available at the link given below:

`https://github.com/MG-kildon/ML-Assignment2`



## 4. Models used


1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbour Classifier
4. Naive Bayes Classifier (Gaussian)
5. Random Forest Classifier (ensemble model)



### 4.1 Comparison table

The values given below are calculated on the 114 test records. All five models were tested on the same records, so the comparison is fair.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| kNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest (Ensemble) | 0.9474 | 0.9937 | 0.9583 | 0.9583 | 0.9583 | 0.8869 |

### 4.2 Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | This model gave the best result. It got the highest value in accuracy, AUC, F1 and MCC. The two classes in this dataset can be separated well by a straight boundary, so this simple model works very well here. |
| Decision Tree | This model gave the lowest result among the five. A single tree splits the data using one feature at a time, so it does not use the combined effect of all features. Its AUC value of 0.9163 is much lower than the other models. |
| kNN | This model gave the second best result. Its recall is 1.0000, which means it identified every benign record correctly. This model works well here because similar cell measurements belong to the same class. |
| Naive Bayes | This model gave a good AUC of 0.9878 but a lower accuracy of 0.9386. The reason is that this model assumes all features are independent of each other. In this dataset the features are related, for example radius and area increase together, so this assumption is not fully correct. |
| Random Forest (Ensemble) | This model combines 300 decision trees, so its result is much better than the single Decision Tree. Its AUC of 0.9937 is high. However, it did not perform better than Logistic Regression on this dataset. |
| Overall winner for our dataset | **Logistic Regression**. It gave the highest accuracy (0.9825), AUC (0.9954), F1 score (0.9861) and MCC (0.9623). This shows that a simple model can perform better than a complex model when the data is clearly separable. |




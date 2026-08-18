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

This dataset satisfies the assignment conditions, because 569 records is more than the required 500, and 30 features is more than the required 12.

### 2.1 How the data is split

We divided the dataset into two parts:

- Training data: 455 records (80%)
- Test data: 114 records (20%)

The split is stratified. This means both classes are present in the same proportion in the training part and the test part. We used `random_state=42` so that the same split is produced every time the program is run.

The five models are trained using only the training part. The models never see the test part during training. The test part is saved as `test_data.csv` and is used in the Streamlit application.

We also created a second file called `sample_upload.csv`. It contains 46 records taken from the test part. This file is used to show that the upload option in the application works with a file other than the default one.

Dataset source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

## 3. GitHub repository link

The complete source code is available at the link given below:

`https://github.com/<your-username>/<your-repository>`

## 4. Steps to run the project

The following commands are used to run the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python model/train_models.py
streamlit run app.py
```

The first three commands create a separate environment and install the required libraries. The fourth command trains the five models and creates the test file. The last command starts the web application.

The application opens at `http://localhost:8501`.

## 5. Models used

We implemented the following five models on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbour Classifier
4. Naive Bayes Classifier (Gaussian)
5. Random Forest Classifier (ensemble model)

For Logistic Regression and K-Nearest Neighbour, we first applied standardisation to the features. Standardisation brings all features to a similar range. This step is required for these two models because they compare feature values directly, and a feature with large values would otherwise dominate the result. Decision Tree, Naive Bayes, and Random Forest do not need this step.

### 5.1 Comparison table

The values given below are calculated on the 114 test records. All five models were tested on the same records, so the comparison is fair.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| kNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest (Ensemble) | 0.9474 | 0.9937 | 0.9583 | 0.9583 | 0.9583 | 0.8869 |

### 5.2 Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | This model gave the best result. It got the highest value in accuracy, AUC, F1 and MCC. The two classes in this dataset can be separated well by a straight boundary, so this simple model works very well here. |
| Decision Tree | This model gave the lowest result among the five. A single tree splits the data using one feature at a time, so it does not use the combined effect of all features. Its AUC value of 0.9163 is much lower than the other models. |
| kNN | This model gave the second best result. Its recall is 1.0000, which means it identified every benign record correctly. This model works well here because similar cell measurements belong to the same class. |
| Naive Bayes | This model gave a good AUC of 0.9878 but a lower accuracy of 0.9386. The reason is that this model assumes all features are independent of each other. In this dataset the features are related, for example radius and area increase together, so this assumption is not fully correct. |
| Random Forest (Ensemble) | This model combines 300 decision trees, so its result is much better than the single Decision Tree. Its AUC of 0.9937 is high. However, it did not perform better than Logistic Regression on this dataset. |
| Overall winner for our dataset | **Logistic Regression**. It gave the highest accuracy (0.9825), AUC (0.9954), F1 score (0.9861) and MCC (0.9623). This shows that a simple model can perform better than a complex model when the data is clearly separable. |

### 5.3 Meaning of the evaluation metrics

| Metric | Meaning |
|---|---|
| Accuracy | How many predictions were correct out of the total predictions. |
| AUC | How well the model separates the two classes. A value close to 1 is good. |
| Precision | Out of the records predicted as benign, how many were actually benign. |
| Recall | Out of the actual benign records, how many the model found. |
| F1 | A single value that balances precision and recall. |
| MCC | A balanced score that considers all four values of the confusion matrix. |

## 6. Streamlit application

The application has the following features:

1. **Upload option** - the user can upload a test file in CSV format.
2. **Model selection** - a dropdown list is provided to select any one of the five models.
3. **Evaluation metrics** - accuracy, AUC, precision, recall, F1 and MCC are displayed for the selected model.
4. **Confusion matrix** - shown as a coloured table.
5. **Classification report** - shown for both classes.
6. **Prediction preview** - the first ten records are shown with the predicted class.
7. **Comparison table** - the results of all five models are shown together at the bottom of the page.

If the user does not upload any file, the application uses the `test_data.csv` file that is already present in the repository. So the application always shows a result when it is opened.

### 6.1 Checking of the uploaded file

The application checks the uploaded file before using it.

- If the file does not contain all the 30 required feature columns, an error message is displayed and the required column names are listed.
- If the file contains the features but does not contain the `target` column, then only the predictions are displayed. The evaluation metrics are not calculated, because the correct answers are needed to calculate them.

## 7. Deployment

The application is deployed using Streamlit Community Cloud. The live link is given below:

`https://<your-streamlit-app-url>`

## 8. Files in the repository

```text
.
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── sample_upload.csv
└── model/
    ├── train_models.py
    └── artifacts/
        ├── logistic_regression.pkl
        ├── decision_tree.pkl
        ├── knn.pkl
        ├── naive_bayes.pkl
        ├── random_forest.pkl
        ├── metadata.json
        └── metrics.csv
```

The purpose of each file is given below:

| File | Purpose |
|---|---|
| `app.py` | The Streamlit web application. |
| `model/train_models.py` | Trains the five models and calculates the metrics. |
| `model/artifacts/*.pkl` | The five trained models saved in files, so that the application does not train them again every time. |
| `model/artifacts/metrics.csv` | The comparison table of all five models. |
| `test_data.csv` | The 114 test records used for evaluation. |
| `sample_upload.csv` | A smaller file of 46 records, used to demonstrate the upload option. |
| `requirements.txt` | The list of libraries required to run the project. |

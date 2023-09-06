# Toxic Tweets Classification Project

## Overview

This project aims to predict the toxicity of tweets using Natural Language Processing (NLP) techniques. The dataset used for this project is the "Toxic Tweets Dataset" obtained from Kaggle, which contains labeled tweets as either toxic (1) or non-toxic (0).

## Dataset

The dataset can be downloaded from the following Kaggle competition:
[Toxic Tweets Dataset](https://www.kaggle.com/datasets/ashwiniyer176/toxic-tweets-dataset)

## Project Steps

1. **Data Preprocessing**: The CSV file is converted into a pandas DataFrame for further analysis.

2. **Text Processing**:
   - Text data is also transformed into TF-IDF features.

3. **Model Training**:
   - Decision Trees
   - Random Forest
   - Naive Bayes Model
   - K-Nearest Neighbors (KNN) Classifier
   - Support Vector Machine (SVM)

4. **Model Evaluation**:
   - Precision, Recall, F1-Score are calculated for each model.
   - Confusion matrices are generated.
   - ROC-AUC curves are plotted.

## Metrics and Evaluation

For each of the models, the following metrics are provided in the Jupyter Notebook:
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Curve

## Dependencies

- Python 3.x
- Jupyter Notebook
- scikit-learn
- pandas
- matplotlib
- seaborn

## Usage

1. Clone this repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Open and run the `toxic_tweets_classification.ipynb` Jupyter Notebook to reproduce the results.
   
## Acknowledgments

- The dataset used in this project was collected by the original contributors on Kaggle.

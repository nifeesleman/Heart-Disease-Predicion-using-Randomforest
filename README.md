Heart Disease Prediction and Analysis
This repository contains the code and resources for predicting heart disease using the Random Forest algorithm. The analysis is performed in a Jupyter Notebook, leveraging various Python libraries for data processing, model building, and visualization.

Project Overview
This project aims to predict the likelihood of heart disease in patients based on medical attributes. The dataset is sourced from the UCI Machine Learning Repository and includes features such as age, sex, chest pain type, resting blood pressure, cholesterol levels, and more.

Repository Contents
Heart Disease models and evalution using RANDEMForest.ipynb: The main Jupyter Notebook containing all the code for data preprocessing, model training, evaluation, and analysis.
Objectives
Develop a predictive model for heart disease.
Analyze key factors contributing to heart disease using feature importance from the Random Forest algorithm.
Provide insights and visualizations to understand the dataset and model performance.
Dataset
The dataset includes the following features:

Age
Sex
Chest pain type
Resting blood pressure
Cholesterol levels
Fasting blood sugar
Resting electrocardiographic results
Maximum heart rate achieved
Exercise-induced angina
Oldpeak
The slope of the peak exercise ST segment
Number of major vessels colored by fluoroscopy
Thalassemia
Methodology
Data Preprocessing
Data Cleaning: Handling missing values and outliers.
Feature Encoding: Converting categorical variables into numerical formats.
Feature Scaling: Normalizing the data to improve model performance.
Model Building
Random Forest Algorithm: Constructs multiple decision trees during training and outputs the mode of the classes for classification.
Training and Testing: Splitting the dataset into training and testing sets to evaluate the model's performance.
Model Evaluation
Accuracy: Measuring the percentage of correctly predicted instances.
Confusion Matrix: Analyzing the true positives, true negatives, false positives, and false negatives.
ROC Curve and AUC: Evaluating the model's ability to discriminate between positive and negative classes.
Feature Importance
Analysis: Identifying the most influential features contributing to heart disease predictions.
Visualization: Plotting the importance of each feature to provide insights into the factors most associated with heart disease.
Results
The Random Forest model demonstrates high accuracy and reliability in predicting heart disease. The feature importance analysis reveals that attributes such as age, cholesterol levels, and chest pain type are significant predictors of heart disease.

Conclusion
This project successfully develops a predictive model for heart disease using the Random Forest algorithm. The findings offer valuable insights into the medical factors contributing to heart disease, aiding in early diagnosis and preventive measures.

Tools and Technologies
Jupyter Notebook: For interactive coding and analysis.
Python: Programming language used for data manipulation, model building, and evaluation.
Pandas and NumPy: Libraries for data processing.
Scikit-learn: Library for implementing machine learning algorithms and model evaluation.
Matplotlib and Seaborn: Libraries for data visualization.
Usage
To run the project, follow these steps:

Clone this repository.
Install the required libraries using pip install -r requirements.txt.
Open the Heart Disease models and evalution using RANDEMForest.ipynb notebook in Jupyter.
Execute the cells to preprocess the data, build the model, and analyze the results.

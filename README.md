# Solubility Prediction Model
# CAT 2

**Name**: Maina S Wachira  
**Registration Number**: PA106/G/14962/21  
**Course**: B.Sc. Software Engineering (Y4S1)

---

## Project Overview

This project builds a machine learning model to predict the solubility of chemical compounds in water. Using various features of the compounds, the model aims to predict their logarithmic solubility, enabling insights into the behavior of compounds in aqueous solutions. The project pipeline includes data loading, cleaning, splitting, training, evaluation, and visualization.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation and Setup](#installation-and-setup)
3. [Data Preparation](#data-preparation)
4. [Model Training and Evaluation](#model-training-and-evaluation)
5. [Results and Visualizations](#results-and-visualizations)
6. [Conclusion](#conclusion)
7. [Dependencies](#dependencies)

---

## Project Structure

- **data**: Contains the dataset for model training and testing.
- **notebooks**: Jupyter notebooks used to develop and test the model.
- **scripts**: Python scripts for data processing, training, and evaluation.
- **README.md**: Project documentation.

## Installation and Setup

Set up a Virtual Environment
Create and activate a virtual environment for dependency management.
bash
Copy code
python -m venv myenv
Windows: myenv\Scripts\activate
MacOS/Linux: source myenv/bin/activate
Install Dependencies
Install the required libraries listed in the requirements.txt file.
bash
Copy code
pip install -r requirements.txt
Data Preparation
The dataset contains information on various chemical compounds. The key steps for data preparation include:

Loading the Data: Import the dataset into a DataFrame using pandas.
Cleaning: Handle missing values, normalize data, and split the data into training and testing sets.
Feature Selection: Select features relevant to predicting solubility.
python
Copy code
# Sample code for loading and cleaning data
import pandas as pd

data = pd.read_csv('data/solubility_data.csv')
data = data.dropna()
# Additional cleaning and feature selection
Model Training and Evaluation
1. Model Selection
We utilize a Random Forest Regressor for its robustness in handling non-linear relationships. The model is trained to minimize the mean squared error between the actual and predicted values.

2. Training
Split the dataset into training and testing sets, then fit the model on the training data.

python
Copy code
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)
3. Evaluation
The model's performance is evaluated using metrics such as Mean Squared Error (MSE) and R-Squared (R²) to gauge accuracy.

python
Copy code
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
Results and Visualizations
Scatter Plot of Predictions
A scatter plot comparing experimental and predicted values helps visualize model performance.

python
Copy code
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(5,5))
plt.scatter(x=y_test, y=y_pred, color="#7CAE00", alpha=0.3)
z = np.polyfit(y_test, y_pred, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), "#F8766D")
plt.xlabel("Experimental LogS")
plt.ylabel("Predicted LogS")
plt.show()
Example Plot

#Conclusion
This model provides a baseline for predicting the solubility of compounds, with reasonable accuracy for initial applications. Further tuning and feature engineering may improve performance, and additional metrics such as MAE or RMSE could provide more insight.

#Dependencies
The primary dependencies for this project are:
pandas
numpy
scikit-learn
matplotlib
To install them individually, use:

bash
Copy code
pip install pandas numpy scikit-learn matplotlib

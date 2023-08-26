# Machine-learning-housePricing
This a Project of projection of housing price, using the Linear Regression.

# Linear Regression with Custom Train-Test Split

This repository contains a Python implementation of linear regression with a custom train-test split function. This code aims to demonstrate how to perform linear regression using gradient descent, split the dataset into training and test sets, and evaluate the model's performance using the R-squared score.

## Overview

Linear regression is a fundamental technique in machine learning used to model the relationship between input features and target variables. This implementation includes the following components:

1. Hypothesis function: Calculates the predicted output given input features and model parameters.
2. Error function: Computes the mean squared error between predicted and actual values.
3. Gradient function: Computes the gradient of the cost function with respect to model parameters.
4. Gradient descent optimization: Updates model parameters iteratively using gradients.
5. R-squared score calculation: Measures the goodness of fit of the model's predictions.

The custom train-test split function splits the dataset into training and test sets without any randomization, ensuring deterministic behavior.

## Usage

1. Clone the repository:
   git clone https://github.com/yourusername/linear-regression-custom-split.git
  cd linear-regression-custom-split

2. Place your dataset file `housing.csv` in the root directory.

3. Run the `housingPrediction.py` script using a Python interpreter:
   python linear_regression_custom_split.py

4. The script will perform linear regression on the training set, make predictions on both the training and test sets, and calculate R-squared scores for evaluation.

## Dependencies

- Python (>=3.6)
- NumPy (>=1.20)
- pandas (>=1.2)

Install the required dependencies using the following command:
pip install numpy pandas


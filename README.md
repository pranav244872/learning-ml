# Learning Machine Learning

Welcome to my journey into the world of Machine Learning! This repository serves as my learning ground where I explore various machine learning concepts and algorithms.

## What I've Learned So Far

### Linear Regression

In this repository, I've implemented a simple linear regression model from scratch using Python. Here's what I've covered:

- Implemented a cost function to evaluate the performance of a linear regression model.
- Implemented a Gradient Descent function to find the optimized values of \( w \) and \( b \).
- Utilized matplotlib for plotting the linear regression line.
- Explored the impact of the learning rate (alpha) and the number of iterations on model convergence and fit.  

Requirements
To run the code for linear regression, you'll need to install the following Python modules:
1. **matplotlib**: For plotting the linear regression line.
2. **numpy**: For numerical operations and array manipulations.

### Logistic Regression

In this document, I've learned the basics of logistic regression, its purpose, and its cost function.
- What is Logistic Regression?  
  Logistic regression is a statistical model used for binary classification tasks. Unlike linear regression, which predicts continuous values, logistic regression predicts the probability of the input belonging to a certain class (usually binary: 0 or 1).

- Why is Logistic Regression Used?  
  Logistic regression is used when the dependent variable is categorical (binary). It's widely used in various fields such as medicine (e.g., predicting disease presence), finance (e.g., predicting loan default), and marketing (e.g., predicting customer churn).

- Cost Function for Logistic Regression  
  The cost function (or loss function) for logistic regression is derived from the negative log-likelihood of the logistic regression model. It penalizes the model based on the difference between predicted probabilities and actual labels:

# Linear Regression

Linear regression is a fundamental supervised learning algorithm used to model the relationship between a dependent variable (target variable) and one or more independent variables (input features). It assumes a linear relationship between the variables.

## Terminology

- **x**: Input variable feature.
- **y**: Output variable or target variable.
- **m**: Number of training examples.
- **(x, y)**: Single training example.
- **(x^(i), y^(i))**: ith training example.

## Representation of a Line

In linear regression, we represent a line using the equation:
\[ f(w, b) = wx + b \]
where:
- \( w \) is the weight (slope) parameter.
- \( b \) is the bias (intercept) parameter.

## Objective

The goal of linear regression is to find the values of \( w \) and \( b \) that best fit the given data.

## Cost Function

The cost function measures the error between the predicted values and the actual values. For linear regression, the cost function is often the Mean Squared Error (MSE):
\[ J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (f(w, b; x^(i)) - y^(i))^2 \]
where:
- \( m \) is the number of training examples.
- \( f(w, b; x^(i)) = wx^(i) + b \) is the predicted value for the ith training example.

## Optimization Goal

The goal is to minimize the cost function \( J(w, b) \) with respect to \( w \) and \( b \), thereby finding the optimal parameters \( w \) and \( b \) that best fit the data.

## Gradient Descent

Gradient descent is an iterative optimization algorithm used to minimize the cost function. It works by updating the parameters \( w \) and \( b \) in the opposite direction of the gradient of the cost function \( J(w, b) \) with respect to \( w \) and \( b \):
\[ w := w - \alpha \frac{\partial J}{\partial w} \]
\[ b := b - \alpha \frac{\partial J}{\partial b} \]
where \( \alpha \) is the learning rate, a hyperparameter that controls the step size of each iteration.

## Learning Rate

The learning rate \( \alpha \) determines how much we update the parameters \( w \) and \( b \) in each iteration of gradient descent. 
- If \( \alpha \) is too large, gradient descent may overshoot the minimum and fail to converge, causing instability.
- If \( \alpha \) is too small, gradient descent may take a long time to converge to the minimum.

Adjusting the learning rate is crucial to ensure gradient descent converges efficiently.

---

This README provides an overview of linear regression, its key concepts, and the optimization process using gradient descent. Feel free to explore the implementations and examples in this folder to deepen your understanding of linear regression and its application in machine learning.

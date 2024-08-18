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

### Neural Networks  
Neural networks are just another way to do supervised machine learning. They can be used to do both regression and classification. They are preferred over normal logistic and linear regression because they can learn how to do feature engineering by themselves. In classification, they can classify more than one classes.
- Learnt how neural networks make predictions (forward propagation).
- How to train a neural network using back propagation.
- Learnt about different activations functions like sigmoid, tanh and relu including their use cases.
- Learnt how to appropriately choose the size of the neural network.

### Project -1 : Sign Language Training
I used a kaggle sign lanugage dataset which had 26 alphabets as 28x28 pixel intensities in sign language. With 34k data samples. Each sample had 784 features. I designed a neural network to train it on the sample dataset and achieved an accuracy of 80%.  
Things I learn’t from this project:
- The size of batch while training matters a lot. Using a small batch with too many features may often lead to overfitting of data.
- The number of layers in a neural network and nodes in each layers matter. If we have too many nodes, the data may overfit for a batch and perform poorly for other batches.
- The initialization of weights and biases should optimally be from np.random.randn. If, we initalize it as static number, we may fail to get convergence.
- I am still currently using all of the 34k samples to achieve an accuracy of 80%. I may optimally try to make it more efficient or achieve the same accuracy with a less number of samples.
- I can do this by experimenting with the number of layers, nodes and activation functions. 

### Convolutional Neural Networks (CNNs)

 **Key Concepts**
- **Convolutions**:
  - **Definition**: Convolutions are mathematical operations used to apply a filter to an input image, extracting features such as edges, textures, and patterns.
  - **Purpose**: They help in detecting patterns in the data, which is crucial for tasks such as image classification and object detection.
- **Pooling**:
  - **Definition**: Pooling is a technique used to downsample feature maps, reducing their spatial dimensions while retaining the most important information.
  - **Types**: 
    - **Max Pooling**: Takes the maximum value from each pooling window.
    - **Average Pooling**: Computes the average value within the pooling window.  
    
**Hyperparameters**

- **Importance**: Proper declaration and tuning of hyperparameters are crucial for building effective models. Hyperparameters include learning rates, batch sizes, number of epochs, and network architecture details.
- **Impact**: They significantly influence the model's performance and training efficiency.

**Normalization**

- **Batch Normalization**:
  - **Definition**: Batch normalization is a technique to normalize the inputs of each layer to improve training stability and speed up convergence.
  - **Purpose**: Helps in reducing internal covariate shift, leading to more stable and faster training.

**Interpretation of Loss and Validation Curves**
- **Loss Curves**:
  - **Training Loss**: Shows how well the model is learning from the training data. A decreasing training loss generally indicates that the model is learning.
  - **Validation Loss**: Represents the model’s performance on unseen data. It helps in assessing if the model is overfitting or underfitting.
  
- **Curves Analysis**:
  - **Overfitting**: If the training loss continues to decrease while the validation loss starts increasing, the model may be overfitting.
  - **Underfitting**: If both training and validation losses are high, the model might be underfitting and may require a more complex architecture or longer training.

**Summary**
- Understanding convolutions and pooling operations is fundamental in CNNs.
- Properly tuning hyperparameters and using techniques like batch normalization are key to building effective neural networks.
- Interpreting loss and validation curves provides insights into the model’s learning process and helps in improving performance.

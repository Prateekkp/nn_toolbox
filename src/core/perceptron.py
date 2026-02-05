'''
Perceptron implementation for binary classification.

The Perceptron is a simple linear classifier that updates its weights based on misclassified examples. It iteratively adjusts the weights to minimize classification errors on the training data.
'''

import numpy as np
import plotly.express as px
import random

# AND gate dataset
and_gate_data = [
                (0, 0, 0),  
                (0, 1, 0),  
                (1, 0, 0),  
                (1, 1, 1)   
]

# OR gate dataset
or_gate_data = [
                (0, 0, 0),  
                (0, 1, 1),  
                (1, 0, 1),  
                (1, 1, 1)   
]

# Initialize weights and bias
w1 = random.uniform(-1, 1)
w2 = random.uniform(-1, 1)
w3 = random.uniform(-1, 1)
b = random.uniform(-1, 1)

learning_rate = 0.1

# Define the activation function
def activation_function(x):
    return 1 if x >= 0 else 0

# Max epochs
max_epochs = 1000

# Training function
for epoch in range(max_epochs):
    print(f'Epoch {epoch+1}')
    total_error = 0

    for x1, x2, y in and_gate_data: 

        # forward pass
        weighted_sum = w1 * x1 + w2 * x2 + w3 * 1 + b  
        y_pred = activation_function(weighted_sum)

        # calculate error
        error = y - y_pred
        total_error += abs(error)

        # update if wrong
        if error != 0:
            w1 += learning_rate * error * x1
            w2 += learning_rate * error * x2
            w3 += learning_rate * error * 1  
            b += learning_rate * error  
        
    print(f'Weights: w1={w1:.2f}, w2={w2:.2f}, w3={w3:.2f}, b={b:.2f}, Total Error: {total_error}')

    if total_error == 0:
        print('Training complete.')
        break
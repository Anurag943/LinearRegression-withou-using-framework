import numpy as np
import copy
import math

def compute_gradient(x, y, w, b):
    """
    Compute the gradient for linear regression 

    Args : 
        x (ndarray) : input to the model
        y (ndarray) : target value
        w,b (scalar) : parameters of the model

    Returns : 
        dj_dw (scalar) : The gradient of the cost w.r.t. the parameters w
        dj_db (scalar) : The gradient of the cost w.r.t. the parameter b
    """

    # Number of training examples
    m = x.shape[0]

    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db



def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs gradient descent to fit w,b. Updates w,b by taking
    num_iters gradient steps with learning rate alpha

    Args:
        x (ndarray (m,)) : Data, m examples
        y (ndarray (m,)) : target values
        w_in, b_in (scalar) : initial values of model parameters
        cost_function : function to call to produce cost
        gradient_dunction : function to call to produce gradient
        alpha (float) : learning rate
        num_iters (int) : number of iterations to run gradient descent

    Returns :
        w (scalar) : Updated value of parameter after running gradient descent
        b (scalar) : Updated value of parameter after running gradient descent
        J_history (List) : History of cost values
        w_history (List) : History of parameter w
    """

    # Number of training examples
    m = len(x)

    # Array to store cost J and w's at each iteration for visualization purpose
    J_history = []
    w_history = []

    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # Update parameters w, b and alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = cost_function(x, y, w, b)
        J_history.append(cost)

        # Print the cost every at intery 10 times or as many itaration if < 10
        if i % math.ceil(num_iters / 10 ) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")

    return w, b, J_history, w_history
import numpy as np

def compute_cost(x, y, w, b):
    """
    Compute the cost function for linear regression

    Args : 
        x (ndarray) : Shape (m,) Input to the model
        y (ndarray) : target value
        w,b (scalar) : Parameters of model

    Returns :
        total_cost (float) : The cost of using w,b as the parameters for linear regression
                            to fit the data points in x and y.
    """

    m = x.shape[0]
    total_cost = 0
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i])**2
    total_cost = cost / (2 * m) 

    return total_cost

def mean_squared_error(actual, predicted):
    """
    Calcualte the mean squard error between true and predicted values.

    Parameters : 
        actual : Array of true values
        predicted : Array of predicted values

    Returns: 
        Mean Squared Error
    """
    
    n = len(actual)
    squared_diff = 0
    # Calculate the sum of squared differences
    for i in range(n):
        squared_diff += ((actual[i] - predicted[i])**2)

    mse = squared_diff / n

    return mse
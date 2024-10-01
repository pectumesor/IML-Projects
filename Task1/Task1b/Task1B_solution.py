# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps.
# First, we import necessary libraries:
import numpy as np
import pandas as pd

# Add any additional imports here (however, the task is solvable without using
# any additional imports)
# import ...

from skimage.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import  RidgeCV

def transform_data(X):
    """
    This function transforms the 5 input features of matrix X (x_i denoting the i-th component of X)
    into 21 new features phi(X) in the following manner:
    5 linear features: phi_1(X) = x_1, phi_2(X) = x_2, phi_3(X) = x_3, phi_4(X) = x_4, phi_5(X) = x_5
    5 quadratic features: phi_6(X) = x_1^2, phi_7(X) = x_2^2, phi_8(X) = x_3^2, phi_9(X) = x_4^2, phi_10(X) = x_5^2
    5 exponential features: phi_11(X) = exp(x_1), phi_12(X) = exp(x_2), phi_13(X) = exp(x_3), phi_14(X) = exp(x_4), phi_15(X) = exp(x_5)
    5 cosine features: phi_16(X) = cos(x_1), phi_17(X) = cos(x_2), phi_18(X) = cos(x_3), phi_19(X) = cos(x_4), phi_20(X) = cos(x_5)
    1 constant feature: phi_21(X)=1

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features

    Returns
    ----------
    X_transformed: array of floats: dim = (700,21), transformed input with 21 features
    """
    X_transformed = np.zeros((700, 21))
    # TODO: Enter your code here

    X_lin = X
    X_quad = np.power(X, 2)
    X_exp = np.exp(X)
    X_cos = np.cos(X)
    feature_21 = np.ones((700, 1))

    # concatenate all the features
    X_transformed = np.concatenate((X_lin, X_quad, X_exp, X_cos, feature_21), axis=1)

    assert X_transformed.shape == (700, 21)
    return X_transformed


def fit(X, y):
    """
    This function receives training data points, transforms them, and then fits the linear regression on this
    transformed data. Finally, it outputs the weights of the fitted linear regression.

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features
    y: array of floats, dim = (700,), input labels)

    Returns
    ----------
    w: array of floats: dim = (21,), optimal parameters of linear regression
    """
    w = np.zeros((21,))
    X_transformed = transform_data(X)
    alphas = np.linspace(0.00001, 0.1, 1000)
    # TODO: Enter your code here
    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
    ridge = RidgeCV(alphas=alphas, fit_intercept=False, cv=cv)
    ridge.fit(X_transformed, y)
    RMSE = mean_squared_error(y, ridge.predict(X_transformed)) ** 0.5
    w = ridge.coef_

    print(RMSE, w)
    assert w.shape == (21,)
    return w

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns=["Id", "y"])
    # print a few data samples
    print(data.head())

    X = data.to_numpy()
    # The function retrieving optimal LR parameters
    w = fit(X, y)
    # Save results in the required format
    np.savetxt("./results.csv", w, fmt="%.12f")

# This serves as a template which will guide you through the implementation of this task. It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps.
# First, we import necessary libraries:
import pandas as pd
import numpy as np

# Add any additional imports here (however, the task is solvable without using
# any additional imports)
# import ...
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge


def average_LR_RMSE(X, y, lambdas, n_folds):
    """
    Main cross-validation loop, implementing 10-fold CV. In every iteration (for every train-test split), the RMSE for every lambda is calculated, 
    and then averaged over iterations.
    
    Parameters
    ---------- 
    X: matrix of floats, dim = (150, 13), inputs with 13 features
    y: array of floats, dim = (150, ), input labels
    lambdas: list of floats, len = 5, values of lambda for which ridge regression is fitted and RMSE estimated
    n_folds: int, number of folds (pieces in which we split the dataset), parameter K in KFold CV
    
    Returns
    ----------
    avg_RMSE: array of floats: dim = (5,), average RMSE value for every lambda
    """
    RMSE_mat = np.zeros((n_folds, len(lambdas)))

    groups = np.linspace(1,200,150)

    # Fill all entries in the matrix 'RMSE_mat'

    for i in range (len(lambdas)):
        curr_lambda = lambdas[i]
        ridge = Ridge(alpha=curr_lambda,fit_intercept=True,random_state=42)
        cv = GroupShuffleSplit(n_splits=n_folds, test_size=0.3, random_state=42)
        scores = cross_val_score(ridge,X,y,scoring = 'neg_root_mean_squared_error',cv=cv.split(X,y,groups))
        RMSE_mat[:,i] = abs(scores)


    avg_RMSE = np.mean(RMSE_mat, axis=0)
    assert avg_RMSE.shape == (5,)
    return avg_RMSE


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns="y")
    # print a few data samples
    print(data)

    X = data.to_numpy()
    # The function calculating the average RMSE
    lambdas = [0.1, 1, 10, 100, 200]
    n_folds = 10
    avg_RMSE = average_LR_RMSE(X, y, lambdas, n_folds)
    print(avg_RMSE)
    # Save results in the required format
    np.savetxt("./results.csv", avg_RMSE, fmt="%.12f")

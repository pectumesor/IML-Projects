# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.experimental import enable_iterative_imputer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, ShuffleSplit


def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("train.csv")

    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(10))
    print('\n')

    # Load test data
    test_df = pd.read_csv("test.csv")

    print("Test data:")
    print(test_df.shape)
    print(test_df.head(10))

    non_null_counts = train_df.notnull().sum()

    # Plot
    plt.figure(figsize=(10, 6))
    non_null_counts.plot(kind='bar', color='skyblue')
    plt.title('Number of Non-Null Entries per Feature')
    plt.xlabel('Features')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot shows that all features have similar non null values, thus we dont drop any feature
    # Plot also shows that season feature has no null values, it may be important



    # Dummy initialization of the X_train, X_test and y_train
    # TODO: Depending on how you deal with the non-numeric data, you may want to
    # modify/ignore the initialization of these variables

    # Perform Ordinal on to the categorial data in the data frame

    X_numerical = encode_categorial(train_df)
    X_test = encode_categorial(test_df)

    # TODO: Perform data preprocessing, imputation and extract X_train, y_train and X_test

    X_train = impute_missing_data(X_numerical)
    y_train = X_train['price_CHF']
    X_test = impute_missing_data(X_test)
    X_train = X_train.drop(['price_CHF'], axis=1)

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (
                X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test


def impute_missing_data(X):
    imp = IterativeImputer(max_iter=10, random_state=42, missing_values=np.nan, initial_strategy='most_frequent')
    imp.fit(X)
    X_imputed = imp.transform(X)
    X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

    return X_imputed


def encode_categorial(df):
    enc = preprocessing.OrdinalEncoder()
    enc.fit(df[["season"]])
    df[["season"]] = enc.transform(df[["season"]])
    return df



def modeling_and_prediction(X_train, y_train, X_test):
    """
    This function defines the model, fits training data and then does the prediction with the test data

    Parameters
    ----------
    X_train: matrix of floats, training input with 10 features
    y_train: array of floats, training output
    X_test: matrix of floats: dim = (100, ?), test input with 10 features

    Returns
    ----------
    y_test: array of floats: dim = (100,), predictions on test set
    """
    # TODO: Define the model and fit it using training data. Then, use test data to make predictions

    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=42)

    gprMatern = GaussianProcessRegressor(kernel=Matern(),random_state=42,n_restarts_optimizer=6)
    gprMatern.fit(X_train, y_train)

    scoreMatern = cross_val_score(gprMatern, X_train, y_train, cv=cv, scoring='r2')
    scoreMatern = scoreMatern.mean()
    print("Accuracy Score for Kernel Matern:", scoreMatern)

    y_pred = gprMatern.predict(X_test)
    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred



# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    # The function retrieving optimal LR parameters
    y_pred = modeling_and_prediction(X_train, y_train, X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred)
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")


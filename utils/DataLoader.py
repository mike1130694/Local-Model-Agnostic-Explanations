"""
Created by Nikolas Bayer
"""

import pandas as pd
import pickle



def data_loader(path='data/german_credit_dataset.csv', raw=False, remove_outliers=True):
    """
    Loads the german credit data
    :param raw: If true, reads the raw dataframe, else it removes columns unused for training
    :param path: Path to file
    :return: dataframe
    """
    df = pd.read_csv(path, index_col=0)

    if raw:
        return df
    else:
        # Create numeric label variable for investigation and model building
        df['label'] = df['classification_'].map({'Approved': 0, 'Rejected': 1})
        # Drop unnecessary columns
        #df.drop(columns=['other_debtors_', 'people_liable_', 'classification_', 'Unnamed: 0'], inplace=True)
        df.drop(columns=['foreign_worker_', 'status_sex_', 'classification_'], inplace=True)
        # Change categories of previous_loans variable
        df['previous_loans_'].replace(['4 or 5', '6 or more'], '4 or more', inplace=True)

        if remove_outliers:
            # Identify thresholds for outliers
            IQR = df['amount_'].quantile(0.75) - df['amount_'].quantile(0.25)
            upper_limit_3iqr = df['amount_'].quantile(0.75) + 3 * IQR
            df.loc[df['amount_'] > upper_limit_3iqr, 'amount_'] = upper_limit_3iqr

        return df


def data_preprocess_nn(df, raw=False, remove_outliers=True):
    """
    Preprocesses data to be usable by the neural network
    :param raw: True, if passed dataframe is still raw
    :param df: credit data
    :return:
    """
    if raw:
        # Create numeric label variable for investigation and model building
        df['label'] = df['classification_'].map({'Approved': 0, 'Rejected': 1})
        # Drop unnecessary columns
        #df.drop(columns=['other_debtors_', 'people_liable_', 'classification_', 'Unnamed: 0'], inplace=True)
        df.drop(columns=['foreign_worker_', 'status_sex_', 'classification_'], inplace=True)
        # Change categories of previous_loans variable
        df['previous_loans_'].replace(['4 or 5', '6 or more'], '4 or more', inplace=True)
        if remove_outliers:
            # Identify thresholds for outliers
            IQR = df['amount_'].quantile(0.75) - df['amount_'].quantile(0.25)
            upper_limit_3iqr = df['amount_'].quantile(0.75) + 3 * IQR
            df.loc[df['amount_'] > upper_limit_3iqr, 'amount_'] = upper_limit_3iqr

    # Create target array and feature matrix
    X = df.drop(columns='label')
    y = df['label'].values

    # Scaling of numerical features
    with open('Preprocessor_ey.pickle', 'rb') as f:
        preprocessor = pickle.load(f)

    X = preprocessor.transform(X)

    return X, y

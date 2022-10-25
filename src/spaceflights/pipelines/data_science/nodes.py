import mlrun
from mlrun.frameworks.sklearn import apply_mlrun
from mlrun.artifacts import get_model, update_model

import logging
from typing import Dict, Tuple
from cloudpickle import load

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

@mlrun.handler(inputs=["data", "parameters"], outputs=["X_train", "X_test", "y_train", "y_test"])
def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    X = data[parameters["features"]]
    y = data["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test

@mlrun.handler(inputs=["X_train", "y_train"])
def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> LinearRegression:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    regressor = LinearRegression()
    apply_mlrun(model=regressor, model_name="regressor")
    regressor.fit(X_train, y_train)
    return regressor


@mlrun.handler(inputs=["model", "X_test", "y_test"])
def evaluate_model(
    regressor, X_test: pd.DataFrame, y_test: pd.DataFrame
):
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    model_file, model_obj, _ = get_model(regressor, suffix=".pkl")
    model = load(open(model_file, "rb"))
    
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2 of %.3f on test data.", score)

    update_model(regressor, metrics={"r2": score})
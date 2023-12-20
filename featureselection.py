"""Helper module for EDA notebook to perform 
feature selection and model training"""
import pandas as pd
import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    multilabel_confusion_matrix,
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
)
from sklearn.model_selection import StratifiedKFold, KFold
import statsmodels.api as sm


def prediction_postprocess(y_pred: pd.Series) -> pd.DataFrame:
    """ "Post-process predicted values to match outcomes"""
    prediction = pd.DataFrame(np.asarray(y_pred).argmax(1), columns=["result"])
    prediction["match_outcome"] = "away"
    prediction.loc[prediction["result"] == 1, "match_outcome"] = "draw"
    prediction.loc[prediction["result"] == 2, "match_outcome"] = "home"
    return prediction


def backward_elimination_MNLogit(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    significance_level: float = 0.05,
) -> None:
    """Perform backward elimination for the MNLogit model
    until all the faetures have p-values under the significant level."""
    num_features: int = X_train.shape[1]
    model = sm.MNLogit(y_train, X_train)
    try:
        model_fit = model.fit(maxiter=500)
    except (ValueError, np.linalg.LinAlgError) as e:
        print("An error occurred during model fitting:")
        print(str(e))
        return None

    print("Initial Model Summary:")
    y_pred: pd.DataFrame = model_fit.predict(X_val)
    prediction: pd.DataFrame = prediction_postprocess(y_pred)
    print(
        "Confusion Matrix : \n",
        multilabel_confusion_matrix(y_val, prediction["match_outcome"]),
    )
    print(
        "Accuracy score: ", accuracy_score(y_val, prediction["match_outcome"])
    )
    print(classification_report(y_val, prediction["match_outcome"]))
    print("Performing backward elimination.\n")

    rounds_count: int = 0
    coefficients: npt.NDarray = model_fit.params.values.reshape(1, -1)
    p_values: npt.NDarray = model_fit.pvalues.values.reshape(1, -1)
    nan_feature_indices: npt.NDarray = np.isnan(p_values)
    for i in range(num_features):
        if np.isnan(p_values).any():
            nan_feature_coefficients: npt.NDarray = coefficients[
                nan_feature_indices
            ]
            min_coefficient: float = np.min(np.abs(nan_feature_coefficients))
            min_index: int = np.where(coefficients[0] == min_coefficient)
            feature_index: int = min_index[0] // 2
        elif p_values.max() > significance_level:
            max_pvalue_index: int = np.unravel_index(
                np.argmax(p_values), p_values.shape
            )
            feature_index: int = max_pvalue_index[1] // 2
        else:
            break

        print(f"feature_index: {feature_index}")
        dropped_feature = X_train.columns[feature_index]
        X_train = X_train.drop(X_train.columns[feature_index], axis=1)
        X_val = X_val.drop(X_val.columns[feature_index], axis=1)
        num_features -= 1

        model = sm.MNLogit(y_train, X_train)
        try:
            model_fit = model.fit(maxiter=500)
        except (ValueError, np.linalg.LinAlgError) as e:
            print("An error occurred during model fitting:")
            print(str(e))
            return None

        print(f"Feature {dropped_feature} dropped.")
        print(f"Pvalue: {p_values.max()}.")

        p_values = model_fit.pvalues.values.reshape(1, -1)
        nan_feature_indices = np.isnan(p_values)
        coefficients = model_fit.params.values.reshape(1, -1)
        y_pred = model_fit.predict(X_val)
        prediction = prediction_postprocess(y_pred)
        rounds_count += 1

    print(f"\nBackward elimination completed with {rounds_count} rounds.")
    print("\nFinal Model Summary:")

    y_pred = model_fit.predict(X_val)
    prediction = prediction_postprocess(y_pred)
    f1: float = f1_score(
        y_val, prediction["match_outcome"], average="weighted"
    )

    print(f"Number of features: {model_fit.params.shape[0]}")
    print(f"Significance level: {significance_level:.3f}")
    print(
        "Confusion Matrix : \n",
        multilabel_confusion_matrix(y_val, prediction["match_outcome"]),
    )
    print(
        "Accuracy score: ", accuracy_score(y_val, prediction["match_outcome"])
    )
    print("F1-score: ", f1)
    print(classification_report(y_val, prediction["match_outcome"]))

    return (model_fit, f1)


def model_selection_MNLogit(
    X: pd.DataFrame,
    y: pd.Series,
    significance_levels: list[float],
    n_splits: int,
):
    """Perform cross validation for the model MNLogit to estimate
    the best threshold value for backward elimination.
    The best performing model is selected based on F1 score."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    model_fits: list = []
    f1_list: list[float] = []

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print("*" * 100)
        print(f"Fold {i}")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model_fit, f1 = backward_elimination_MNLogit(
            X_train, y_train, X_test, y_test, significance_levels[i]
        )
        if model_fit is not None:
            model_fits.append(model_fit)
            f1_list.append(f1)

    print("*" * 100)
    print(
        f"\nBased on F1 score the selected model is Model {np.argmax(f1_list)+1}"
    )
    print(
        f"Number of features: {model_fits[np.argmax(f1_list)].params.shape[0]}"
    )
    print(
        f"Significance level: {significance_levels[np.argmax(f1_list)]:.3f}"
    )
    print(f"F1-score: {np.argmax(f1_list):.3f}")

    return model_fits[np.argmax(f1_list)]


def backward_elimination_Poisson(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    significance_level: float = 0.05,
) -> None:
    """Perform backward elimination for the Poisson regression model
    until all the faetures have p-values under the significant level."""
    num_features: int = X_train.shape[1]
    model = sm.GLM(y_train, X_train, family=sm.families.Poisson())
    try:
        model_fit = model.fit()
    except (ValueError, np.linalg.LinAlgError) as e:
        print("An error occurred during model fitting:")
        print(str(e))
        return None

    print("Initial Model Summary:")
    y_pred: pd.DataFrame = model_fit.predict(X_val)
    prediction = np.random.poisson(y_pred)

    print("Performing backward elimination.\n")

    rounds_count: int = 0
    coefficients: pd.Series = model_fit.params
    p_values: pd.Series = model_fit.pvalues
    nan_feature_indices: pd.Series = p_values.isnull()
    for i in range(num_features):
        if np.isnan(p_values).any():
            nan_feature_coefficients: pd.Series = coefficients.loc[
                nan_feature_indices
            ]
            feature_index: str = nan_feature_coefficients.idxmin()
        elif p_values.max() > significance_level:
            feature_index: str = model_fit.pvalues.idxmax()
        else:
            break

        print(f"feature_index: {feature_index}")
        X_train = X_train.drop(columns=[feature_index])
        X_val = X_val.drop(columns=[feature_index])
        num_features -= 1

        model = sm.GLM(y_train, X_train, family=sm.families.Poisson())
        try:
            model_fit = model.fit()
        except (ValueError, np.linalg.LinAlgError) as e:
            print("An error occurred during model fitting:")
            print(str(e))
            return None

        print(f"Feature {feature_index} dropped.")
        print(f"Pvalue: {p_values.max()}.")

        coefficients: pd.Series = model_fit.params
        p_values: pd.Series = model_fit.pvalues
        nan_feature_indices: pd.Series = p_values.isnull()
        y_pred = model_fit.predict(X_val)
        prediction = np.random.poisson(y_pred)
        rounds_count += 1

    print(f"\nBackward elimination completed with {rounds_count} rounds.")
    print("\nFinal Model Summary:")

    y_pred: pd.Series = model_fit.predict(X_val)
    prediction: pd.Series = np.random.poisson(y_pred)
    mae: float = mean_absolute_error(y_val, prediction)

    print(f"Number of features: {model_fit.params.shape[0]}")
    print(f"Significance level: {significance_level:.3f}")
    print(f"Mean absolute error: {mae:.3f}")

    return (model_fit, mae)


def model_selection_Poisson(
    X: pd.DataFrame,
    y: pd.Series,
    significance_levels: list[float],
    n_splits: int,
):
    """Perform cross validation for the model Poisson regression to estimate
    the best threshold value for backward elimination.
    The best performing model is selected based on mean absolute error."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    model_fits = []
    mae_list = []

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        print("*" * 100)
        print(f"Fold {i}")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model_fit, mae = backward_elimination_Poisson(
            X_train, y_train, X_test, y_test, significance_levels[i]
        )
        if model_fit is not None:
            model_fits.append(model_fit)
            mae_list.append(mae)

    print("*" * 100)
    print(
        f"\nBased on mean absolute error the selected model is Model {np.argmin(mae_list)+1}"
    )
    print(
        f"Number of features: {model_fits[np.argmin(mae_list)].params.shape[0]}"
    )
    print(
        f"Significance level: {significance_levels[np.argmin(mae_list)]:.3f}"
    )
    print(f"Mean absolute error: {np.min(mae_list):.3f}")

    return model_fits[np.argmin(mae_list)]

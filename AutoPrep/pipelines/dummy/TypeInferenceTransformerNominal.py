# %%
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.utils.validation import check_array, check_is_fitted
import pdb
import warnings
# Warnungen vom Versuch des castens ignorieren
warnings.filterwarnings("ignore")

from sklearn import set_config

set_config(transform_output="pandas")


class TypeInferenceTransformerNominal(BaseEstimator, TransformerMixin):
    """
    SchemaTransformer for a certain Pandas DataFrame input.

    Steps:
        (1) Attempt to convert object columns into a better data type format.\n
        (2) Attempt to convert columns with a time series schema into the correct data type.\n
        (3) Attempt to convert numerical data with the incorrect data type into the correct data type.\n
            Example: "col1": [1, "2", 3]  to "col1": [1, 2, 3]\n
        (4) NaN values are formatted correctly for subsequent processing.\n
        (5) Return of the adjusted dataframe.\n


    Parameters
    ----------
    datetime_columns : list
        List of certain Time-Columns that should be converted in timestamp data types.

    exclude_columns : list
        List of Columns that will be dropped.

    name_transformer : list
        Is used for the output, so the enduser can check what Columns are used for a certain Transformation.

    nominal_columns : list
        Only nominal_columns will be transformed.

    """

    def __init__(
        self,
        datetime_columns=None,
        exclude_columns: list = None,
        nominal_columns: list = None,
        name_transformer="",
    ):
        self.datetime_columns = datetime_columns
        self.nominal_columns = nominal_columns

        self.exclude_columns = exclude_columns
        self.feature_names = None
        self.name_transformer = name_transformer


    def convert_schema_nans(self, X):
        X_Copy = X.copy()

        for col in X_Copy.columns:
            X_Copy[col] = X_Copy[col].replace("NaN", np.nan)
            X_Copy[col] = X_Copy[col].replace("nan", np.nan)
            X_Copy[col] = X_Copy[col].replace(" ", np.nan)
            X_Copy[col] = X_Copy[col].replace("", np.nan)
        return X_Copy


    def fit(self, X, y=None):
        return self

    def transform(self, X) -> pd.DataFrame:
        if self.nominal_columns is None:
            self.nominal_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        else:
            missing_cols = [col for col in self.nominal_columns if col not in X.columns]
            if missing_cols:
                raise KeyError(f"Columns not found in the DataFrame: {missing_cols}")

        X = X[self.nominal_columns]

        if self.exclude_columns is not None:
            self.exclude_columns = [col for col in self.exclude_columns if col in X.columns]
            X.drop(columns=self.exclude_columns, inplace=True)

        self.feature_names = X.columns

        X_copy = X.copy()
        X_copy = self.convert_schema_nans(X_copy)

        X_copy = X_copy.astype(str)


        return X_copy


    def get_feature_names(self, input_features=None):
        return self.feature_names


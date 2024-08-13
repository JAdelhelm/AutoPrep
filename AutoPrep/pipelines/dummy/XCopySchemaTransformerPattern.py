# %%
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline

import warnings

# Warnungen vom Versuch des castens ignorieren
warnings.filterwarnings("ignore")

from sklearn import set_config

set_config(transform_output="pandas")


class XCopySchemaTransformerPattern(BaseEstimator, TransformerMixin):
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

    include_columns : list
        List of Columns for pattern recognition.

    name_transformer : list
        Is used for the output, so the enduser can check what Columns are used for a certain Transformation.

    """

    def __init__(
        self, datetime_columns=None, include_columns = None, name_transformer=""
    ):
        self.datetime_columns = datetime_columns
        self.include_columns = include_columns
        self.feature_names = None
        self.name_transformer = name_transformer

        if isinstance(self.include_columns, list) is False:
            raise ValueError("Columns for pattern recognition has to be defined with pattern_recognition_columns!")

    def convert_schema_nans(self, X):
        X_Copy = X.copy()

        for col in X_Copy.columns:
            X_Copy[col] = X_Copy[col].replace("NaN", np.nan)
            X_Copy[col] = X_Copy[col].replace("nan", np.nan)
            X_Copy[col] = X_Copy[col].replace(" ", np.nan)
            X_Copy[col] = X_Copy[col].replace("", np.nan)
        return X_Copy

    def infer_schema_X(self, X_copy):
        try:
            X_copy = X_copy.infer_objects()
        except:
            pass

        for col in X_copy.columns:
            if X_copy[col].dtype == "object":
                if self.datetime_columns is not None and col in self.datetime_columns:
                    try:
                        X_copy[col] = pd.to_datetime(
                            X_copy[col], infer_datetime_format=True, errors="coerce"
                        )
                        # print("\nColumns to time dtype:", col, "\n")
                    except:
                        pass
                else:
                    try:
                        X_copy[col] = pd.to_datetime(
                            X_copy[col], infer_datetime_format=True
                        )
                        # print("\nColumns to time dtype:", col, "\n")
                    except:
                        pass

                try:
                    X_copy[col] = X_copy[col].astype(np.float64)
                    # print("\nColumns to numeric dtype:", col, "\n")
                except (ValueError, TypeError):
                    pass

        X_copy.convert_dtypes()

        return X_copy

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> pd.DataFrame:

        all_columns = X.columns.tolist()

        exclude_columns = [col for col in all_columns if col not in self.include_columns]

        if exclude_columns is not None:
            for col in exclude_columns:
                try:
                    # X.drop([col], axis=1, inplace=True)
                    X = X.drop(columns=exclude_columns, axis=1)

                except:
                    print(f"Column {col} could not be dropped.")

        self.feature_names = X.columns

        X_copy = X.copy()
        X_copy = self.convert_schema_nans(X_copy)

        X_copy = self.infer_schema_X(X_copy=X_copy)

        print(f"\n\nDtypes-Schema / Columns for {self.name_transformer}:\n")
        print(X_copy.dtypes, "\n")

        return X_copy

    def get_feature_names(self, input_features=None):
        return self.feature_names

    # def convert_column_to_naive_timestamp(self, X, column_name):
    #     """
    #     Konvertieren einer 'datetime64[ns, UTC]' Spalte zu 'datetime64[ns]'
    #     """
    #     try: return X[column_name].dt.tz_convert(None)
    #     except: pass


### Test

# test_daten = pd.DataFrame(
#     {
#         "COLTestCAT1": np.array(["Hund","Hund", "Hund123"]),
#         "COLTestCAT2": np.array(["K*atze","K*atze", np.nan]),
#         "timestamp": np.array(["2023-02-08 06:58:14.017000+00:00", "2023-02-08 15:54:13.693000+00:00", np.nan])
#     })

# train_daten = pd.DataFrame(
#     {
#         "COLTestCAT1": np.array(["Hund","Hund", "Hund123"]),
#         "COLTestCAT2": np.array(["K*atze","K*atze", np.nan]),
#         "timestamp": np.array(["2023-02-08 06:58:14.017000+00:00", "2023-02-08 15:54:13.693000+00:00", np.nan])
#     })

# df = pd.concat([test_daten, train_daten])

# preprocessor = make_pipeline(
#             ColumnTransformer(transformers=[
#                 ("XCopy", XCopySchemaTransformerPattern(include_columns=["availability"]), make_column_selector(dtype_include=None))
#             ], remainder="passthrough", n_jobs=-1),
#             ColumnTransformer(transformers=[
#                 ("C_imputed", SimpleImputer(strategy="most_frequent", missing_values=np.nan), make_column_selector(dtype_include=np.object_)),
#                 ("N_imputed", SimpleImputer(strategy="median"), make_column_selector(dtype_include=np.number)),
#             ], remainder="passthrough", n_jobs=-1)
#             )

# transformed_data = preprocessor.fit_transform(df)

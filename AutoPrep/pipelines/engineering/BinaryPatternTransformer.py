# %%
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector
from sklearn.compose import ColumnTransformer
from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd
import numpy as np
import re

from bitstring import BitArray
from collections import Counter

from sklearn import set_config

set_config(transform_output="pandas")


class BinaryPatternTransformer(BaseEstimator, TransformerMixin):
    """
    PatternTransformer to extract certain patterns from categorical data.

    This transformer is designed to extract patterns from features and represent them in a binary format:
        - Big Letters: "00" / Small Letters: "01" / Non-Alphanumeric: "10" / Remaining: "11"\n
    Example "Dog":
        -"00 01 01" translates to 40.

    Steps:
        (1) Iterate over every single Column in X.\n
        (2) Create two lists: temp_word_encoded and count_len\n
            (2.1) temp_word_encoded is used to save the corresponding decimals w.r.t. the binary transformation\n
        (3) Iterate over every row in X.\n
            (3.1) Convert that row into a binary format.\n
            (3.2) Convert that binary into an int/decimal.\n
        (4) Parse the corresponding values to a new DataFrame.\n
        (5) Return the new DataFrame with the corresponding length of every value entry and binary representation.\n

    """

    def __init__(self):
        self.new_feature_names = None

    def extract_patterns(self, X):
        X_Transformed = pd.DataFrame()

        for col in X.columns:
            # print("Actual column: ", col)

            temp_word_encoded = []
            count_len = []
            # count_unique = []
            for value in X[col].values:
                str_w_encoded = []
                count_len.append(len(value))
                # count_unique.append(len(set(value)))

                for character in value:
                    if character.isupper():
                        str_w_encoded.append("00")
                    elif character.islower():
                        str_w_encoded.append("01")
                    elif character.isdigit():
                        str_w_encoded.append("10")
                    else:
                        str_w_encoded.append("11")
                str_w_encoded = "0" + "".join(str_w_encoded)

                # Has to be str, else the BinaryEncoder cant handle it.
                str_w_encoded = str(BitArray(bin=str_w_encoded))
                temp_word_encoded.append(str_w_encoded)

            X_Transformed[col] = temp_word_encoded
            X_Transformed[col + "_len"] = count_len
            # X_Transformed[col + "_unique"] = count_unique

        self.new_feature_names = X_Transformed.columns
        
        if X_Transformed.columns.duplicated().any():
            raise ValueError("Duplicate columns found in X_Transformed")
        

        return X_Transformed

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if X.eq(0).all().all():
            return X
        else:
            transformed = self.extract_patterns(X)
            transformed.index = X.index
            return transformed

    def get_feature_names(self, input_features=None):
        return self.new_feature_names

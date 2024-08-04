from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn import set_config
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# from graphviz import Digraph
from category_encoders import BinaryEncoder


from sklearn.utils import estimator_html_repr



import os
from joblib import dump
import itertools
from pathlib import Path

from pipelines.runs import PipelineRuns

try:  from pipelines.configuration_control import ConfigurationControl
except: from configuration_control import ConfigurationControl

set_config(transform_output="pandas")

class AutoPrep():
    """
    The AutoPrep (Automated Preprocessing Pipeline with Univariate Anomaly Marking) class manages and executes configured pipelines 
    for anomaly detection, automating the preprocessing of data.

    Parameters
    ----------
    datetime_columns : list, optional
        List of column names representing time data that should be converted to timestamp data types. Default is None.

    nominal_columns : list, optional
        Columns that should be transformed to nominal data types. Default is None.

    ordinal_columns : list, optional
        Columns that should be transformed to ordinal data types. Default is None.

    exclude_columns : list, optional
        List of columns to be dropped from the dataset. Default is None.

    pattern_recognition_exclude_columns : list, optional
        List of columns to be excluded from pattern recognition. Default is None.

    exclude_columns_no_variance : bool, optional
        If set to True, all columns with zero standard deviation/variance will be removed. Default is True.

    deactivate_pattern_recognition : bool, optional
        If set to True, the pattern recognition transformer will be deactivated. Default is True.

    """

    def __init__(
        self,
        datetime_columns: list = None,
        nominal_columns: list = None,
        ordinal_columns: list = None,
        exclude_columns: list = None,
        pattern_recognition_exclude_columns: list = None,
        exclude_columns_no_variance: bool = True,
        deactivate_pattern_recognition: bool = True):

        self.datetime_columns = datetime_columns
        self.nominal_columns = nominal_columns
        self.ordinal_columns = ordinal_columns
        self.exclude_columns = exclude_columns
        self.pattern_recognition_exclude_columns = pattern_recognition_exclude_columns
        self.deactivate_pattern_recognition = deactivate_pattern_recognition


        self.exclude_columns_no_variance = exclude_columns_no_variance

        self.model_name = ""

        if self.datetime_columns is not None:
            if self.exclude_columns is None:
                self.exclude_columns = []
            self.exclude_columns = self.exclude_columns + self.datetime_columns



        config_control = ConfigurationControl(
            datetime_columns = self.datetime_columns,
            nominal_columns = self.nominal_columns,
            ordinal_columns = self.ordinal_columns,
            pattern_recognition_exclude_columns = self.pattern_recognition_exclude_columns,
            deactivate_pattern_recognition =self.deactivate_pattern_recognition
        )
        self.pipeline_structure = config_control.pipeline_configuration()

        self.runs = PipelineRuns(self.pipeline_structure,
                                 remove_columns_with_no_variance = self.exclude_columns_no_variance,
                                 exclude_columns = self.exclude_columns)
        

    def preprocess(
            self,
            df: pd.DataFrame
    )  -> pd.DataFrame:
        """
        Preprocesses the given DataFrame.

        This method applies a preprocessing pipeline to the input DataFrame,
        which may include operations such as encoding columns and other 
        transformations necessary for the dataset to be in a suitable form 
        for further analysis or modeling.

        Parameters:
        -----------
        df : pd.DataFrame
            The input DataFrame that needs to be preprocessed.

        Returns:
        --------
        pd.DataFrame
            The preprocessed DataFrame with all necessary transformations applied.
        """
        return self.runs.preprocess_pipeline(
            df = df
        )
          


    def visualize_pipeline_structure_html(self, filename="./visualization/PipelineDQ"):
        """
        Save the pipeline structure as an HTML file.

        This method creates the necessary directories (if they do not already exist) 
        and saves a visual representation of the pipeline structure to an HTML file.

        Parameters
        ----------
        filename : str, optional
            The path and filename for the HTML file. The default is "./visualization/PipelineDQ".

        Returns
        -------
        None
            This function does not return any value. It only saves the HTML file.

        """
        Path("./visualization").mkdir(parents=True, exist_ok=True)
        Path("./visualization/Pipeline").mkdir(parents=True, exist_ok=True)
        with open(file=f"{filename}.html", mode="w", encoding="utf-8") as f:
            f.write(estimator_html_repr(self.pipeline_structure))
            f.close()





    

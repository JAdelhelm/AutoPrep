from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion, make_union
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, MissingIndicator
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# from graphviz import Digraph
from category_encoders import BinaryEncoder


from sklearn.utils import estimator_html_repr


# import pdfkit

# Activate if you use PyTorch algorithms
# import torch
try:
    from AutoPrep.pipeline_configuration import PipelinesConfiguration
    from AutoPrep.pipelines.nan_handling.NaNColumnCreator import NaNColumnCreator

except ImportError:
    from pipeline_configuration import PipelinesConfiguration
    from pipelines.nan_handling.NaNColumnCreator import NaNColumnCreator



class PipelineControl(PipelinesConfiguration):
    """
    The ConfigurationControl class extends PipelinesConfiguration and manages the configuration 
    of preprocessing pipelines for anomaly detection.

    Parameters
    ----------
    datetime_columns : list, optional
        List of column names representing time data that should be converted to timestamp data types. Default is None.

    nominal_columns : list, optional
        Columns that should be transformed to nominal data types. Default is None.

    ordinal_columns : list, optional
        Columns that should be transformed to ordinal data types. Default is None.

    pattern_recognition_columns : list, optional
        List of columns to be included from pattern recognition. 

    exclude_columns : list, optional
        List of columns to be dropped from the dataset. Default is None.


    Methods
    -------
    standard_pipeline_configuration()
        Returns the standard pipeline configuration with profiling, datatypes, and preprocessing steps.

    pipeline_configuration()
        Returns the complete pipeline configuration based on provided columns and settings.

    """
    def __init__(self,
        datetime_columns: list = None,
        nominal_columns: list = None,
        ordinal_columns: list = None,
        numerical_columns: list = None,
        pattern_recognition_columns: list = None,
        exclude_columns: list = None,
        n_jobs: int = -1,
        activate_numeric_scaling: bool = False
                 ) -> None:
        super().__init__()
        self.datetime_columns = datetime_columns
        self.nominal_columns = nominal_columns
        self.ordinal_columns = ordinal_columns
        self.numerical_columns = numerical_columns
        self.pattern_recognition_columns = pattern_recognition_columns
        self.exclude_columns = exclude_columns
        self.n_jobs = n_jobs
        self.activate_numeric_scaling = activate_numeric_scaling

        self._all_columns = (self.nominal_columns + 
                             self.ordinal_columns + 
                             self.numerical_columns + 
                             self.datetime_columns)
        
        self.standard_pipeline = None
        self.categorical_columns = None



    def standard_dtype_transformer(self, df) -> pd.DataFrame:
        """
            - Infers dtypes of Dataframe before inject it to the main pipeline.
            - Excludes specified columns from DataFrame.
            - Find categorical columns that are not specified as parameter.
        """
        self.column_check_input_parameters(df=df)


        pipeline_type_inference = super().pre_pipeline()
        df_transformed = pipeline_type_inference.fit_transform(df)

        self.init_standard_pipeline()
        self.find_categorical_columns(df = df_transformed)

        return  df_transformed
    


    def init_standard_pipeline(self):
        self.standard_pipeline =  Pipeline(
            steps=[
                (
                    "Standard Preprocessing",
                    ColumnTransformer(
                        transformers=[
                            (
                                "numerical",
                                super().numeric_pipeline(),
                                make_column_selector(dtype_include=np.number),
                            ),
                            (
                                "date",
                                super().timeseries_pipeline(),
                                make_column_selector(
                                    dtype_include=(
                                        np.dtype("datetime64[ns]"),
                                        np.datetime64,
                                        "datetimetz",
                                    )
                                ),
                            ),
                            
                        ],
                        remainder="drop",
                        n_jobs=self.n_jobs,
                        verbose=True,
                    ),
                ),
                
            ]
        )


    def pipeline_control(self):
        """
        Configures and returns a preprocessing pipeline based on the presence of nominal and ordinal columns.

        Constructs a pipeline that applies numerical and time series transformations, with optional 
        handling for nominal and ordinal columns. If neither nominal nor ordinal columns are specified, 
        a standard pipeline is returned. Columns not specified as nominal or ordinal will be passed 
        through a BinaryEncoder in the transformation process.

        Returns:
            Pipeline or FeatureUnion: The configured pipeline for preprocessing the data.
        """               

        if len(self.nominal_columns) > 0 and len(self.ordinal_columns) == 0: 
            return FeatureUnion(
                    transformer_list=[
                        ("Standard", self.standard_pipeline),
                        ("Nominal", super().nominal_pipeline()),
                        ("NaN", super().nan_marker_pipeline()),
                        ("PatternExtraction", super().pattern_extraction())
                    ],
                    n_jobs=self.n_jobs
                )
        elif len(self.nominal_columns) == 0 and len(self.ordinal_columns) > 0: 
            return FeatureUnion(
                    transformer_list=[
                        ("Standard", self.standard_pipeline),
                        ("Ordinal", super().ordinal_pipeline()),
                        ("NaN", super().nan_marker_pipeline()),
                        ("PatternExtraction", super().pattern_extraction())
                    ],
                    n_jobs=self.n_jobs
                )
        elif len(self.nominal_columns) > 0 and len(self.ordinal_columns) > 0: 
            return FeatureUnion(
                    transformer_list=[
                        ("Standard", self.standard_pipeline),
                        ("Nominal", super().nominal_pipeline()),
                        ("Ordinal", super().ordinal_pipeline()),
                        ("NaN", super().nan_marker_pipeline()),
                        ("PatternExtraction", super().pattern_extraction())
                    ],
                    n_jobs=self.n_jobs
                )
            
        return self.standard_pipeline





    def find_categorical_columns(self, df):
        """
        Identifies categorical columns in the given DataFrame that have not been explicitly specified.

        This method scans the provided DataFrame `df` to determine which columns contain categorical 
        data types that were not predefined or specified by the user. These identified categorical 
        columns will be processed separately in an additional transformation step.
        """
        self.categorical_columns = list(df.select_dtypes(include=[object]).columns)

        try: 
            for col in self.nominal_columns:
                try:self.categorical_columns.remove(col)
                except: pass
        except: pass
        
        try:

            for col in self.ordinal_columns:
                try: self.categorical_columns.remove(col)
                except:pass
        except: pass


        if len(self.categorical_columns) > 0:
            self.standard_pipeline = FeatureUnion(
                    transformer_list=[
                        ("Standard", self.standard_pipeline),
                        ("categorical", super().categorical_pipeline() ) 
                        ],
                        n_jobs=self.n_jobs)



    def column_check_input_parameters(self, df):
        """
        Checks that all specified columns exist in the dataframe and that there are no duplicates.

        Parameters:
        -----------
        df : pandas.DataFrame
            The dataframe to validate against.

        Raises:
        -------
        KeyError:
            If any specified column is not found in the dataframe.
        ValueError:
            If duplicate columns are detected in the input parameters.
        """
        for col in self._all_columns:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in the dataframe.")

        if len(self._all_columns) != len(set(self._all_columns)):
            duplicate_columns = [col for col in set(self._all_columns) if self._all_columns.count(col) > 1]
            raise ValueError(f"Duplicate columns found: {duplicate_columns}")
"""
Module for configuring and creating preprocessing pipelines.

This module provides the `PipelinesConfiguration` class which includes various methods to create
and configure pipelines for preprocessing numerical, categorical, timeseries, and pattern data.

Classes:
    PipelinesConfiguration: Configures pipelines for preprocessing different types of data.
    XPatternDropper: Drops specified columns from the data.

Imports:
    numpy as np
    pandas as pd
    sklearn and other relevant libraries for data preprocessing and transformation.
"""

import numpy as np
import pandas as pd
try:
    from AutoPrep.pipelines.statistical.TukeyTransformer import TukeyTransformer
    from AutoPrep.pipelines.statistical.TukeyTransformerTotal import TukeyTransformerTotal
    from AutoPrep.pipelines.statistical.MedianAbsolutDeviation import MedianAbsolutDeviation
    from AutoPrep.pipelines.statistical.MedianAbsolutDeviationTotal import MedianAbsolutDeviationTotal
    from AutoPrep.pipelines.statistical.SpearmanCheck import SpearmanCorrelationCheck
    from AutoPrep.pipelines.engineering.BinaryPatternTransformer import BinaryPatternTransformer
    from AutoPrep.pipelines.dummy.TypeInferenceTransformer import TypeInferenceTransformer
    from AutoPrep.pipelines.dummy.TypeInferenceTransformerNominal import TypeInferenceTransformerNominal
    from AutoPrep.pipelines.dummy.TypeInferenceTransformerOrdinal import TypeInferenceTransformerOrdinal
    from AutoPrep.pipelines.dummy.TypeInferenceTransformerPattern import TypeInferenceTransformerPattern
    from AutoPrep.pipelines.timeseries.DateEncoder import DateEncoder
    from AutoPrep.pipelines.timeseries.TimeSeriesImputer import TimeSeriesImputer
    from AutoPrep.pipelines.nan_handling.NaNColumnCreator import NaNColumnCreator
    from AutoPrep.pipelines.nan_handling.NaNColumnCreatorTotal import NaNColumnCreatorTotal
except ImportError:
    from pipelines.statistical.TukeyTransformer import TukeyTransformer
    from pipelines.statistical.TukeyTransformerTotal import TukeyTransformerTotal
    from pipelines.statistical.MedianAbsolutDeviation import MedianAbsolutDeviation
    from pipelines.statistical.MedianAbsolutDeviationTotal import MedianAbsolutDeviationTotal
    from pipelines.statistical.SpearmanCheck import SpearmanCorrelationCheck
    from pipelines.engineering.BinaryPatternTransformer import BinaryPatternTransformer
    from pipelines.dummy.TypeInferenceTransformer import TypeInferenceTransformer
    from pipelines.dummy.TypeInferenceTransformerNominal import TypeInferenceTransformerNominal
    from pipelines.dummy.TypeInferenceTransformerOrdinal import TypeInferenceTransformerOrdinal
    from pipelines.dummy.TypeInferenceTransformerPattern import TypeInferenceTransformerPattern
    from pipelines.timeseries.DateEncoder import DateEncoder
    from pipelines.timeseries.TimeSeriesImputer import TimeSeriesImputer
    from pipelines.nan_handling.NaNColumnCreator import NaNColumnCreator
    from pipelines.nan_handling.NaNColumnCreatorTotal import NaNColumnCreatorTotal



from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, LabelEncoder, OrdinalEncoder, RobustScaler, StandardScaler, MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from category_encoders import BinaryEncoder
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn import set_config
import warnings
import itertools
try: 
    from ydata_profiling import ProfileReport
except:
    print("For Data profiling install ydata_profiling...")
set_config(transform_output="pandas")


class PipelinesConfiguration():
    """
    The PipelinesConfiguration class represents the class to configure pipelines for data preprocessing.

    There are different SchemaTransformer, to handle different datatypes as input.

    Methods
    -------
    pre_pipeline(datetime_columns=None, exclude_columns=None):
        Creates a preprocessing pipeline to prepare data for transformation.
        
    nan_marker_pipeline():
        Creates a pipeline that marks columns with NaN values.

    numeric_pipeline():
        Creates a pipeline for preprocessing numerical data.

    categorical_pipeline():
        Creates a pipeline for preprocessing categorical data.

    timeseries_pipeline():
        Creates a pipeline for preprocessing timeseries data.

    pattern_extraction(pattern_recognition_columns=None, datetime_columns_pattern=None, deactivate_pattern_recognition=False):
        Creates a pipeline to extract patterns from categorical data.

    nominal_pipeline(nominal_columns=None, datetime_columns=None):
        Creates a pipeline for separate preprocessing of nominal data.

    ordinal_pipeline(ordinal_columns=None, datetime_columns=None):
        Creates a pipeline for separate preprocessing of ordinal data.

    Parameters
    ----------
    datetime_columns : list
        List of Time-Columns that should be converted to timestamp data types.

    exclude_columns : list
        List of columns that should be dropped.
    """
    def __init__(self):
        """
        Inherits all attributes from PipelineControl
        """
        pass

    def pre_pipeline(self):

        original_preprocessor = Pipeline(
            steps=[
                (
                    "Preprocessing",
                    ColumnTransformer(
                        transformers=[
                            (
                                "Standard TypeCast Transformer",
                                TypeInferenceTransformer(
                                    datetime_columns=self.datetime_columns,
                                    exclude_columns=self.exclude_columns,
                                    numerical_columns = self.numerical_columns,
                                    name_transformer="Schema Standard Pipeline",
                                ),
                                make_column_selector(dtype_include=None),
                            )
                        ],
                        remainder="passthrough",
                        n_jobs=-1,
                        verbose=True,
                        # Disable Prefix behaviour
                        verbose_feature_names_out=False
                    ),
                )
            ]
        )

        return original_preprocessor

    def nan_marker_pipeline(self):
        nan_marker_preprocessor = Pipeline(
            steps=[
                (
                    "NanMarker",
                    ColumnTransformer(
                        transformers=[
                            (
                                "nan_marker_columns",
                                Pipeline(
                                    steps=[
                                        (
                                            "X_nan",
                                            TypeInferenceTransformer(
                                                numerical_columns = self.numerical_columns,
                                                datetime_columns=self.datetime_columns,
                                                name_transformer="Schema NaNMarker",
                                            ),
                                        ),
                                        ("nan_marker", MissingIndicator(features="all"))

                                    ]
                                ),
                                make_column_selector(dtype_include=None),
                            ),
                        ],
                        remainder="passthrough",
                        n_jobs=-1,
                        verbose=True,
                        verbose_feature_names_out=False
                    ),
                )
            ]
        )

        return nan_marker_preprocessor

    def numeric_pipeline(self):
        if self.scaler_option_num == "deactivate":
            return  Pipeline(
                steps=[
                    (
                        "Preprocessing_Numerical",
                        ColumnTransformer(
                            transformers=[
                                (
                                    "numeric",
                                    Pipeline(
                                        steps=[
                                            (
                                                "N",
                                                SimpleImputer(strategy="median", keep_empty_features=True),
                                            ),

                                        ]
                                    ),
                                    # make_column_selector(dtype_include=np.number),
                                    self.numerical_columns
                                ),
                            ],
                            remainder="passthrough",
                            n_jobs=-1,
                            verbose=True,
                            verbose_feature_names_out=False
                        ),
                    ),
                    (
                        "Statistical methods",
                        ColumnTransformer(
                            transformers=[
                                (
                                    "tukey",
                                    Pipeline(
                                        steps=[
                                            ("Tukey_impute", IterativeImputer(initial_strategy="median")),
                                            ("tukey", TukeyTransformer(factor=1.5)),
                                            ("tukey_total", TukeyTransformerTotal())
                                        ]
                                    ),
                                    make_column_selector(dtype_include=np.number),
                                ),
                                (
                                    "z_mod",
                                    Pipeline(
                                        steps=[
                                            ("z_mod_impute", IterativeImputer(initial_strategy="median")),
                                            ("z_mod", MedianAbsolutDeviation()),
                                            ("z_mod_total", MedianAbsolutDeviationTotal())
                                        ]
                                    ),
                                    make_column_selector(dtype_include=np.number),
                                ),
                                (
                                    "pass_cols",
                                    Pipeline(
                                        steps=[
                                            (
                                                "_pass_cols_",
                                                "passthrough",
                                            ),
                                            # (
                                            #     "robust_scaler",
                                            #     RobustScaler(),
                                            # ),
                                        ]
                                    ),
                                    make_column_selector(dtype_include=np.number),
                                ),
                            ],
                            remainder="passthrough",
                            n_jobs=-1,
                            verbose=True,
                            verbose_feature_names_out=False
                        ),
                    ),
                ]
            )
        else:
            return self.numeric_pipeline_2()



    def numeric_pipeline_2(self):
        self.scaler_pick = StandardScaler()

        if self.scaler_option_num == "standard": self.scaler_pick = StandardScaler()
        elif self.scaler_option_num == "robust": self.scaler_pick = RobustScaler()
        elif self.scaler_option_num == "minmax": self.scaler_pick = MinMaxScaler()

        if self.scaler_option_num not in ["standard", "robust", "minmax"]:
            raise ValueError("No valid scaler option picked!")


        return  Pipeline(
            steps=[
                (
                    "Preprocessing_Numerical",
                    ColumnTransformer(
                        transformers=[
                            (
                                "numeric",
                                Pipeline(
                                    steps=[
                                        (
                                            "N",
                                            SimpleImputer(strategy="median", keep_empty_features=True),
                                        ),

                                    ]
                                ),
                                make_column_selector(dtype_include=np.number),
                            ),
                        ],
                        remainder="passthrough",
                        n_jobs=-1,
                        verbose=True,
                        verbose_feature_names_out=False
                    ),
                ),
                (
                    "Statistical methods",
                    ColumnTransformer(
                        transformers=[
                            (
                                "tukey",
                                Pipeline(
                                    steps=[
                                        ("Tukey_impute", IterativeImputer(initial_strategy="median")),
                                        ("tukey", TukeyTransformer(factor=1.5)),
                                        ("tukey_total", TukeyTransformerTotal())
                                    ]
                                ),
                                make_column_selector(dtype_include=np.number),
                            ),
                            (
                                "z_mod",
                                Pipeline(
                                    steps=[
                                        ("z_mod_impute", IterativeImputer(initial_strategy="median")),
                                        ("z_mod", MedianAbsolutDeviation()),
                                        ("z_mod_total", MedianAbsolutDeviationTotal())
                                    ]
                                ),
                                make_column_selector(dtype_include=np.number),
                            ),
                            (
                                "pass_cols",
                                Pipeline(
                                    steps=[
                                        (
                                            "_pass_cols_",
                                            "passthrough",
                                        ),
                                        (
                                            "scaler",
                                            # RobustScaler(),
                                            self.scaler_pick
                                        ),
                                    ]
                                ),
                                make_column_selector(dtype_include=np.number),
                            ),
                        ],
                        remainder="passthrough",
                        n_jobs=-1,
                        verbose=True,
                        verbose_feature_names_out=False
                    ),
                ),
            ]
        )








    def categorical_pipeline(self):
        return Pipeline(
            steps=[
                (
                    "Preprocessing_Categorical",
                    ColumnTransformer(
                        transformers=[
                            (
                                "categorical",
                                Pipeline(
                                    steps=[
                                        (
                                        "Categorical",
                                            TypeInferenceTransformerNominal(
                                                datetime_columns=self.datetime_columns,
                                                name_transformer="Inference Categorical",
                                            ),
                                        ),
                                        (
                                            "C",
                                            SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                                        ),
                                        (
                                            "BinaryEnc",
                                            BinaryEncoder(handle_unknown="indicator"),
                                        ),
                                    ]
                                ),
                                self.categorical_columns
                                # make_column_selector(dtype_include=np.object_),
                            ),
                        ],
                        remainder="drop",
                        n_jobs=-1,
                        verbose=True,
                    ),
                )
            ]
        )

    def timeseries_pipeline(self):
        timeseries_preprocessor = Pipeline(
            steps=[
                (
                    "Preprocessing_Timeseries",
                    ColumnTransformer(
                        transformers=[
                            (
                                "timeseries",
                                Pipeline(
                                    steps=[
                                        ("T", TimeSeriesImputer(impute_method="ffill")),
                                        # ("num_time_dates", TimeTransformer())
                                        ("num_time_dates", DateEncoder()),
                                        # ("robust_scaler", RobustScaler()),
                                        # ("BinaryEnc",BinaryEncoder(handle_unknown="indicator")),
                                    ]
                                ),
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
                        n_jobs=-1,
                        verbose=True,
                        verbose_feature_names_out=False
                    ),
                )
            ]
        )

        return timeseries_preprocessor

    def pattern_extraction(
        self
    ):
        if self.pattern_recognition_columns is None:
            return  Pipeline(
                steps=[
                    (
                        "PatternRecognition Deactivated",
                        ColumnTransformer(
                            transformers=[
                                (
                                    "drop_columns",
                                    Pipeline(
                                        steps=[
                                            (
                                                "DropColumns", XPatternDropper(),
                                            ),
                                        ]
                                    ),
                                    make_column_selector(dtype_include=None),
                                ),
                            ],
                            remainder="drop",
                            n_jobs=-1,
                            verbose=True,
                        ),
                    )
                ]
            )
        
        else:
            return Pipeline(
                steps=[
                    (
                        "X_pattern",
                        TypeInferenceTransformerPattern(
                            include_columns=self.pattern_recognition_columns,
                            datetime_columns=self.datetime_columns,
                            name_transformer="Schema PatternExtraction",
                        ),
                    ),
                    (
                        "pattern_processing",
                        ColumnTransformer(
                            transformers=[
                                (
                                    "pattern",
                                    Pipeline(
                                        steps=[
                                            (
                                                "impute_pattern",
                                                SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                                            ),
                                            (
                                                "pattern_extraction",
                                                BinaryPatternTransformer(),
                                            ),
                                            (
                                                "BinaryEnc",
                                                BinaryEncoder(
                                                    handle_unknown="indicator"
                                                ),
                                            ),
                                        ]
                                    ),
                                    make_column_selector(dtype_include=np.object_),
                                )
                            ],
                            remainder="drop",
                        ),
                    ),
                ]
            )



    def nominal_pipeline(
        self):
        return Pipeline(
            steps=[
                (
                    "nominal_preprocessing",
                    ColumnTransformer(
                        transformers=[
                            (
                                "nominal",
                                Pipeline(
                                    steps=[
                                        (
                                            "Nominal Caster", 
                                            TypeInferenceTransformerNominal(
                                                datetime_columns=self.datetime_columns,
                                                exclude_columns=self.exclude_columns,
                                                nominal_columns=self.nominal_columns
                                            )
                                        ),
                                        (
                                            "impute_nominal",
                                            SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                                        ),                                        
                                        (
                                            "BinaryEnc",
                                            BinaryEncoder(handle_unknown="indicator"),
                                        ),
                                    ]
                                ),
                                self.nominal_columns
                            )
                        ],
                        remainder="drop",  
                        n_jobs=-1,
                        verbose=True,
                        # verbose_feature_names_out=False 
                    ),
                ),
            ]
        )

    def ordinal_pipeline(
        self    ):
        if len(self.ordinal_columns) == 0:
            raise Exception("Ordinal columns are empty...")
        
        return Pipeline(
            steps=[
                (
                    "ordinal_preprocessing",
                    ColumnTransformer(
                        transformers=[
                            (
                                "ordinal",
                                Pipeline(
                                    steps=[
                                        (
                                            "Ordinal Caster", 
                                            TypeInferenceTransformerOrdinal(
                                                datetime_columns=self.datetime_columns,
                                                exclude_columns = self.exclude_columns,
                                                ordinal_columns=self.ordinal_columns
                                            )
                                        ),
                                        (
                                            "impute_ordinal",
                                            SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                                        ),
                                        (
                                            "OrdinalEnc",
                                            OrdinalEncoder(
                                                handle_unknown="use_encoded_value",
                                                unknown_value=-1,
                                            ),
                                        ),
                                    ]
                                ),
                                # make_column_selector(dtype_include=None)
                                self.ordinal_columns
                            )
                        ],
                        remainder="drop",
                    ),
                ),
                # (
                #     "SpearmanCorrelationCheck",
                #     SpearmanCorrelationCheck(),
                # ),
            ]
        )
    
class XPatternDropper():
    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        return df[[]]
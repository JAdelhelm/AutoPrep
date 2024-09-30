
#%%
from numpy import ndarray
from AutoPrep.auto_decorators import PipelineDecorator
from AutoPrep.pipeline_configuration import PipelinesConfiguration
from sklearn.pipeline import FeatureUnion
import pandas as pd
from typing import Any, ClassVar, Iterable, TypeVar
from sklearn import set_config
set_config(transform_output="pandas")

class AutoPrep():
    def __init__(self, 
                 remove_columns_no_variance=False,
                 n_jobs = -1,
                 datetime_columns=None,
                 numerical_columns=None):

        self.remove_columns_no_variance = remove_columns_no_variance
        self.n_jobs = n_jobs
        self.datetime_columns = datetime_columns
        self.numerical_columns = numerical_columns
        
        self.auto_pipeline = None
        self.columns_with_no_variance = [] # save columns with no variance for further preprocessing of test data


        self.X_fit = None
        self.X_transformed = None


    def create_pipeline(self):
        """
        Creates preprocessing pipeline structure.
        """
        pipelines = [
            ("Numerical", PipelinesConfiguration.numeric_pipeline()),
            ("Categorical", PipelinesConfiguration.categorical_pipeline()),
            ("Timeseries", PipelinesConfiguration.timeseries_pipeline()),
            ("MissingIndicator", PipelinesConfiguration.nan_marker_pipeline())
        ]

        return FeatureUnion(
            transformer_list = pipelines,
            n_jobs = self.n_jobs
        )


    @PipelineDecorator.check_no_variance
    @PipelineDecorator.preprocess_data_for_pipeline
    def fit(self, X: ndarray | pd.DataFrame | Any, y= None | ndarray | pd.DataFrame | Any, **fit_params):
        self.X_fit = X

        self.auto_pipeline = self.create_pipeline()
        self.auto_pipeline_fitted = self.auto_pipeline.fit(X=self.X_fit)


    @PipelineDecorator.check_no_variance
    def transform(self, X: ndarray | pd.DataFrame | Any, y= None | ndarray | pd.DataFrame | Any, **fit_params):
        self.X_transformed = X
        if list(self.X_fit.columns) != list(X.columns):
            raise Exception(f"Column names must be identical!\n\n{self.X_fit.columns}\n{X.columns}")
        
        return self.auto_pipeline_fitted.transform(X=self.X_transformed)


    def fit_transform(self, X: ndarray | pd.DataFrame | Any, y: None | ndarray | pd.DataFrame | Any = None, **fit_params) -> ndarray:
        self.auto_pipeline = self.create_pipeline()

        return self.auto_pipeline.fit(X=X).transform(X=X)

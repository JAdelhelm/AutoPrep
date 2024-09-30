import os, sys, functools
import pandas as pd
from pandas.api.types import is_numeric_dtype
from datetime import datetime
from AutoPrep.pipeline_configuration import PipelinesConfiguration


class PipelineDecorator():

    @staticmethod
    def check_no_variance(func):
        """Wrapper to check if some columns have no variance."""
        @functools.wraps(func)
        def inner(*args, **kwargs):
            X_copy = kwargs.get('X')
            instance = args[0]
            
            for col in X_copy.columns:
                if True is is_numeric_dtype(X_copy[col]) and X_copy[col].std() <= 0.5:
                    instance.columns_with_no_variance.append(col)
                if False is is_numeric_dtype(X_copy[col]):
                    check_col  = pd.Categorical(X_copy[col]).codes
                    if check_col.std() <= 0.5:
                        instance.columns_with_no_variance.append(col)
           

            if instance.remove_columns_no_variance is True and len(instance.columns_with_no_variance) > 0:
                print(f"No variance detected in columns: {instance.columns_with_no_variance}")
                X_copy = X_copy.drop(columns=instance.columns_with_no_variance)

            # Returns X with removed variance columns
            kwargs['X'] = X_copy
            return func(*args, **kwargs)
        return inner

    
    @staticmethod
    def preprocess_data_for_pipeline(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            X = kwargs.get('X')
            instance = args[0]

            df_transformed = PipelinesConfiguration.pre_pipeline(
                datetime_columns=instance.datetime_columns,
                numerical_columns=instance.numerical_columns).fit_transform(X=X)
            instance.df_preprocessed_for_pipeline = df_transformed

            # func_preprocess = func(*args, **kwargs)

            return func(*args, **kwargs)
        return inner
    

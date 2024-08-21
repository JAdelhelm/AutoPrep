from sklearn.pipeline import FeatureUnion
import numpy as np
# from graphviz import Digraph


from AutoPrep.pipeline_configuration import PipelinesConfiguration


class PipelineControl(PipelinesConfiguration):
    """
    The PipelineControl class inherits all pipeline configurations
    and methods from the PipelinesConfiguration class.

    Parameters
    ----------
    datetime_columns : list, optional
        List of column names representing time data that should 
        be converted to timestamp data types. Default is None.

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
        Returns the standard pipeline configuration with profiling,
        datatypes, and preprocessing steps.

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
        scaler_option_num: str = "standard",
        deactivate_missing_indicator: bool = False
                 ) -> None:
        self._all_columns = (nominal_columns+ordinal_columns+ 
                        numerical_columns+datetime_columns)
        super().__init__(
            datetime_columns=datetime_columns,
            nominal_columns=nominal_columns,
            ordinal_columns=ordinal_columns,
            numerical_columns=numerical_columns,
            pattern_recognition_columns=pattern_recognition_columns,
            exclude_columns=exclude_columns,
            n_jobs=n_jobs,
            scaler_option_num=scaler_option_num,
            deactivate_missing_indicator=deactivate_missing_indicator,
            all_columns = self._all_columns
        )
    
    def manage_numerical_columns(self, df):
        """
        Manages the list of numerical columns in the given DataFrame.

        If `self.numerical_columns` is not set, this method identifies numerical columns in `df`
        and excludes any columns explicitly specified as nominal or ordinal.
        """
        if self.numerical_columns is None:
            # Select numerical columns from the DataFrame
            self.numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()

            # Exclude columns specified in nominal_columns, if it exists
            if hasattr(self, 'nominal_columns'):
                self.numerical_columns = [col for col in self.numerical_columns if col not in self.nominal_columns]

            # Exclude columns specified in ordinal_columns, if it exists
            if hasattr(self, 'ordinal_columns'):
                self.numerical_columns = [col for col in self.numerical_columns if col not in self.ordinal_columns]

    def pipeline_control(self):
        """
        Configures and returns a preprocessing pipeline based 
        on the presence of nominal and ordinal columns.

        Constructs a pipeline that applies numerical and time series transformations, with optional 
        handling for nominal and ordinal columns. If neither nominal nor ordinal columns are
        specified, a standard pipeline is returned. Columns not specified as nominal or 
        ordinal will be passed through a BinaryEncoder in the transformation process.

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
        if len(self.nominal_columns) == 0 and len(self.ordinal_columns) > 0: 
            return FeatureUnion(
                    transformer_list=[
                        ("Standard", self.standard_pipeline),
                        ("Ordinal", super().ordinal_pipeline()),
                        ("NaN", super().nan_marker_pipeline()),
                        ("PatternExtraction", super().pattern_extraction())
                    ],
                    n_jobs=self.n_jobs
                )
        if len(self.nominal_columns) > 0 and len(self.ordinal_columns) > 0: 
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
        Identifies categorical columns in the given DataFrame that 
        have not been explicitly specified.

        This method scans the provided DataFrame `df` to determine 
        which columns contain categorical data types that were not
        predefined or specified by the user. These identified categorical 
        columns will be processed separately in an additional transformation step.
        """
        self.categorical_columns = list(df.select_dtypes(include=[object]).columns)

        if hasattr(self, 'nominal_columns'):
            self.categorical_columns = [col for col in self.categorical_columns if col not in self.nominal_columns]

        if hasattr(self, 'ordinal_columns'):
            self.categorical_columns = [col for col in self.categorical_columns if col not in self.ordinal_columns]

        if self.categorical_columns:
            self.standard_pipeline = FeatureUnion(
                transformer_list=[
                    ("Standard", self.standard_pipeline),
                    ("categorical", super().categorical_pipeline())
                ],
                n_jobs=self.n_jobs
            )

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
        # Check if inputs are lists
        if self.datetime_columns is not None and not isinstance(self.datetime_columns, list):
            raise TypeError("datetime_columns must be a list")
        if self.nominal_columns is not None and not isinstance(self.nominal_columns, list):
            raise TypeError("nominal_columns must be a list")
        if self.ordinal_columns is not None and not isinstance(self.ordinal_columns, list):
            raise TypeError("ordinal_columns must be a list")
        if self.numerical_columns is not None and not isinstance(self.numerical_columns, list):
            raise TypeError("numerical_columns must be a list")
        if self.pattern_recognition_columns is not None and not isinstance(self.pattern_recognition_columns, list):
            raise TypeError("pattern_recognition_columns must be a list")
        if self.exclude_columns is not None and not isinstance(self.exclude_columns, list):
            raise TypeError("exclude_columns must be a list")
            
        for col in self._all_columns:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in the dataframe.")

        if len(self._all_columns) != len(set(self._all_columns)):
            duplicate_columns = [col for col in set(self._all_columns) if self._all_columns.count(col) > 1]
            raise ValueError(f"Duplicate columns found: {duplicate_columns}")
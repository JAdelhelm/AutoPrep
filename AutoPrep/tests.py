import pandas as pd
import numpy as np
import pytest
from autoprep import AutoPrep
import pdb

# Create some dummy data
X_train = pd.DataFrame({
    'ID': [1, 2, 3, 4],
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Rank': ['A', 'A', 'A', 'A'],
    'Age': [25, 30, 35, np.nan],
    'Salary': [50000.00, 60000.50, 75000.75, 80000],
    'Hire Date': pd.to_datetime(['2020-01-15', '2019-05-22', '2018-08-30', '2021-04-12']),
    'Is Manager': [False, True, False, ""]
})

X_test = pd.DataFrame({
    'ID': [1, 2, 3, 4],
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Rank': ['A', 'B', 'C', 'D'],
    'Age': [25, 30, 35, np.nan],
    'Salary': [50000.00, 60000.50, 75000.75, 8000000],  # Outlier in Salary
    'Hire Date': pd.to_datetime(['2020-01-15', '2019-05-22', '2018-08-30', '2021-04-12']),
    'Is Manager': [False, True, False, ""]
})



import unittest
class TestPipelineConsistency(unittest.TestCase):

    def test_inconsistent_column_1(self):
        """Open Issue"""
        df_inconsistent = pd.DataFrame({"ID": [1, 2, 3, "42"]})
        pipeline = AutoPrep()
        pipeline.fit_transform(df_inconsistent)

    def test_inconsistent_column_2(self):
        """Open Issue"""
        df_inconsistent = pd.DataFrame({"Name": ['Alice', 'Bob', 'Charlie', 42]})
        pipeline = AutoPrep()
        pipeline.fit_transform(df_inconsistent)

    def test_remove_variance_train_test(self):
        """
        Remove columns in X_train and X_test, if no variance in X_test,
        if remove_columns_no_variance = True
        """
        pipeline = AutoPrep(remove_columns_no_variance=True)
        pipeline.fit(X=X_train, y=None)
        pipeline.transform(X=X_test, y=None)
        # pdb.set_trace()

        self.assertEqual(list(pipeline.X_fit.columns), list(pipeline.X_transformed.columns))


    def test_different_dtypes(self):
        df_numerical = pd.DataFrame({"ID": [1, 2, 3, 4]})
        pipeline = AutoPrep()
        transformed_numerical = pipeline.fit_transform(df_numerical)
        self.assertEqual(transformed_numerical.shape[0], df_numerical.shape[0], "Row count mismatch for numerical DataFrame.")
        

        df_categorical = pd.DataFrame({'Name': ['Alice', 'Bob', 'Charlie', 'David']})
        transformed_categorical = pipeline.fit_transform(df_categorical)

        self.assertGreaterEqual(transformed_categorical.shape[1], 1, "Categorical DataFrame transformation did not produce new features.")
        self.assertEqual(transformed_categorical.shape[0], df_categorical.shape[0], "Row count mismatch for categorical DataFrame.")

        df_timeseries = pd.DataFrame({'Hire Date': pd.to_datetime(['2020-01-15', '2019-05-22', '2018-08-30', '2021-04-12'])})
        transformed_timeseries = pipeline.fit_transform(df_timeseries)

        self.assertGreaterEqual(transformed_timeseries.shape[1], 1, "Datetime DataFrame transformation did not produce new features.")
        self.assertEqual(transformed_timeseries.shape[0], df_timeseries.shape[0], "Row count mismatch for datetime DataFrame.")

        df_nan_values = pd.DataFrame({"COL": ['A', np.nan, 'C', np.nan]})
        transformed_nan_values = pipeline.fit_transform(df_nan_values)
        self.assertFalse(transformed_nan_values.isna().any().any(), "NaN values were not handled correctly.")


        df_combined = pd.DataFrame({
            "ID": [1, 2, 3, 4],
            "Name": ['Alice', 'Bob', 'Charlie', 'David'],
            "Hire Date": pd.to_datetime(['2020-01-15', '2019-05-22', '2018-08-30', '2021-04-12']),
            "COL": ['A', np.nan, 'C', np.nan]
        })
        transformed_combined = pipeline.fit_transform(df_combined)
        self.assertEqual(transformed_combined.shape[0], df_combined.shape[0], "Row count mismatch for combined DataFrame.")
        self.assertGreaterEqual(transformed_combined.shape[1], df_combined.shape[1], 
                                "Combined DataFrame transformation did not produce expected features.")


    def test_column_consistency_after_transformation(self):
        """
        Test that the output columns after transformation still retain the original names 
        in some form even if the transformation alters them.
        """
        pipeline = AutoPrep(remove_columns_no_variance=False)
        pipeline.fit(X=X_train, y=None)
        transformed_test = pipeline.transform(X=X_test, y=None)

        # Verify that all original columns are present, possibly renamed
        for col in X_train.columns:
            self.assertTrue(any(col in transformed_col for transformed_col in transformed_test.columns),
                            f"Original column {col} not found in transformed columns.")
    
    def test_transform_column_count(self):
        """
        Test that the number of columns in the transformed data is greater than or equal 
        to the original data (indicating possible feature expansion).
        """
        pipeline = AutoPrep(remove_columns_no_variance=True)
        pipeline.fit(X=X_train, y=None)
        transformed_test = pipeline.transform(X=X_test, y=None)

        # Check that the number of columns in the transformed data is >= the original data
        self.assertGreaterEqual(len(transformed_test.columns), len(X_train.columns),
                                "The number of columns in the transformed data is less than expected.")

    def test_data_integrity_after_transform(self):
        """
        Test that the transformed data retains values from the original dataset, ensuring data integrity.
        """
        pipeline = AutoPrep(remove_columns_no_variance=True)
        pipeline.fit(X=X_train, y=None)
        transformed_test = pipeline.transform(X=X_test, y=None)

        # Check that the values from the original 'Age' column are retained in the transformed data
        self.assertTrue(all(X_test['Age'].dropna().isin(transformed_test['Age'])),
                        "Values in 'Age' column do not match between original and transformed data.")
    
    def test_pipeline_fitting(self):
        """
        Test that the pipeline fitting does not raise any errors and correctly stores
        the fitted pipeline for future transformations.
        """
        pipeline = AutoPrep(remove_columns_no_variance=True)
        try:
            pipeline.fit(X=X_train, y=None)
            self.assertIsNotNone(pipeline.auto_pipeline_fitted, "Pipeline did not fit properly.")
        except Exception as e:
            self.fail(f"Pipeline fitting raised an unexpected exception: {e}")


    def test_transform_with_nan_values(self):
        """
        Test that the pipeline can handle NaN values in the data during transformation.
        """
        pipeline = AutoPrep(remove_columns_no_variance=True)
        pipeline.fit(X=X_train, y=None)
        transformed_test = pipeline.transform(X=X_test, y=None)

        self.assertFalse(transformed_test.isna().any().any(),
                         "NaN values are still present in the transformed data.")

    def test_transform_with_outliers(self):
        """
        Test how the pipeline handles outliers, such as an unusually high salary in the test data.
        """
        pipeline = AutoPrep(remove_columns_no_variance=True)
        pipeline.fit(X=X_train, y=None)
        transformed_test = pipeline.transform(X=X_test, y=None)

        self.assertIn(8000000, transformed_test['Salary'].values,
                      "Outlier in 'Salary' column has been incorrectly removed or altered.")

    def test_pipeline_with_datetime_columns(self):
        """
        Test that the pipeline correctly processes datetime columns such as 'Hire Date'.
        """
        pipeline = AutoPrep(remove_columns_no_variance=True, datetime_columns=['Hire Date'])
        pipeline.fit(X=X_train, y=None)
        transformed_test = pipeline.transform(X=X_test, y=None)

        self.assertTrue(any('Hire Date' in col for col in transformed_test.columns),
                        "'Hire Date' column is missing after transformation.")

    def test_pipeline_with_categorical_columns(self):
        """
        Test that the pipeline correctly processes categorical columns such as 'Rank'.
        """
        pipeline = AutoPrep(remove_columns_no_variance=False)
        pipeline.fit(X=X_train, y=None)
        transformed_test = pipeline.transform(X=X_test, y=None)

        self.assertTrue(any('Rank' in col for col in transformed_test.columns),
                        "'Rank' column is missing after transformation or was not properly processed as categorical.")


    def test_pipeline_with_dropped_columns(self):
        """
        Test that columns are properly dropped when specified as having no variance or for other reasons.
        """
        pipeline = AutoPrep(remove_columns_no_variance=True)
        pipeline.fit(X=X_train, y=None)
        transformed_test = pipeline.transform(X=X_test, y=None)

        self.assertNotIn('Rank', transformed_test.columns,
                         "'Rank' column was not dropped as expected due to lack of variance in training data.")

    def test_AutoPrep_empty_df(self):
        pipeline = AutoPrep()
        empty_df = pd.DataFrame()
        X_output = pipeline.fit_transform(empty_df)

        assert isinstance(X_output, pd.DataFrame)
        assert X_output.empty, "Output should be empty for an empty input DataFrame"

    def test_AutoPrep_single_row(self):
        pipeline = AutoPrep()
        df_single = pd.DataFrame({'Name': ['Alice'], 'Age': [25], 'Sex': ['female']})
        X_output = pipeline.fit_transform(df_single)

        assert isinstance(X_output, pd.DataFrame)
        assert len(X_output) == 1, "Output should contain one row"

    def test_AutoPrep_test_default_behavior(self):
        pipeline = AutoPrep()
        X_output = pipeline.fit_transform(X_train)
        # pdb.set_trace()
        
        assert isinstance(X_output, pd.DataFrame)

if __name__ == "__main__":
    # Running the tests
    pytest.main()

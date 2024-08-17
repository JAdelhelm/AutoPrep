#%%

############## dummy data #############
import pandas as pd
import numpy as np
data = {

    'ID': [1, 2, 3, 4],                 
    'Name': ['Alice', 'Bob', 'Charlie', 42],  
    'Rank': ['A','B','C','D'],
    'Age': [25, 30, 35, np.nan],                 
    'Salary': [50000.00, 60000.50, 75000.75, 80000.00], 
    'Hire Date': pd.to_datetime(['2020-01-15', '2019-05-22', '2018-08-30', '2021-04-12']), 
    'Is Manager': [False, True, False, ""]  
}
data = pd.DataFrame(data)
########################################

df_with_dates = pd.DataFrame({
    'DateColumn': ['2021-01-01', '2022-01-01', '2023-01-01'],
    'Value': [100, 200, 300]
})


from autoprep import AutoPrep

pipeline = AutoPrep(
    datetime_columns=["DateColumn"],
    # pattern_recognition_columns=["Name"]
)
X_output = pipeline.preprocess(df=df_with_dates)

# pipeline.get_profiling(X=data)
pipeline.visualize_pipeline_structure_html()
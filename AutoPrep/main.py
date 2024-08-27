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
import sys
sys.path.append("../")
sys.path.append("./")

from AutoPrep.autoprep import AutoPrep

pipeline = AutoPrep(
    nominal_columns=["ID", "Name", "Is Manager", "Age"],
    datetime_columns=["Hire Date"],
    pattern_recognition_columns=["Name"],
    scaler_option_num="standard",
    deactivate_missing_indicator=True
)
# Automated Preprocessing of data
# X_output = pipeline.preprocess(df=data)

# Automated Preprocessing + Anomalies in data with pyod library
# X_output = pipeline.find_anomalies(df=data)

# pipeline.get_profiling(X=data)
pipeline.visualize_pipeline_structure_html()
# %%

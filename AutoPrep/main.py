#%%

############## dummy data #############
import pandas as pd
data = {
    'ID': [1, 2, 3, 4],                 
    'Name': ['Alice', 'Bob', 'Charlie', 'R2D2'],  
    'Age': [25, 30, 35, 90],                 
    'Salary': [50000.00, 60000.50, 75000.75, 80000.00], 
    'Hire Date': pd.to_datetime(['2020-01-15', '2019-05-22', '2018-08-30', '2021-04-12']), 
    'Is Manager': [False, True, False, True]  
}
data = pd.DataFrame(data)
########################################


from autoprep import AutoPrep

pipeline = AutoPrep(
    nominal_columns=["ID", "Name", "Is Manager"],
    datetime_columns=["Hire Date"],
    pattern_recognition_columns=["Name"]

)
X_output = pipeline.preprocess(df=data)

# pipeline.get_profiling(X=data)
# pipeline.visualize_pipeline_structure_html()
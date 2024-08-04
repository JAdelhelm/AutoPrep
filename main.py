# %%
import numpy as np
import pandas as pd
from sklearn import set_config

set_config(transform_output="pandas")


from pipelines.control import AutoPrep

if __name__ == "__main__":
    df_data = pd.read_csv("./temperature_USA.csv")


    pipeline = AutoPrep()    
    X_output = pipeline.preprocess(df=df_data)




    


# %%



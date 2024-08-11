# AutoPrep -  Automated Preprocessing Pipeline with Univariate Anomaly Marking

This pipeline focuses on data preprocessing, standardization, and cleaning, with additional features to identify univariate anomalies.
<!-- <a href="https://html-preview.github.io/?url=https://github.com/JAdelhelm/Automated-Anomaly-Detection-Preprocessing-Pipeline/blob/main/visualization/Pipeline.html" target="_blank">Structure of Preprocessing Pipeline</a> -->

## Description
The Automated Preprocessing Pipeline with Univariate Anomaly Marking saves time by providing robust preprocessing, standardization, and anomaly detection, ensuring high data quality and integrity for analysis and machine learning applications ✅.

## Highlights ⭐


#### 📌 Implementation of univariate methods / *Detection of univariate anomalies*
Both methods (MOD Z-Value and Tukey Method) are resilient against outliers, ensuring that the position measurement will not be biased. They also support multivariate anomaly detection algorithms in identifying univariate anomalies.

#### 📌 BinaryEncoder instead of OneHotEncoder for nominal columns / *Big Data and Performance*
   Newest research shows similar results for encoding nominal columns with significantly fewer dimensions.
   - (John T. Hancock and Taghi M. Khoshgoftaar. "Survey on categorical data for neural networks." In: Journal of Big Data 7.1 (2020), pp. 1–41.), Tables 2, 4
   - (Diogo Seca and João Mendes-Moreira. "Benchmark of Encoders of Nominal Features for Regression." In: World Conference on Information Systems and Technologies. 2021, pp. 146–155.), P. 151

#### 📌 Transformation of time series data and standardization of data with RobustScaler / *Normalization for better prediction results*

#### 📌 Labeling of NaN values in an extra column instead of removing them / *No loss of information*


## Basic Usage
To utilize this pipeline, you need to import the necessary libraries and initialize the AutoPrep pipeline. Here is a basic example:

````python
############## dummy data ##############
import pandas as pd
data = {
    'ID': [1, 2, 3, 4],                 
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],  
    'Age': [25, 30, 35, 40],                 
    'Salary': [50000.00, 60000.50, 75000.75, 80000.00], 
    'Hire Date': pd.to_datetime(['2020-01-15', '2019-05-22', '2018-08-30', '2021-04-12']), 
    'Is Manager': [False, True, False, True]  
}
data = pd.DataFrame(data)
########################################


from AutoPrep import AutoPrep

from AutoPrep import AutoPrep

pipeline = AutoPrep(
    nominal_columns=["ID", "Name", "Is Manager"],
    datetime_columns=["Hire Date"]
)
X_output = pipeline.preprocess(df=data)
````

The resulting output dataframe can be accessed by using:

````python
X_output

> Output:
    col_1  col_2  ...   col_n
1   data   ...    ...   data   
2   data   ...    ...   data  
... ...    ...    ...   ...   
````





---
---
## Pipeline - Built-in Logic
<!-- ![Logic of Pipeline](./images/decision_rules.png) -->
![Logic of Pipeline](https://raw.githubusercontent.com/JAdelhelm/AutoPrep/main/images/decision_rules.png) 





## Abstract View (Code Structure)
<!-- ![Abstract view of the project](./images/project.png) -->
![Abstract view of the project](https://raw.githubusercontent.com/JAdelhelm/AutoPrep/main/images/project.png)




---

---

## Feel free to contribute 🙂

### Reference
- https://www.researchgate.net/publication/379640146_Detektion_von_Anomalien_in_der_Datenqualitatskontrolle_mittels_unuberwachter_Ansatze (German Thesis)

### Further Information

- I used sklearn's Pipeline and Transformer concept to create this preprocessing pipeline
    - Pipeline: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
    - Transformer: https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html

















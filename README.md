# AutoPrep -  Automated Preprocessing Pipeline with Univariate Anomaly Indicators
![PyPIv](https://img.shields.io/pypi/v/AutoPrep)
![PyPI status](https://img.shields.io/pypi/status/AutoPrep)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/AutoPrep) ![PyPI - License](https://img.shields.io/pypi/l/AutoPrep)
<!-- [![Downloads](https://static.pepy.tech/badge/AutoPrep)](https://pepy.tech/project/AutoPrep) -->



This pipeline focuses on data preprocessing, standardization, and cleaning, with additional features to identify univariate anomalies.
<a href="https://html-preview.github.io/?url=https://raw.githubusercontent.com/JAdelhelm/AutoPrep/main/visualization/PipelineStructure.html" target="_blank">Structure of Preprocessing Pipeline</a>


```python
pip install AutoPrep
```
#### Dependencies
- scikit-learn
- category_encoders
- bitstring
- ydata_profiling


## Basic Usage
To utilize this pipeline, you need to import the necessary libraries and initialize the AutoPrep pipeline. Here is a basic example:

````python
import pandas as pd
import numpy as np
import sys
sys.path.append("../")
sys.path.append("./")

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
````
````python
from AutoPrep import AutoPrep


pipeline = AutoPrep(
    nominal_columns=["ID", "Name", "Is Manager", "Age"],
    datetime_columns=["Hire Date"],
    pattern_recognition_columns=["Name"],
    scaler_option_num="standard",
    deactivate_missing_indicator=True
)
#### Automated Preprocessing of data
X_output_preprocessed = pipeline.fit_transform(df=data)

#### Automated Preprocessing + Anomalies in data with pyod library
X_output_anomalies = pipeline.find_anomalies(df=data)


#### Profiling of DataFrame / Visualization of pipeline structure
# pipeline.get_profiling(X=data)
# pipeline.visualize_pipeline_structure_html()
````

## Highlights ⭐


#### 📌 Implementation of univariate methods / *Detection of univariate anomalies*
Both methods (MOD Z-Value and Tukey Method) are resilient against outliers, ensuring that the position measurement will not be biased. They also support multivariate anomaly detection algorithms in identifying univariate anomalies.

#### 📌 BinaryEncoder instead of OneHotEncoder for nominal columns / *Big Data and Performance*
   Newest research shows similar results for encoding nominal columns with significantly fewer dimensions.
   - (John T. Hancock and Taghi M. Khoshgoftaar. "Survey on categorical data for neural networks." In: Journal of Big Data 7.1 (2020), pp. 1–41.), Tables 2, 4
   - (Diogo Seca and João Mendes-Moreira. "Benchmark of Encoders of Nominal Features for Regression." In: World Conference on Information Systems and Technologies. 2021, pp. 146–155.), P. 151

#### 📌 Transformation of time series data and standardization of data with RobustScaler / *Normalization for better prediction results*

#### 📌 Labeling of NaN values in an extra column instead of removing them / *No loss of information*



---





## Pipeline - Built-in Logic
<!-- ![Logic of Pipeline](./images/decision_rules.png) -->
![Logic of Pipeline](https://raw.githubusercontent.com/JAdelhelm/AutoPrep/main/AutoPrep/img/decision_rules.png) 





<!-- ## Abstract View (Code Structure) -->
<!-- ![Abstract view of the project](./images/project.png) -->
<!-- ![Abstract view of the project](https://raw.githubusercontent.com/JAdelhelm/AutoPrep/main/images/project.png) -->




---

---

## Feel free to contribute 🙂

### Reference
- https://www.researchgate.net/publication/379640146_Detektion_von_Anomalien_in_der_Datenqualitatskontrolle_mittels_unuberwachter_Ansatze (German Thesis)

### Further Information

- I used sklearn's Pipeline and Transformer concept to create this preprocessing pipeline
    - Pipeline: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
    - Transformer: https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html




















import pandas as pd 
from inference_controller import InferenceEngine

# load data
data = pd.read_csv("datasets/test_data.csv")
# filter rows with null values
data = data.dropna().reset_index(drop = True)

# build inference engine object
"""
Keyword arguments:
path_model -- write path of ML model
path_db_categories -- write path of json database of categorical features
Return: inference object
"""

inference = InferenceEngine(path_model = "models/lgbm_scc_perform_v10.pkl", 
                            path_db_categories = "datasets/db_features.json")

# check metadata
print(inference.feature_names)

# preprocessing
"""
Keyword arguments:
data -- put entire dataset, filtering missing values
Return: features (X) and outputs (Y)
"""
x, y = inference.prepare_df(data)

# to get prediction
"""sumary_line

Keyword arguments:
data -- put data frame corresponded by dataset after filter rows with null values
Return: vector of predictions
"""

y_pred = inference.predict(x)
print(y_pred)

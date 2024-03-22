import pandas as pd
import numpy as np
import pickle
from typing import List
from preprocess_controller import Preprocessor
from sklearn.metrics import mean_squared_error, r2_score

#       --- class for inference engine ---
class InferenceEngine:
    # initialize params
    # provide path of model and db_categories file
    def __init__(self, path_model, path_metadata):        
        self.model = self.load_model(path_model)
        self.preprocessor = Preprocessor(path_metadata)
        self.feature_names = self.preprocessor.dict_metadata["scc_metadata"]        

    # function to preprocess input data
    def prepare_df(self, data: pd.DataFrame) -> pd.DataFrame:
        # make preprocessing
        print("start preprocessing...")
        input, output = self.preprocessor.preprocessing_pipeline(data)

        return input, output

    # function to load model
    def load_model(self, filename):
        print("loading model...")
        with open(filename, "rb") as file:
            return pickle.load(file)

    # function to get feature names
    @property
    def model_keys(self) -> List[str]:
        return self.feature_names

    # function to predict output
    def predict(self, x) -> pd.Series:
        print("generation of predictions...")
        y_pred = self.model.predict(x)
        results = pd.Series(np.power(10, y_pred) - 1)
        return results

    # function to calculate performance metrics
    def evaluate(self, y, y_pred):
        y_pred = np.log10(1 + y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        return rmse, r2
    
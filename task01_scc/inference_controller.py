import pandas as pd
import numpy as np
import pickle
from typing import List
from preprocess_controller import Preprocessor

#       --- class for inference engine ---
class InferenceEngine:
    # initialize params
    # provide path of model and db_categories file
    def __init__(self, path_model, path_db_categories):        
        self.model = self.load_model(path_model)
        self.preprocessor = Preprocessor(path_db_categories)
        self.feature_names = self.preprocessor.db_categories["metadata"]

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
        return pd.Series(y_pred)
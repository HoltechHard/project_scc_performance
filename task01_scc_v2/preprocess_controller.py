import pandas as pd
import numpy as np
import json
import calendar
import time
from datetime import datetime

class Preprocessor:
    def __init__(self, path_metadata):
        self.path_metadata = path_metadata
        self.dict_metadata = self.load_dictionary()

    # load dictionary
    def load_dictionary(self):
        with open(self.path_metadata, "r") as file:
            return json.load(file)

    # reasign states
    def replace_cancelled_states(self, status):
        if "CANCELLED" in status:
            return "CANCELLED"
        return status
    
    # encode states
    def encoding_states(self, data, column):
        data[column] = data[column].apply(lambda x: self.replace_cancelled_states(x))
        return data

    # function to convert transcurred time to logaritmic scale of seconds
    def time_conversion(self, time_data):
        # extract time units
        if "-" in time_data:
            days, timer = time_data.split("-")
        else:
            days = 0
            timer = time_data 
        
        days = int(days)
        hours, min, sec = map(int, timer.split(":"))

        # calculate total seconds
        total_sec = days*24*60*60 + hours*60*60 + min*60 + sec
        
        return total_sec

    # function to encode transcurred time for entire column
    def encoding_time(self, data, name_col):
        data["Log"+name_col+"Sec"] = data[name_col].apply(self.time_conversion)
        data["Log"+name_col+"Sec"] = np.log10(1 + data["Log"+name_col+"Sec"])
        
        return data

    # function to calculate difference between 2 dates
    def difference_dates(self, date1, date2):
        dformat = "%Y-%m-%dT%H:%M:%S"
        date1 = datetime.strptime(date1, dformat)
        date2 = datetime.strptime(date2, dformat)
        tdiff = date2 - date1

        return tdiff.total_seconds()

    # function to calculate wait time between submission and start task
    def encode_twatting(self, data, col1, col2):
        data["LogTimeWaitting"] = data.apply(lambda row: self.difference_dates(row[col1], row[col2]), axis = 1)
        data["LogTimeWaitting"] = np.log10(1 + data["LogTimeWaitting"])
        
        return data

    def encoding_date(self, data, name_col):
        #categorical variants of values
        year_names = list(np.arange(2020, 2025, 1))
        year_variants = [f"{name_col}Year {i}" for i in year_names]
        month_names = list(calendar.month_name)
        month_variants = [f"{name_col}Month {j}" for j in month_names if j!='']
        dayw_names = list(calendar.day_name)
        dayw_variants = [f"{name_col}DayWeek {k}" for k in dayw_names]
        # temporal dataframe
        tmp_cols = year_variants + month_variants + dayw_variants
        tmp_frame = pd.DataFrame(np.zeros((len(data), len(tmp_cols))), columns = tmp_cols)
        tmp_frame.index = data.index
        # generate datetime
        dates_mod = pd.to_datetime(data[name_col])
        dates_mod.index = data.index
        
        for i, date_val in dates_mod.items():
            tmp_frame.loc[i, f"{name_col}Year {date_val.year}"] = 1
            tmp_frame.loc[i, f"{name_col}Month {calendar.month_name[date_val.month]}"] = 1
            tmp_frame.loc[i, f"{name_col}DayWeek {calendar.day_name[date_val.day_of_week]}"] = 1

        # concatenate data-frames
        data = pd.concat([data, tmp_frame], axis = 1, ignore_index = False)

        return data

    # one-hot encoder for categorical variables 
    def encoding_category(self, data, category):
        col_instances = [f"{category} {i}" for i in self.dict_metadata[category]]
        tmp_df = pd.DataFrame(np.zeros((len(data), len(col_instances))), columns = col_instances)
        tmp_df.index = data.index

        for i, cat_val in data[category].items():
            tmp_df.loc[i, f"{category} {cat_val}"] = 1
        # concatenate data-frames
        data = pd.concat([data, tmp_df], axis = 1, ignore_index = False)

        return data

    def preprocessing_pipeline(self, data):
        # start time
        start_t = time.time()     
        # encode states
        data = self.encoding_states(data, "State")
        # encode the Time limit
        data = self.encoding_time(data, "Timelimit")
        # encode the Time elapsed
        data = self.encoding_time(data, "Elapsed")
        # encode the Time of waitting
        data = self.encode_twatting(data, "Submit", "Start")
        # encode the Submit dates
        data = self.encoding_date(data, "Submit")
        # encode the Start dates
        data = self.encoding_date(data, "Start")
        
        # one-hot encoding for categorical variables    
        list_cats = [key for key in self.dict_metadata.keys() if key!="scc_metadata"]

        for category in list_cats:
            data = self.encoding_category(data, category)
        
        # drop the original preprocessed columns
        data = data.drop(columns = [c for c in data.columns.tolist() if c not in self.dict_metadata["scc_metadata"]])
    
        # split x and y
        x = data.drop(columns = ["LogElapsedSec"])
        y = data["LogElapsedSec"]

        # count time
        end_t = time.time()
        print(f"Time of preprocessing {end_t - start_t}")

        return x, y

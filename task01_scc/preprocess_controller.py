import pandas as pd
import numpy as np
import json
import calendar
from datetime import datetime

class Preprocessor:
    def __init__(self, path_db_categories):
        self.path_db_categories = path_db_categories
        self.db_categories = self.load_dictionary()

    # function to encoding date
    def encoding_date(self, data, name_col):
        #categorical variants of values
        year_names = list(np.arange(2020, 2025, 1))
        year_variants = [f"{name_col}Year{i}" for i in year_names]
        month_names = list(calendar.month_name)
        month_variants = [f"{name_col}Month{j}" for j in month_names if j!='']
        dayw_names = list(calendar.day_name)
        dayw_variants = [f"{name_col}DayofWeek{k}" for k in dayw_names]
        # temporal dataframe
        tmp_cols = year_variants + month_variants + dayw_variants
        tmp_frame = pd.DataFrame(np.zeros((len(data), len(tmp_cols))), columns = tmp_cols)        
        # generate datetime
        dates_mod = pd.to_datetime(data[name_col])
        
        for i in range(len(data)):
            tmp_frame.loc[i, f"{name_col}Year{dates_mod[i].year}"] = 1
            tmp_frame.loc[i, f"{name_col}Month{calendar.month_name[dates_mod[i].month]}"] = 1
            tmp_frame.loc[i, f"{name_col}DayofWeek{calendar.day_name[dates_mod[i].day_of_week]}"] = 1

        # concatenate data-frames
        data = pd.concat([data, tmp_frame], axis = 1, ignore_index = False)

        return data

    def replace_cancelled_states(self, status):
        if "CANCELLED" in status:
            return "CANCELLED"
        return status
    
    # function to encode states
    def encoding_states(self, data, column):
        data[column] = data[column].apply(lambda x: self.replace_cancelled_states(x))
        return data

    # function to convert transcurred time to seconds
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
        data[name_col+"Tsec"] = data[name_col].apply(self.time_conversion)

        return data

    # function to calculate difference between 2 dates
    def difference_dates(self, date1, date2):
        dformat = "%Y-%m-%dT%H:%M:%S"
        date1 = datetime.strptime(date1, dformat)
        date2 = datetime.strptime(date2, dformat)
        tdiff = date2 - date2

        return tdiff.total_seconds()
    
    # function to calculate wait time between submission and start task
    def encode_twatting(self, data, col1, col2):
        data["WaittingTsec"] = data.apply(lambda row: self.difference_dates(row[col1], row[col2]), axis = 1)

        return data

    # load database of categorical levels
    def load_dictionary(self):
        with open(self.path_db_categories, "r") as file:
            return json.load(file)

    # one-hot encoder for categorical variables 
    def encoding_category(self, data, category):
        col_instances = [f"{category}{i}" for i in self.db_categories[category]]        
        tmp_df = pd.DataFrame(np.zeros((len(data), len(col_instances))), columns = col_instances)
        for i in range(len(data)):            
            tmp_df.loc[i, f"{category}{data[category][i]}"] = 1
        # concatenate data-frames
        data = pd.concat([data, tmp_df], axis = 1, ignore_index = False)

        return data
    
    # final pipeline for data preprocessing
    def preprocessing_pipeline(self, data):                    
        # drop Id, UID, GID
        data = data.drop(columns = ["Id", "UID", "GID", "JobName", "ExitCode"])
        # encode the Submit dates
        data = self.encoding_date(data, "Submit")
        # encode the Start dates
        data = self.encoding_date(data, "Start")
        # encode states
        data = self.encoding_states(data, "State")
        # encode the Time limit
        data = self.encoding_time(data, "Timelimit")
        # encode the Time elapsed
        data = self.encoding_time(data, "Elapsed")
        # encode the Time of waitting
        data = self.encode_twatting(data, "Submit", "Start")
        # drop some preprocessed columns
        data = data.drop(columns = ["Submit", "Start", "Timelimit", "Elapsed"])
        
        # one-hot encoding for categorical variables
        list_cats = [key for key in self.db_categories.keys() if key!="metadata"]

        for category in list_cats:
            data =self.encoding_category(data, category)
        # drop the original preprocessed columns
        data = data.drop(columns = list_cats)        

        # split x and y
        x = data.drop(columns = ["ElapsedTsec"])
        y = data["ElapsedTsec"]

        return x, y
    
import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split

# def Create_Data(Path):
#     data = pd.read_csv(Path)
#     Output = data["Type"]
#     Data = data.drop('Type', axis=1)
#     return Data.values, Output.values

# Path = 'processed_datasets/24_riboswitches_final.csv'
# Data, Output = Create_Data(Path)

# Data_train, Data_test, Output_train, Output_test = train_test_split(Data, Output, test_size=0.1, stratify=Output)

# Data_test = pd.DataFrame(Data_test)
# Output_test = pd.DataFrame(Output_test)
# print (Data_test)

def Create_Data(Path):
    data = pd.read_csv(Path)
    data.drop_duplicates(keep='first', inplace=True)
    output = data["Type"]
    return data, output

Path = 'processed_datasets/final_32classes.csv'
Data, Output = Create_Data(Path)


print (Data)
# print (Output)

Data_train, Data_test, Output_train, Output_test = train_test_split(Data, Output, test_size=0.3, stratify=Output)
print (Data_train['Type'].value_counts())
print (Data_test['Type'].value_counts())

file_name = 'processed_datasets/final_32train.csv'
Data_train.to_csv(file_name, sep=',', encoding='utf-8', index=False)
file_name = 'processed_datasets/final_32test.csv'
Data_test.to_csv(file_name, sep=',', encoding='utf-8', index=False)

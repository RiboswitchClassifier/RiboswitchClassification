import pandas as pd
import numpy as np

tsv_file='/Users/ramitb/Downloads/RF01055.tsv'
csv_table=pd.read_table(tsv_file,sep='\t')
csv_table.to_csv('/Users/ramitb/Downloads/RF01055.csv',index=False)

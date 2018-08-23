import csv

# inputfile = csv.reader(open('RF00050','r'))

# for row in inputfile:
#     print row[1]

import pandas as pd
f=pd.read_csv("RF01057.csv")
keep_col = ['sequence']
new_f = f[keep_col]
new_f["target"] = 15
new_f.to_csv("cleanData/RF01057.csv", index=False)
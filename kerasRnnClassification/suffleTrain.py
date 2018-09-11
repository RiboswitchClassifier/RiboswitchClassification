import pandas as pd

df = pd.read_csv('cleanData/16riboswitches.csv', header=None)
ds = df.sample(frac=1)
ds.to_csv('cleanData/shuffled16riboswitches.csv')

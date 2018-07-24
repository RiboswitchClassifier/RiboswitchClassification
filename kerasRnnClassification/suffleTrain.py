import pandas as pd

df = pd.read_csv('splice.csv', header=None)
ds = df.sample(frac=1)
ds.to_csv('splice_new1.csv')

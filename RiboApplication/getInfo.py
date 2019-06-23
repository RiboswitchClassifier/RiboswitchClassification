import pandas as pd

riboswitch_names = [
    'RF00050',
    'RF00059',
    'RF00162',
    'RF00167',
    'RF00168',
    'RF00174',
    'RF00234',
    'RF00380',
    'RF00504',
    'RF00521',
    'RF00522',
    'RF00634',
    'RF01051',
    'RF01054',
    'RF01055',
    'RF01057',
    'RF01725',
    'RF01726',
    'RF01727',
    'RF01734',
    'RF01739',
    'RF01763',
    'RF01767',
    'RF02683'
]

a = []
b = []
for name in riboswitch_names:
    df = pd.read_csv("original_datasets/24_riboswitches_csv/" + name + ".csv", sep='\t')
    # print ("Name of Riboswitch : " + name)
    count_row = df.shape[0]
    a.append(count_row)
    # print ("Number of Sequences :" + str(count_row))
    df['Sequence_Length'] = df['Sequence'].apply(len)
    b.append(df['Sequence_Length'].mean())
    # print("Average Sequence Length : " + str(df['Sequence_Length'].mean()))

print (sum(a))
print (sum(b) / float(len(b)))
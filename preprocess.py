import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler, label_binarize
import subprocess
import os
import re
from Bio import SeqIO
import functools

def processingscript():
    subprocess.call(['mkdir','original_datasets/31_riboswitches_new_csv'])
    def makecsv(file):
        input_file = open(file, 'r')
        csvfiles = 'original_datasets/31_riboswitches_new_csv'
        file=file.split("/")
        filename=file[2]
        filename=filename.split(".")[0]
        output_file = open("{}/{}.csv".format(csvfiles,filename),'w')

        output_file.write('Gene,A,T,G,C,AA,AC,AG,AT,CA,CC,CG,CT,GA,GC,GG,GT,TA,TC,TG,TT\n')
        for cur_record in SeqIO.parse(input_file, "fasta") :

            gene_name = cur_record.name


            A_count   = cur_record.seq.count('A')
            T_count   = cur_record.seq.count('T')
            G_count   = cur_record.seq.count('G')
            C_count   = cur_record.seq.count('C')
            AA_count  = cur_record.seq.count('AA')
            AC_count  = cur_record.seq.count('AC')
            AG_count  = cur_record.seq.count('AG')
            AT_count  = cur_record.seq.count('AT')
            CA_count  = cur_record.seq.count('CA')
            CC_count  = cur_record.seq.count('CC')
            CG_count  = cur_record.seq.count('CG')
            CT_count  = cur_record.seq.count('CT')
            GA_count  = cur_record.seq.count('GA')
            GC_count  = cur_record.seq.count('GC')
            GG_count  = cur_record.seq.count('GG')
            GT_count  = cur_record.seq.count('GT')
            TA_count  = cur_record.seq.count('TA')
            TC_count  = cur_record.seq.count('TC')
            TG_count  = cur_record.seq.count('TG')
            TT_count  = cur_record.seq.count('TT')
            monocount = A_count+ T_count+ G_count+ C_count
            dicount = AA_count+ AC_count+ AG_count+ AT_count+ CA_count+ CC_count+ CG_count+ CT_count+ GA_count+ GC_count+ GG_count+GT_count+TA_count+TC_count+TG_count+TT_count
            output_file.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(gene_name,cur_record.seq, A_count/monocount, T_count/monocount, G_count/monocount, C_count/monocount, AA_count/dicount, AC_count/dicount, AG_count/dicount, AT_count/dicount, CA_count/dicount, CC_count/dicount, CG_count/dicount, CT_count/dicount, GA_count/dicount, GC_count/dicount, GG_count/dicount, GT_count/dicount, TA_count/dicount, TC_count/dicount, TG_count/dicount, TT_count/dicount))
        output_file.close()
        return output_file



    nu_of_genes=[]
    for i in os.listdir('original_datasets/31_riboswitches_new_csv'):

        g=subprocess.Popen(['wc','-l','original_datasets/31_riboswitches_new_csv/{}'.format(i)],stdout=subprocess.PIPE)
        g=g.stdout.read()
        g=g.decode("utf-8")
        g=g.strip("\n")
        nu_of_genes.append(g)

    f=open("count.csv",'w+')
    k=0
    for i,j in zip(os.listdir('original_datasets/31_riboswitches_new_csv'),nu_of_genes):
        print(i,j)
        j=(j.strip(" "))
        j=(j.split(" ")[0])
        f.write("{},{},{}\n".format(i.split(".")[0],k,j))
        k+=1
    f=f.close()

    final_csv="processed_datasets/final_31classes.csv"
    output = open(final_csv,'w+')

    type=0
    for j in os.listdir('original_datasets/31_riboswitches_new_csv'):
        dir='original_datasets/31_riboswitches_new_csv'
        file = open(os.path.join(dir,j),'r')
        next(file)
        file = file.readlines()
        for i in file:
            i = i.strip("\n").split(",")
            output.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(i[1],type,i[2],i[3],i[4],i[5],i[6],i[7],i[8],i[9],i[10],i[11],i[12],i[13],i[14],i[15],i[16],i[17],i[18],i[19],i[20],i[21]))
        type=type+1
    output.close()


def split_dataset():
    def Create_Data(Path):
        data = pd.read_csv(Path)
    #     Data = data.drop('Type', axis=1)
        data.drop_duplicates(keep='first', inplace=True)
        output = data["Type"]
    #     return Data.values, Output.values
        return data, output

    Path = 'processed_datasets/final_32classes.csv'
    Data, Output = Create_Data(Path)

    print (Data)

    #using train_test_split of 90:10 -- given by test_size fraction
    # Data_test = pd.DataFrame(Data_test)
    Data_train, Data_test, Output_train, Output_test = train_test_split(Data, Output, test_size=0.1, stratify=Output)

    print (Data_train['Type'].value_counts())
    print (Data_test['Type'].value_counts())

    file_name = 'processed_datasets/final_32train.csv'
    Data_train.to_csv(file_name, sep=',', encoding='utf-8', index=False)
    file_name = 'processed_datasets/final_32test.csv'
    Data_test.to_csv(file_name, sep=',', encoding='utf-8', index=False)

#Load Datasetfor Base Models
def Load_Data_baseModel(Path, Data, Output):
    with open(Path) as csvfile:
        Data_Path = list(csv.DictReader(csvfile))
        for x in Data_Path:
                #Creating the feature vector of mono and di nucleotides
                Data.append([x["A"], x["T"], x["G"], x["C"],x["AA"], x["AC"], x["AG"], x["AT"],x["CA"], x["CC"], x["CG"], x["CT"],x["GA"], x["GC"], x["GG"], x["GT"],x["TA"], x["TC"], x["TG"], x["TT"]])
                Output.append(x["Type"])
        return Data, Output

# Load Data to be used for DL model training and validation (RNN and CNN)
def load_data(input_file, flag ,test_split = 0.0, maxlen = MAXLEN):
    df = pd.read_csv(input_file)
    df['Sequence'] = df['Sequence'].apply(preprocess.character_mapping)
    df['Sequence'] = df['Sequence'].apply(lambda x: [int(preprocess.letter_to_index(e)) for e in x])
    # df = df.reindex(np.random.permutation(df.index))
    X = np.array(df['Sequence'].values)
    Y = np.array(df['Type'].values)
    if flag:
        global CLASSES
        number_of_classes = np.unique(Y)
        CLASSES = len(number_of_classes)
        print (CLASSES)
    return pad_sequences(X, maxlen=maxlen), Y

#Converting the values to Float for calculations
def Convert_to_Float(Data, Output):
    for i in range(len(Data)):
        for j in range(20):
            Data[i][j]=float(Data[i][j])
        Output[i]= int(Output[i])
    return Data, Output

def get_totalclass(f):
    file = open(f,'r')
    next(file)
    file=file.readlines()
    class_num=0
    for i in file:
        i=i.strip("\n").split(",")
        if int(i[1]) > class_num:
            class_num = int(i[1])
    return class_num + 1

# Convert letters to numbers 
def letter_to_index(letter):
    if letter not in ALLOWED_ALPHABETS:
        print ("Letter not present")
        print (letter)
    return next((i for i, _letter in enumerate(ALLOWED_ALPHABETS) if _letter == letter), None)

# Character mapping to achieve ATGCN
def character_mapping(x):
    repls = {'R' : 'G', 'Y' : 'T', 'M' : 'A', 'K' : 'G', 'S' : 'G', 'W' : 'A', 'H' : 'A', 'B' : 'G', 'V' : 'G', 'D' : 'G'}
    x = functools.reduce(lambda a, kv: a.replace(*kv), repls.items(), x)
    return x

def binarize(outputdata):
    unique_classes = list(set(outputdata))
    unique_classes.sort()
    print (unique_classes)
#    bin_output = label_binarize(Output_test, classes=unique_classes)
    return label_binarize(outputdata, classes=unique_classes)

#Preprocessing the data
def scalingData(Data_train, Data_test=Data_train)
    scaler = StandardScaler()
    scaler.fit(Data_train)
#    Data_train = scaler.transform(Data_train)
#    Data_test = scaler.transform(Data_test)
    return scaler.transform(Data_test)

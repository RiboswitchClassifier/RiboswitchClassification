import subprocess
import os
import re
from Bio import SeqIO
"""
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

"""
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

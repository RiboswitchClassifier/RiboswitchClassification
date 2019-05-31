import subprocess
import os
from Bio import SeqIO

csvfiles = 'RiboApplication/original_datasets/24_riboswitches_new_csv'

subprocess.call(['mkdir',csvfiles])

sample = 'RiboApplication/original_datasets/24_riboswitches_fasta'

for i in os.listdir(sample):
    input_file = open(os.path.join(sample,i), 'r')

    output_file = open("{}/{}.csv".format(csvfiles,i.split(".")[0]),'w')

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
        output_file.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(gene_name,cur_record.seq, round(A_count/monocount,2), round(T_count/monocount,2), round(G_count/monocount,2), round(C_count/monocount,2), round(AA_count/dicount,2), round(AC_count/dicount,2), round(AG_count/dicount,2), round(AT_count/dicount,2), round(CA_count/dicount,2), round(CC_count/dicount,2), round(CG_count/dicount,2), round(CT_count/dicount,2), round(GA_count/dicount,2), round(GC_count/dicount,2), round(GG_count/dicount,2), round(GT_count/dicount,2), round(TA_count/dicount,2), round(TC_count/dicount,2), round(TG_count/dicount,2), round(TT_count/dicount,2)))

"""
    #Now we have finished all the genes, we can close the output file:
count = 0
final_csv="RiboApplication/processed_datasets/final.csv"
output = open(final_csv,'w')
output.write('Sequence,Type,A,T,G,C,AA,AC,AG,AU,CA,CC,CG,CU,GA,GC,GG,GU,UA,UC,UG,UU\n')
for i in os.listdir(csvfiles):
    csv=open(os.path.join(csvfiles,i))
    next(csv)
    csv=csv.readlines()
    for i in csv:

    #print(csv)
    break
"""
final_csv="RiboApplication/processed_datasets/final.csv"
output = open(final_csv,'w')
output.write('Sequence,Type,A,T,G,C,AA,AC,AG,AU,CA,CC,CG,CU,GA,GC,GG,GU,UA,UC,UG,UU\n')
type = 0
for i in sorted(os.listdir(csvfiles)):
    file = open(os.path.join(csvfiles,i))
    next(file)
    file = file.readlines()
    for i in file:
        i = i.strip("\n").split(",")
        output.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(i[1],type,i[2],i[3],i[4],i[5],i[6],i[7],i[8],i[9],i[10],i[11],i[12],i[13],i[14],i[15],i[16],i[17],i[18],i[19],i[20],i[21]))
    type += 1

#!/usr/bin/env python3
import argparse
import subprocess
import os
import re
from Bio import SeqIO


def check_fasta(a):
    with open(a,"r") as f:
        f=f.readlines()
        i=0
        list=[]
        while i < len(f):
            if f[i].startswith(">"):
                list.append(i)
            i=i+1
    list.append(len(f))
    sequences=[]
    j=0
    while j < len(list) -1:
        s=""
        for i in range(list[j],list[j+1]-1):
            i=i+1
            s+=f[i].strip("\n")
        sequences.append(s)
        j=j+1
    DNA = re.compile('[EFIJOPQXYZefijopqxyz]')
    for i in sequences:
        if not DNA.search(i):
            return True
    return False


def makecsv(file):
    input_file = open(file, 'r')
    csvfiles = 'original_datasets/32_riboswitches_new_csv'
    output_file = open("{}/{}.csv".format(csvfiles,file.split(".")[0]),'w')

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



def main():

    parser=argparse.ArgumentParser()
    parser.add_argument("-fa",action="store",help="file1")
    parser.add_argument("-d",help="directory of files")
    args = parser.parse_args()
    try:
        if args.fa is None and args.d is None:
            raise Exception('fasta file or directory of fasta files is required')
        if args.fa is not None and args.d is not None:
            raise Exception('Either fasta file or Directory of fasta files is required, both not acceptable')
        if args.d is not None:
            if len(os.listdir(args.d)) == 0:
                raise Exception('The directory is empty')
        if args.fa is not None:
            if not check_fasta(args.fa):
                raise Exception('File can only contain nucleotide sequences')
        if args.d is not None:
            for files in os.listdir(args.d):
                if not check_fasta(files):
                    raise Exception('File can only contain nucleotide sequences')
    except Exception as e:
        print(e)
    else:
        if args.fa is not None:
            makecsv(args.fa)
            csvfiles = 'original_datasets/32_riboswitches_new_csv'
            csv="{}.csv".format(args.fa.split(".")[0])
            final_csv="processed_datasets/final_dynamic.csv"
            output = open(final_csv,'r')
            lastclass=output.readlines()
            lastclass=int(lastclass[-1].split(",")[1])
            output.close()
            output = open(final_csv,'a')
            file = open(os.path.join(csvfiles,csv))
            next(file)
            file = file.readlines()
            type = lastclass + 1
            for i in file:
                i = i.strip("\n").split(",")
                output.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(i[1],type,i[2],i[3],i[4],i[5],i[6],i[7],i[8],i[9],i[10],i[11],i[12],i[13],i[14],i[15],i[16],i[17],i[18],i[19],i[20],i[21]))
            output.close()

        if args.d is not None:
            for eachfile in os.listdir(args.d):
                makecsv(eachfile)
                csvfiles = 'original_datasets/32_riboswitches_new_csv'
                csv="{}.csv".format(args.fa.split(".")[0])
                final_csv="processed_datasets/final_dynamic.csv"
                output = open(final_csv,'r')
                lastclass=output.readlines()
                lastclass=int(lastclass[-1].split(",")[1])
                output.close()
                output = open(final_csv,'a')
                file = open(os.path.join(csvfiles,csv))
                next(file)
                file = file.readlines()
                type = lastclass + 1
                for i in file:
                    i = i.strip("\n").split(",")
                    output.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(i[1],type,i[2],i[3],i[4],i[5],i[6],i[7],i[8],i[9],i[10],i[11],i[12],i[13],i[14],i[15],i[16],i[17],i[18],i[19],i[20],i[21]))
                output.close()



main()

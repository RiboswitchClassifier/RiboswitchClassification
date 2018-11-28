#!/usr/bin/env python3
import sys
def fasta2csv(s):
    list=[]
    with open(s,"r") as f:
        f=f.readlines()
        i=0
        while i < len(f):
            if f[i].startswith(">"):
                list.append(i)
            i=i+1
    sequences=[]
    j=0
    while j < len(list) - 1:
        s=""
        for i in range(list[j],list[j+1]-1):
            i=i+1
            s+=f[i].strip("\n")
        s+=(",{}".format(sys.argv[2]))
        sequences.append([s])
        j=j+1

    return sequences


def main():
    a = fasta2csv(sys.argv[1])
    output=open(sys.argv[3],"w+")
    for i in a:
        for j in i:
            output.write("{}\n".format(j))
    output.close()
main()

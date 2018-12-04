"""
BioPython Example File - Using FASTA nucleotide files

This is an example Python program to calculate GC percentages for
each gene in an nucleotide FASTA file - using the Biopython
SeqIO library.

It calculates GC percentages for each gene in a FASTA nucleotide file,
writing the output to a tab separated file for use in a spreadsheet.

It has been tested on BioPython 1.43 with Python 2.3, and is suitable
for Windows, Linux etc.

The suggested input file 'NC_005213.ffn' is available from the NCBI
from here:

ftp://ftp.ncbi.nlm.nih.gov/genomes/Bacteria/Nanoarchaeum_equitans/

See associated webpage:

http://www2.warwick.ac.uk/fac/sci/moac/currentstudents/peter_cock/python/fasta_n/

Peter Cock, MOAC, University of Warwick, UK
17 April 2007
"""

#Open a FASTA input file of gene nucleotides sequences:
#input_file = open('NC_005213.ffn', 'r')
# input_file = open('purine.ffn', 'r')
input_file = open('original_datasets/8_riboswitches_fasta/RF01725.fa', 'r')

#Note - you might like to also download the complete
#genome nucleotide sequence, 'NC_005213.ffn' which is
#a single FASTA record.
#input_file = open('NC_005213.fna', 'r')

#Open an output file to record the counts in.
#tsv is short for "Tab Separated Variables",
#also known as "Tab Delimited Format".
#
#This is a universal format, you can read it
#with any text editor - Microsoft Excel is
#also a good choice.
# output_file = open('purine_frequency.csv','w')
output_file = open('processed_datasets/8_riboswitches_csv/RF01725.csv','w')

#We will now write a header line to our output file.
#
#We must write \t to mean a tab, and \n to mean
#an end of line (new line) character.
#
#i.e.
#Gene (tab) A (tab) C (tab) G (tab) T (tab) Length (tab) CG%
output_file.write('Gene\tSequence\tType\tA\tT\tG\tC\tAA\tAC\tAG\tAU\tCA\tCC\tCG\tCU\tGA\tGC\tGG\tGU\tUA\tUC\tUG\tUU\n')

#We are going to need BioPython's SeqIO library, so we
#must tell Python to load this ready for us:
from Bio import SeqIO

#X = ['A','T','G','C','AA','AC','AG','AU','CA','CC','CG','CU','GA','GC','GG','GU','UA','UC','UG','UU']
#Get SeqIO to read this file in "fasta" format,
#and use it to see each record in the file one-by-one
for cur_record in SeqIO.parse(input_file, "fasta") :
    #Because we used the Bio.SeqIO parser, each record
    #is SeqRecord object which includes name and seq
    #properties.
    gene_class = 16
    gene_name = cur_record.name
    gene_sequence = cur_record.seq

    #Just like a string in python, a Biopython sequence
    #object has a 'count' method we can use:
    A_count   = cur_record.seq.count('A')
    T_count   = cur_record.seq.count('T')
    G_count   = cur_record.seq.count('G')
    C_count   = cur_record.seq.count('C')
    AA_count  = cur_record.seq.count('AA')
    AC_count  = cur_record.seq.count('AC')
    AG_count  = cur_record.seq.count('AG')
    AU_count  = cur_record.seq.count('AU')
    CA_count  = cur_record.seq.count('CA')
    CC_count  = cur_record.seq.count('CC')
    CG_count  = cur_record.seq.count('CG')
    CU_count  = cur_record.seq.count('CU')
    GA_count  = cur_record.seq.count('GA')
    GC_count  = cur_record.seq.count('GC')
    GG_count  = cur_record.seq.count('GG')
    GU_count  = cur_record.seq.count('GU')
    UA_count  = cur_record.seq.count('UA')
    UC_count  = cur_record.seq.count('UC')
    UG_count  = cur_record.seq.count('UG')
    UU_count  = cur_record.seq.count('UU')


    output_line = '%s\t%s\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%i\n' % \
    (gene_name, gene_sequence, gene_class, A_count, T_count, G_count, C_count, AA_count, AC_count, AG_count, AU_count, CA_count, CC_count, CG_count, CU_count, GA_count, GC_count, GG_count, GU_count, UA_count, UC_count, UG_count, UU_count)
    output_file.write(output_line)

#Now we have finished all the genes, we can close the output file:
output_file.close()

#and close the input file:
input_file.close()

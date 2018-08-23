out_lines = []
temp_line = ''
with open('/Users/ramitb/Downloads/RF01055.fa','r') as fp:
     for line in fp:
         if line.startswith('>'):
             out_lines.append(temp_line)
             temp_line = line.strip() + '\t'
         else:
             temp_line += line.strip()

with open('/Users/ramitb/Downloads/RF01055.tsv', 'w') as fp_out:
    fp_out.write('\n'.join(out_lines))

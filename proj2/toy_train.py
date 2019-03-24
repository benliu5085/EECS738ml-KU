from HMM_log import *

s = ""
with open("pure_seq.txt",'r') as fin:
    for line in fin:
        s += line[0:-1]

seq = list(s)
alphabet = list("123456")
print("start training!")
(trP, emP, initP) = trainHMM(2, alphabet, seq, 100)
print("training done!")
fout = open("toyhmm_parameter.txt",'w')
fout.write(str(emP.shape[0]) + '\n')

for it in alphabet:
    fout.write(it)
fout.write('\n')

for x in initP:
    fout.write(str(x) + ' ')
fout.write('\n')

for line in trP:
    for x in line:
        fout.write(str(x) + ' ')
    fout.write('\n')
for line in emP:
    for x in line:
        fout.write(str(x) + ' ')
    fout.write('\n')
fout.close()

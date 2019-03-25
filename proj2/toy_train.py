from HMM_log import *

s = ""
with open("pure_seq.txt",'r') as fin:
    for line in fin:
        s += line[0:-1]

seq = list(s)
alphabet = list("123456")

K = range(0,2)      # states, 0, 1, ... , N-1
T = len(seq)        # number of observations
A = np.zeros((len(K), len(K)))
B = np.zeros((len(K), len(alphabet)))
P = np.random.random_sample((len(K)))
"""random initialization"""
P = P/P.sum()

for i in range(0, A.shape[0]):
    t = np.random.random_sample((A.shape[1]))
    A[i,] = t/t.sum()

for i in range(0, B.shape[0]):
    t = np.random.random_sample((B.shape[1]))
    B[i,] = t/t.sum()

print("start training!")
(trP, emP, initP) = trainHMM(2, alphabet, seq, 100, A, B, P)
print("training done!")
fout = open("toyhmm_parameter.txt",'w')
fout.write(str(emP.shape[0]) + '\n')

for it in alphabet:
    fout.write(it + ' ')
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

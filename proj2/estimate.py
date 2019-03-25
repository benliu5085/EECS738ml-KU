from HMM_log import *
import numpy as np
import sys

with open("hmm_parameter.txt",'r') as fin:
    N = 0
    cnt = 0
    alphabet = []
    P = []
    A = []
    B = []
    for line in fin:
        if cnt == 0:
            N = int(line)
        elif cnt == 1:
            alphabet = line[0:-1].split()
        elif cnt == 2:
            cont = line[0:-1].split()
            for it in cont:
                P.append(float(it))
        elif 3 <= cnt < N+3:
            cont = line[0:-1].split()
            tt = []
            for it in cont:
                tt.append(float(it))
            A.append(tt)
        else:
            cont = line[0:-1].split()
            tt = []
            for it in cont:
                tt.append(float(it))
            B.append(tt)
        cnt += 1

initP = np.array(P)
trP   = np.array(A)
emP   = np.array(B)

seq = "a question not to be asked"
if len(sys.argv) == 2:
    seq = sys.argv[1]

test_seq = seq.split()
print("test_seq1: " + str(test_seq))
(V, PTR) = viterbiHMM(alphabet, test_seq, trP, emP, initP)
tscore = np.max(V[-1,])
print(tscore)

np.random.shuffle(test_seq)
print("test_seq2: " + str(test_seq))
(V, PTR) = viterbiHMM(alphabet, test_seq, trP, emP, initP)
print(np.max(V[-1,]))

ss = []
for i in range(0,30):
    np.random.shuffle(test_seq)
    (V, PTR) = viterbiHMM(alphabet, test_seq, trP, emP, initP)
    ss.append(np.max(V[-1,]))

background = np.array(ss)
mb = np.mean(background)
sb = np.std(background)
print(mb)
print(sb)
if tscore > mb:
    print("'" + seq + "' comes from the HMM")
else:
    print("'" + seq + "' doesn't come from the HMM")

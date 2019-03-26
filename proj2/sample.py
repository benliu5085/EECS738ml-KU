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

sl = 10
if len(sys.argv) == 2:
    sl = int(sys.argv[1])
ss = sampleHMM(sl, alphabet, trP, emP, initP)
print(ss)

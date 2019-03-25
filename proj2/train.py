from HMM_log import *

print("loading data...")
data = []
words = set([])
with open("alllines.txt", 'r') as fin:
    for line in fin:
        sentence = line[1:-2].lower()
        t = sentence.split()
        for i in t:
            if i[-1] < 'a' or i[-1] > 'z':
                words.add(i[:-1])
                data.append(i[:-1])
            else:
                words.add(i)
                data.append(i)

alphabet = list(words)
print("loading done!")

N = 10
T = len(data)        # number of observations
A = np.zeros((N, N))
B = np.zeros((N, len(alphabet)))
P = np.random.random_sample(N)

"""random initialization"""
P = P/P.sum()
for i in range(0, B.shape[0]):
    t = np.random.random_sample((B.shape[1]))
    B[i,] = t/t.sum()

for i in range(0, A.shape[0]):
    t = np.random.random_sample((A.shape[1]))
    A[i,] = t/t.sum()

print(A)

print("start training ...")
(trP, emP, initP) = trainHMM(N, alphabet, data, 1, A, B, P)
print("training done!")

fout = open("hmm_parameter.txt",'w')
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

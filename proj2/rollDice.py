import numpy as np

fout1 = open("pure_seq.txt", 'w')
if np.random.random() < 0.4:
    ff_dice = 1
else:
    ff_dice = 2

seq = ""
path = ""
for i in range(1,30001):
    t = np.random.random()
    d = np.random.random()
    if ff_dice == 1:
        if d < 1.0/6:
            seq += '1'
            path += 'F'
        elif d < 2.0/6:
            seq += '2'
            path += 'F'
        elif d < 3.0/6:
            seq += '3'
            path += 'F'
        elif d < 4.0/6:
            seq += '4'
            path += 'F'
        elif d < 5.0/6:
            seq += '5'
            path += 'F'
        else:
            seq += '6'
            path += 'F'

        if t < 0.05:
            ff_dice = 2
    else:
        if d < 0.1:
            seq += '1'
            path += 'L'
        elif d < 0.2:
            seq += '2'
            path += 'L'
        elif d < 0.3:
            seq += '3'
            path += 'L'
        elif d < 0.4:
            seq += '4'
            path += 'L'
        elif d < 0.5:
            seq += '5'
            path += 'L'
        else:
            seq += '6'
            path += 'L'

        if t < 0.1:
            ff_dice = 1

    if i % 60 == 0:
        fout1.write(seq + '\n')
        seq = ""
        path = ""

fout1.close()

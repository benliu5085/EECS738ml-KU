import numpy as np

ACTION_SPACE = {0:'up', 1:'down', 2:'right', 3:'left', 4:'explore'}
GRID_X = 3
GRID_Y = 4

tm = np.zeros([GRID_X,GRID_Y])
tm[0,0] = 0
tm[0,1] = -5
tm[0,2] = 10
tm[0,3] = 0

tm[1,0] = 0
tm[1,1] = -1
tm[1,2] = 0
tm[1,3] = -1

tm[2,0] = 0
tm[2,1] = 0
tm[2,2] = 0
tm[2,3] = 50

fout = open("map.csv",'w')
for i in range(tm.shape[0]):
    for j in range(tm.shape[1]):
        fout.write(str(tm[i,j]))
        fout.write('\t')
    fout.write('\n')
fout.close()

def pickAction(candidates):
    max_id = []
    max_c = candidates.max()

    for i in range(candidates.shape[0]):
        if candidates[i] == max_c:
            max_id.append(i)

    return np.random.choice(max_id, 1)[0]

def sampleAction():
    return np.random.choice(ACTION_SPACE.keys(), 1)[0]

def getReward(M, state, action):
    """ return reward and other info.
        NOTE: each action takes 1 life, no matter what.
        if IE = false:
            return rewards of nearby 4 blocks, take 1 LP off
        if IE = true:
            return rewards of nearby 12 blocks that can be reached by 2 action, take 2 LP off
    """
    x = int(state / GRID_Y)
    y = int(state % GRID_Y)

    if action == 0:
        if x == 0:
            return (-1, x*GRID_Y + y, False)
        else:
            if M[x-1, y] == 0:
                return (-1, (x-1)*GRID_Y + y, False)
            elif M[x-1, y] == -1:
                return (-1, x*GRID_Y + y, False)
            elif M[x-1, y] == -5:
                return (-6, (x-1)*GRID_Y + y, False)
            elif M[x-1, y] == 10:
                M[x-1, y] = 0
                return (9, (x-1)*GRID_Y + y, False)
            elif M[x-1, y] == 50:
                return (49, (x-1)*GRID_Y + y, True)

    elif action == 1:
        if x == GRID_X-1:
            return (-1, x*GRID_Y + y, False)
        else:
            if M[x+1, y] == 0:
                return (-1, (x+1)*GRID_Y + y, False)
            elif M[x+1, y] == -1:
                return (-1, x*GRID_Y + y, False)
            elif M[x+1, y] == -5:
                return (-6, (x+1)*GRID_Y + y, False)
            elif M[x+1, y] == 10:
                M[x+1, y] = 0
                return (9, (x+1)*GRID_Y + y, False)
            elif M[x+1, y] == 50:
                return (49, (x+1)*GRID_Y + y, True)

    elif action == 2:
        if y == GRID_Y-1:
            return (-1, x*GRID_Y + y, False)
        else:
            if M[x, y+1] == 0:
                return (-1, x*GRID_Y + y + 1, False)
            elif M[x, y+1] == -1:
                return (-1, x*GRID_Y + y, False)
            elif M[x, y+1] == -5:
                return (-6, x*GRID_Y + y + 1, False)
            elif M[x, y+1] == 10:
                M[x, y+1] = 0
                return (9, x*GRID_Y + y + 1, False)
            elif M[x, y+1] == 50:
                return (49, x*GRID_Y + y + 1, True)

    elif action == 3:
        if y == 0:
            return (-1, x*GRID_Y + y, False)
        else:
            if M[x, y-1] == 0:
                return (-1, x*GRID_Y + y - 1, False)
            elif M[x, y-1] == -1:
                return (-1, x*GRID_Y + y, False)
            elif M[x, y-1] == -5:
                return (-6, x*GRID_Y + y - 1, False)
            elif M[x, y-1] == 10:
                M[x, y-1] = 0
                return (9, x*GRID_Y + y - 1, False)
            elif M[x, y-1] == 50:
                return (49, x*GRID_Y + y - 1, True)

    elif action == 4:
        rewards = np.zeros([13]) - 3
        final_state = np.zeros([13])
        dones = [False] * 13
        rewards[0] = -10000000
        ## case 3, go up
        reward3, state3, done3 = getReward(M, state, 0)
        rewards[3]    += reward3
        final_state[3] = state3
        dones[3]       = done3
        ## case 1, go up + go up
        reward1, state1, done1 = getReward(M, state3, 0)
        rewards[1]    += reward1 + reward3
        final_state[1] = state1
        dones[1]       = done1

        ## case 10, go down
        reward10, state10, done10 = getReward(M, state, 1)
        rewards[10]    += reward10
        final_state[10] = state10
        dones[10]       = done10
        ## case 12, go down + go down
        reward12, state12, done12 = getReward(M, state10, 1)
        rewards[12]    += reward12 + reward10
        final_state[12] = state12
        dones[12]       = done12

        ## case 7, go right
        reward7, state7, done7 = getReward(M, state, 2)
        rewards[7]    += reward7
        final_state[7] = state7
        dones[7]       = done7
        ## case 8, go right + go right
        reward8, state8, done8 = getReward(M, state7, 2)
        rewards[8]    += reward8 + reward7
        final_state[8] = state8
        dones[8]       = done8

        ## case 6, go left
        reward6, state6, done6 = getReward(M, state, 3)
        rewards[6]    += reward6
        final_state[6] = state6
        dones[6]       = done6

        ## case 5, go left + go left
        reward5, state5, done5 = getReward(M, state6, 3)
        rewards[5]    += reward5 + reward6
        final_state[5] = state5
        dones[5]       = done5

        ## case 2, up+left or left+up
        reward2_1, state2, done2_1 = getReward(M, state3, 3)
        reward2_2, state2, done2_2 = getReward(M, state6, 0)
        reward2 = reward3+reward2_1
        dones[2] = done2_1
        if reward3+reward2_1 < reward6+reward2_2:
            reward2 = reward6+reward2_2
            dones[2] = done2_2
        rewards[2] += reward2
        final_state[2] = state2

        ## case 4, up+right or right+up
        reward4_1, state4, done4_1 = getReward(M, state3, 2)
        reward4_2, state4, done4_2 = getReward(M, state7, 0)
        reward4 = reward3+reward4_1
        dones[4] = done4_1
        if reward3+reward4_1 < reward7+reward4_2:
            reward4 = reward7+reward4_2
            dones[4] = done4_2
        rewards[4] += reward4
        final_state[4] = state4

        ## case 11, down+right or right+down
        reward11_1, state11, done11_1 = getReward(M, state10, 2)
        reward11_2, state11, done11_2 = getReward(M, state7, 1)
        reward11 = reward10+reward11_1
        dones[11] = done11_1
        if reward10+reward11_1 < reward7+reward11_2:
            reward11 = reward7+reward11_2
            dones[11] = done11_2
        rewards[11] += reward11
        final_state[11] = state11

        ## case 9, down+left or left+down
        reward9_1, state9, done9_1 = getReward(M, state10, 3)
        reward9_2, state9, done9_2 = getReward(M, state6, 1)
        reward9 = reward10+reward9_1
        dones[9] = done9_1
        if reward10+reward9_1 < reward6+reward9_2:
            reward9 = reward6+reward9_2
            dones[9] = done9_2
        rewards[9] += reward9
        final_state[9] = state9

        freward = rewards.max()
        fstate = int(final_state[rewards.argmax()])
        fdone = dones[rewards.argmax()]
        return (freward, fstate, fdone)
    else:
        return (None, None, None)

# state_id of (i, j) = i*Y + j, if IE = FALSE
# state_id of (i, j) = i*Y + j + X*Y, if IE = TRUE

q_table = np.zeros([GRID_X*GRID_Y, len(ACTION_SPACE)])

alpha = 0.1
gamma = 0.6
epsilon = 0.2

for i in range(1, 100001):
    state = 0
    epochs, penalties, reward, = 0, 0, 0
    done = False

    while not done:
        """ policy of agent here """
        if np.random.random() < epsilon:
            action = sampleAction()
        else:
            action = pickAction(q_table[state])

        reward, next_state, done = getReward(tm, state, action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1

    if i % 10000 == 0:
        print(str(i) + " Episodes passed!")

print(q_table)
fout = open("q_table.csv",'w')
for i in range(q_table.shape[0]):
    for j in range(q_table.shape[1]):
        fout.write(str(q_table[i,j]))
        fout.write('\t')
    fout.write('\n')
fout.close()

## traceBack
state = 0
done = False
actions = []
while not done:
    action = pickAction(q_table[state])
    actions.append(ACTION_SPACE[action])
    reward, next_state, done = getReward(tm, state, action)
    state = next_state
print(actions)

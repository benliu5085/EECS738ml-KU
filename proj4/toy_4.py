import numpy as np

ACTION_SPACE = {0:'up', 1:'down', 2:'right', 3:'left'} # , 4:'explore'
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
    else:
        return (None, None, None)

# state_id of (i, j) = i*Y + j, if IE = FALSE
# state_id of (i, j) = i*Y + j + X*Y, if IE = TRUE

q_table = np.zeros([GRID_X*GRID_Y, len(ACTION_SPACE)])

alpha = 0.1
gamma = 0.6
epsilon = 0.5

for i in range(1, 100001):
    state = 0
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
        state = next_state

    if i % 10000 == 0:
        print(str(i) + " Episodes passed!")

# print(q_table)
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

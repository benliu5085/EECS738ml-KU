import numpy as np

ACTION_SPACE = {0:'up', 1:'down', 2:'right', 3:'left'} 
GRID_X = 10
GRID_Y = 10

def map(x,y,ss = 27):
    """ generate a x-by-y grid.
        0  - empty grid
        -1 - block, can't cross
        -5 - trap
        10 - life reward
        50 - trophy, win!
    """
    np.random.seed(ss)
    ans = np.zeros([x,y])
    for i in range(0,x):
        for j in range(0,y):
            dice = np.random.random()
            if dice < 0.5:
                if j % 2 == 1:
                    ans[i,j] = -1
            elif dice < 0.6:
                ans[i,j] = -5
            elif dice < 0.7:
                ans[i,j] = 10

    tx = np.random.randint(x)
    ty = np.random.randint(y)
    ans[tx,ty] = 50
    return ans

""" test map """
tm = map(GRID_X, GRID_Y, GRID_X*GRID_X*GRID_Y*GRID_Y)

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
        NOTE: each action takes 1 LP, no matter what.
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
epsilon = 0.2

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

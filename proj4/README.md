This the source code for project 4 Reinforcement Learning, EECS 738 machine learning 
Given by Dr. Martin Kuehnhausen From (author) Ben Liu

===== Ideas ==============================================================

The environment setting:
Given a X-by-Y grid (map), there could be 5 things appearing in each block:
1) nothing
2) a blocker which the agent cannot cross
3) a trap which takes 5 LP to get out of
4) a life-supply which give 10 extra LP to the agent
5) the trophy which give 50 LP and ends the game.

The agent can either take 4 or 5 actions:
1) go up, takes 1 LP to perform
2) go down, takes 1 LP to perform
3) go right, takes 1 LP to perform
4) go left, takes 1 LP to perform
5) explore, takes 3 LP to perform
here explore is a combination of 2 other actions. In detail, when the agent choose to explore, it has to pay 3 LP first to gain the ability to know the environment near it, so it will know all 12 rewards(4 single action and 8 double actions) after 2 actions and it will choose the best one, of course.

I used Q-learning algorithm to train the agent, during training, the agent has 0.2 chance to perform random action, which is explore in the comparison "exploit vs explore". (All explore elsewhere should refer to action explore, unless specified differently). For each agent, I will train the agent by playing 100000 games and update the q_table by the formula online.

===== Files ==============================================================

myTH_4.py -- the treasure hunt game in which the agent can only perform 4 actions.
myTH_5.py -- the treasure hunt game in which the agent can only perform 5 actions, including explore.
toy_4.py  -- code for the toy example latter
toy_5.py  -- code for the toy example latter
map_10.csv -- the map file, 0 for nothing, -1 for blocker, -5 for trap, 10 for life-supply, 50 for trophy.
all q_table_xx.csv are the final q_table after training.

===== toy example and analysis ===========================================

for simplicity, I will use a toy example to explain what happens when the agent can perform explore action.

The toy map is a 3-by-4 grid:

STHN

NXNX

NNNE

where:
N is empty block.
S is the place where the agent start from.
E is the trophy.
T stands for trap, we only have 1 here.
H stands for life-supply(heart).
X stands for blocker, they are at (1,1) and (1,3), so the agent can never go to (1,3) from (1,0) since it cannot cross the blocker.

By calling toy_4.py, the final solution the agent made is ['down', 'down', 'right', 'right', 'right'], so it finished the goal, with a final LP of 45.
By calling toy_5.py, then final solution the agent made is ['right', 'right', 'down', 'explore', 'down', 'right'], so it finished the goal too, but with a final LP of 50. Apparently, with the ability to explore the local environment, the agent is able to find a more globally optimal solution to finish the treasure hunting game. 
I use "more" above because my code cannot actually guarantee the global optimality of the final solution, for example, again, if the agent starts from (1,0), the global optimal solution should be ['down', 'right', 'right', 'up', 'up', 'down', 'down', 'right'] with final LP of 52, but both of the agent above will give the same solution ['down', 'right', 'right', 'right'] with LP of 46.

Another notable difference is that it takes about thousands times more time to train the 4-action agent than the 5-action agent on 10-by-10 map, it proves that explore can help the agent identify where the trophy is and push the agent acting to get closer to the trophy, rather than random wandering around and hopefully find the trophy. 

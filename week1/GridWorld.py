from __future__ import print_function
import numpy as np

SIZE = 4
R = -1.0
PROB = 0.25

board = np.zeros((SIZE, SIZE))

possible_actions = ['L', 'U', 'R', 'D']

next_i = []
for i in range(0, SIZE):
    next_i.append([])
    for j in range(0, SIZE):
        next_action = {}
        if i == 0:
            next_action['U'] = [i, j]
        else:
            next_action['U'] = [i - 1, j]

        if i == SIZE - 1:
            next_action['D'] = [i, j]
        else:
            next_action['D'] = [i + 1, j]

        if j == 0:
            next_action['L'] = [i, j]
        else:
            next_action['L'] = [i, j - 1]

        if j == SIZE - 1:
            next_action['R'] = [i, j]
        else:
            next_action['R'] = [i, j + 1]

        next_i[i].append(next_action)

states = []
for l in np.arange(0, SIZE):
    for k in np.arange(0, SIZE):
        if not ((l == SIZE - 1 and k == SIZE - 1) or (l == 0 and k == 0)):
            states.append([l, k])
convergence=False
count=0
while convergence!=True:
    newboard = np.zeros((SIZE, SIZE))
    for m, n in states:
        for action in possible_actions:
            newpos = next_i[m][n][action]
            newboard[m, n] += PROB * (R + board[newpos[0], newpos[1]])
    count+=1
    if np.sum(np.abs(board - newboard)) < 1e-4:
        print('Figure 4.1')
        print(newboard)
        print('Policy Convergence occurred at k='+str(len(states)))
        convergence=True
    board = newboard
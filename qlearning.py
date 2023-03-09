import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import sys


def move_up(coord: tuple[int, int]) -> tuple[int, int]:
    x, y = coord
    return (x, y + 1)

def move_left(coord: tuple[int, int]) -> tuple[int, int]:
    x, y = coord
    return (x - 1, y)

def move_right(coord: tuple[int, int]) -> tuple[int, int]:
    x, y = coord
    return (x + 1, y)

def move_down(coord: tuple[int, int]) -> tuple[int, int]:
    x, y = coord
    return (x, y - 1)


ACTION_MAP = {0: move_up, 1: move_left, 2: move_right, 3: move_down}
NB_STATES_PER_CENTER_CELL = 16
NB_STATES_PER_BORDER_CELL = 8
NB_STATES_PER_CORNER_CELL = 4


class QLearning:
    def __init__(self, N, M, init_coord=(0,0), eps=0.1, discount=0.6, learning=0.1, decay=0.5, nb_train=100):
        print("Initialized Q-Learing")
        self.N = N
        self.M = M
        self.init_coord = init_coord
        self.eps = eps
        self.discount = discount
        self.learning = learning
        self.decay = decay
        self.nb_train = nb_train
        self.nb_actions = len(ACTION_MAP)
        self.visited_reward = 0
        self.unvisited_reward = -1
        self.last = None

        self.pos2coord = self.init_pos2coord()
        self.nb_states = self.get_nb_states()

        self.Q = np.zeros((self.nb_states, self.nb_actions))
        self.n = np.zeros((self.nb_states, self.nb_actions))
        self.visited_pos = set()

        self.corner_cell_states = {'ff': 1, 'ft': 2, 'tf': 3, 'tt': 4}
        self.border_cell_states = {'fff': 1, 'fft': 2, 'ftf': 3, 'ftt': 4, 'tff': 5, 'tft': 6, 'ttf': 7, 'ttt': 8}
        self.center_cell_states = {'ffff': 1, 'ffft': 2, 'fftf': 3, 'fftt': 4, 'ftff': 5, 'ftft': 6, 'fttf': 7, 'fttt': 8,
                                   'tfff': 9, 'tfft': 10, 'tftf': 11, 'tftt': 12, 'ttff': 13, 'ttft': 14, 'tttf': 15, 'tttt': 16}

  
    def get_cell_type(self, coord):
        x, y = coord
        if x == 0 or x == self.N - 1:
            if y == 0 or y == self.M - 1:
                type = "corner"
            else:
                type = "border"
        elif y == 0 or y == self.M - 1:
            type = "border"
        else:
            type = "center"
        return type
            

    def eps_greedy(self, coord):
        valid_actions = self.get_valid_actions(coord)
        if random.uniform(0,1) < self.eps:
            action = random.choice(valid_actions)
        else:
            state = self.get_state(coord)
            action = valid_actions[np.argmax([self.Q[state, a] for a in valid_actions])]
        return action


    def get_valid_actions(self, coord):
        valid_actions = []
        pos = self.get_position(coord)
        if pos >= self.N:
            valid_actions.append(3)
        if pos % self.N != 0:
            valid_actions.append(1)
        if pos % self.N != self.N - 1:
            valid_actions.append(2)
        if pos < self.N * (self.M - 1):
            valid_actions.append(0)
        return valid_actions
    

    def get_position(self, coord):
        x, y = coord
        return y * self.N + x
    

    def get_state(self, coord):
        pos = self.get_position(coord)
        previous_pos = np.arange(0, pos)
        temp_state = 0
        for prev_pos in previous_pos:
            prev_coord = self.pos2coord[prev_pos]
            type = self.get_cell_type(prev_coord)
            if type == "center":
                temp_state += NB_STATES_PER_CENTER_CELL
            elif type == "border":
                temp_state += NB_STATES_PER_BORDER_CELL
            else:
                temp_state += NB_STATES_PER_CORNER_CELL

        state = temp_state + self.get_state_in_cell(coord) - 1
        return state


    def init_pos2coord(self):
        pos2coord = {}
        for x in range(self.N):
            for y in range(self.M):
                pos = self.get_position((x,y))
                pos2coord[pos] = (x,y)
        return pos2coord
    
    
    def get_state_in_cell(self, coord):
        x, y = coord
        str = ''
        if self.get_cell_type(coord) == 'corner':
            if x == 0 and y == 0:
                str += 't' if self.get_position((x, y+1)) in self.visited_pos else 'f'
                str += 't' if self.get_position((x+1, y)) in self.visited_pos else 'f'
            elif x == 0 and y == self.M - 1:
                str += 't' if self.get_position((x+1, y)) in self.visited_pos else 'f'
                str += 't' if self.get_position((x, y-1)) in self.visited_pos else 'f'
            elif x == self.N - 1 and y == 0:
                str += 't' if self.get_position((x, y+1)) in self.visited_pos else 'f'
                str += 't' if self.get_position((x-1, y)) in self.visited_pos else 'f'
            else:
                str += 't' if self.get_position((x-1, y)) in self.visited_pos else 'f'
                str += 't' if self.get_position((x, y-1)) in self.visited_pos else 'f'

            state = self.corner_cell_states[str]

        elif self.get_cell_type(coord) == 'border':
            if x == 0:
                str += 't' if self.get_position((x, y+1)) in self.visited_pos else 'f'
                str += 't' if self.get_position((x+1, y)) in self.visited_pos else 'f'
                str += 't' if self.get_position((x, y-1)) in self.visited_pos else 'f'
            elif x == self.N - 1:
                str += 't' if self.get_position((x, y+1)) in self.visited_pos else 'f'
                str += 't' if self.get_position((x-1, y)) in self.visited_pos else 'f'
                str += 't' if self.get_position((x, y-1)) in self.visited_pos else 'f'
            elif y == 0:
                str += 't' if self.get_position((x, y+1)) in self.visited_pos else 'f'
                str += 't' if self.get_position((x-1, y)) in self.visited_pos else 'f'
                str += 't' if self.get_position((x+1, y)) in self.visited_pos else 'f'
            else:
                str += 't' if self.get_position((x-1, y)) in self.visited_pos else 'f'
                str += 't' if self.get_position((x+1, y)) in self.visited_pos else 'f'
                str += 't' if self.get_position((x, y-1)) in self.visited_pos else 'f'
            
            state = self.border_cell_states[str]
        
        else:
            str += 't' if self.get_position((x, y+1)) in self.visited_pos else 'f'
            str += 't' if self.get_position((x-1, y)) in self.visited_pos else 'f'
            str += 't' if self.get_position((x+1, y)) in self.visited_pos else 'f'
            str += 't' if self.get_position((x, y-1)) in self.visited_pos else 'f'

            state = self.center_cell_states[str]
        return state


    def get_nb_states(self):
        all_pos = np.arange(0, self.N * self.M)
        nb_states = 0
        for pos in all_pos:
            coord = self.pos2coord[pos]
            type = self.get_cell_type(coord)
            if type == "center":
                nb_states += NB_STATES_PER_CENTER_CELL
            elif type == "border":
                nb_states += NB_STATES_PER_BORDER_CELL
            else:
                nb_states += NB_STATES_PER_CORNER_CELL
        return nb_states


    def visited_all_cells(self):
        if len(self.visited_pos) >= 0.7* self.N * self.M:
            return True
        return False

    def transition(self, coord, action):
        new_coord = ACTION_MAP[action](coord)
        if self.get_position(new_coord) in self.visited_pos:
            reward = self.visited_reward
        else:
            reward = self.unvisited_reward
        return new_coord, reward

    def update(self, coord, new_coord, action, reward):
        s = self.get_state(coord)
        s1 = self.get_state(new_coord)
        self.Q[s, action] += self.learning * (reward + self.discount*max(self.Q[s1,:]) - self.Q[s, action])


    def training(self):
        for i in tqdm(range(self.nb_train), desc='Trainings'):
            self.visited_pos.clear()
            coord = self.init_coord
            #coord = (random.randint(0,self.N-1), random.randint(0,self.M-1))
            self.visited_pos.add(self.get_position(coord))
            while not self.visited_all_cells():
                action = self.eps_greedy(coord)
                new_coord, reward = self.transition(coord, action)
                self.update(coord, new_coord, action, reward)
                coord = new_coord
                self.visited_pos.add(self.get_position(coord))
                


    def show(self):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)

        # create contour
        ax.plot([-1, self.N], [-1, -1], c='b')
        ax.plot([-1, self.N], [self.M, self.M], c='b')
        ax.plot([-1, -1], [-1, self.M], c='b')
        ax.plot([self.N, self.N], [-1, self.M], c='b')

        # create grid and ticks
        xmajor_ticks = np.arange(0, self.N, 5)
        xminor_ticks = np.arange(0, self.N, 1)
        ymajor_ticks = np.arange(0, self.M, 5)
        yminor_ticks = np.arange(0, self.M, 1)

        ax.set_xticks(xmajor_ticks)
        ax.set_xticks(xminor_ticks, minor=True)
        ax.set_yticks(ymajor_ticks)
        ax.set_yticks(yminor_ticks, minor=True)

        ax.grid(which='both', zorder=-10.0)

        # plot trajectory
        coord = self.init_coord
        visited_during_plotting = [coord]
        
        while len(visited_during_plotting) < self.N * self.M:
            valid_actions = self.get_valid_actions(coord)
            state = self.get_state(coord)
            action = valid_actions[np.argmax([self.Q[state, a] for a in valid_actions])]
            new_coord = ACTION_MAP[action](coord)
            x0, y0 = coord
            x1, y1 = new_coord
            ax.arrow(x0, y0, (x1-x0)*0.8, (y1-y0)*0.8, width=0.1, head_width=0.3, color="r")
            coord = new_coord
            visited_during_plotting.append(coord)
        
        
        plt.show()












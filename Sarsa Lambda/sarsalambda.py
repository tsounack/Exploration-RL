import matplotlib.pyplot as plt
import numpy as np
import random
# petit test

from tqdm import tqdm

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


ACTION_MAP = {0: move_up, 1: move_left, 2: move_down, 3: move_right}


class SarsaLambda:
    """
    This class is used to find the best policy for a given dataset using the sarsa
    lambda algorithm.
    """
    
    def __init__(self, n: int, m: int, discount: float,
                 learning: float, decay: float, epsilon: float, nb_events: int, initial_coord: tuple[int, int]) -> None:
        """
        This function initiates the object using various parameters entered by the user.
        """
        self.n         = n
        self.m         = m
        self.nb_state  = self.n * self.m * 2
        self.nb_action = len(ACTION_MAP)
        self.nb_events = nb_events

        self.discount   = discount
        self.learning   = learning
        self.decay      = decay
        self.epsilon    = epsilon

        # Initialising the various penalties
        self.visited_reward = 0
        self.unvisited_reward = 10       
        self.obstacle_reward = -1000     # (we never want to hit an obstacle)
        self.gostraight_reward = 10
        self.turn90_reward     = 0

        self.obstacles = [[(10, 15), (20, 35)],             # bottom left and top right corner
                            [(50, 1), (60, 4)]]

        self.last          = None
        self.initial_coord = initial_coord
        self.initial_s     = self.get_state_index(self.initial_coord, 0)
        self.direction     = 0

        self.visited = set(self.initial_coord)
        self.obstacle_states = set()
        self.state_coord_map = self.create_state_coord_map()

        self.Q = np.zeros((self.nb_state, self.nb_action))
        self.N = np.zeros((self.nb_state, self.nb_action))
        


    def create_state_coord_map(self):
        """
        Creates the dictionary that maps state indexes with coordinates
        """
        state_coord_map = {}

        for x in range(1, self.n + 1):
            for y in range(1, self.m + 1):
                for visited in [0, 1]:
                    state = self.get_state_index((x, y), visited)
                    state_coord_map[state] = (x, y)

        return state_coord_map


    def epsilon_greedy(self, curr_state: int) -> int:
        """
        Implements the epsilon greedy policy to choose the next action.
        It takes the current state as argument.
        """
        valid_actions = self.get_valid_actions(self.state_coord_map[curr_state])

        if np.random.uniform(0, 1) < self.epsilon:
            action = random.choice(valid_actions)
        else:
            action = valid_actions[np.argmax([self.Q[curr_state, a] for a in valid_actions])]
        return action

    
    def update(self, s, a, r) -> None:
        """
        Helper method for the simulate method. Updates Q and N based on the
        current state's observations
        """
        if self.last != None:
            ls, la, lr = self.last[0], self.last[1], self.last[2]
            self.N[ls, la] += 1
            delta = lr + self.discount * self.Q[s, a] - self.Q[ls, la]
            
            self.Q  += self.learning * delta * self.N
            self.N  *= self.discount * self.decay
        self.last = [s, a, r]


    def simulate(self, init_state: int) -> None:
        """
        Runs the sarsa lambda algorithm *nb_events* times. The policy chosen is
        epsilon greedy
        """
        s = init_state

        for event in range(self.nb_events):
            a = self.epsilon_greedy(s)
            s1, r = self.transition(s, a)
            self.update(s, a, r)
            s = s1
            self.direction = a
        

    def run_simulations(self, nb_simulations) -> None:
        """
        
        """
        for simulation in tqdm(range(nb_simulations), desc="Simulations"):
            init_state = self.randomize_init()
            self.simulate(init_state)


        
    def randomize_init(self) -> int:
        """
        
        """
        init_coord = (random.randint(1, self.n), random.randint(1, self.m))
        init_state = self.get_state_index(init_coord, 1)

        nb_visited = random.randint(0, self.n * self.m - 1)
        self.visited = set()

        for visits in range(nb_visited):
            x, y = random.randint(1, self.n), random.randint(1, self.m)
            self.visited.add((x, y))

        self.visited.add(init_coord)

        self.direction = random.randint(1, 4)

        return init_state


    def transition(self, s: int, a: int) -> tuple[int, int]:
        """
        Returns the next state and reward for a given state and action
        """
        new_coord = ACTION_MAP[a](self.state_coord_map[s])

        # if the new coordinate has been visited
        if new_coord in self.visited:
            visited = 1

            if new_coord in self.obstacle_states:
                reward = self.obstacle_reward

            else:
                reward = self.visited_reward
        
        # if this is the first time we get to this next coordinate
        else:
            visited = 0

            if self.is_in_obstacle(new_coord):
                reward = self.obstacle_reward
                self.obstacle_states.add(new_coord)

            else:
                reward = self.unvisited_reward
            
            self.visited.add(new_coord)

        # reward keeping the same direction, and penalize turning
        if a == self.direction:
            reward += self.gostraight_reward
        else:
            reward += self.turn90_reward

        
        return (self.get_state_index(new_coord, visited), reward)
        

    def get_state_index(self, coord: tuple[int, int], visited: bool) -> int:
        """
        Returns the state index for a given coordinate tuple and the information
        on whether this tuple has been visited or not
        """
        position = self.get_position(coord)
        return (position - 1) * 2 + visited


    def get_position(self, coord: tuple[int, int]) -> int:
        """
        Returns the position on the grid for a given coordinate tuple
        """
        x, y = coord
        return (y - 1) * self.n + x


    def is_in_obstacle(self, coord: tuple[int, int]) -> bool:
        """
        Given a coordinate tuple, returns a tuple corresponding to whether we are
        in an obstacle (boundary included) or not
        """
        x, y = coord
        for obstacle in self.obstacles:
            x1, y1 = obstacle[0]
            x2, y2 = obstacle[1]
            if x >= x1 and x <= x2 and y >= y1 and y <= y2:
                return True
        return False


    def get_valid_actions(self, coord: tuple[int, int]) -> list:
        """
        Gather list of possible actions taking the bounds of the matrix into account
        """
        position = self.get_position(coord)
        valid_actions = []

        ### TODO: Probably easier to just use the coordinates ###

        # up
        if position <= self.n * (self.m - 1):
            valid_actions.append(0)
        # right
        if position % self.n != 0:
            valid_actions.append(3)
        # left
        if position % self.n != 1:
            valid_actions.append(1)
        # down
        if position > self.n:
            valid_actions.append(2)
        
        # remove turning around
        invalid_action = (self.direction + 2) % 4
        if invalid_action in valid_actions:
            valid_actions.remove(invalid_action)

        return valid_actions
    

    def save_policy_text(self, dir: str) -> None:
        """
        This function saves the policy as a .policy file.
        """
        with open(dir + ".policy", "w") as f:
            for state in range(self.nb_state):
                action = np.argmax(self.Q[state]) + 1
                f.write(str(action) + "\n")

    def show(self):
        """
        Plots the corresponding trajectory
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)

        # create contour
        ax.plot([0, self.n + 1], [0, 0], c='b')
        ax.plot([0, self.n + 1], [self.m + 1, self.m + 1], c='b')
        ax.plot([0, 0], [0, self.m + 1], c='b')
        ax.plot([self.n + 1, self.n + 1], [0, self.m + 1], c='b')

        # create obstacles
        for obstacle in self.obstacles:
            (x_bl, y_bl) = obstacle[0]
            (x_tr, y_tr) = obstacle[1]
            ax.plot([x_bl, x_tr], [y_bl, y_bl], c='b')
            ax.plot([x_bl, x_tr], [y_tr, y_tr], c='b')
            ax.plot([x_bl, x_bl], [y_bl, y_tr], c='b')
            ax.plot([x_tr, x_tr], [y_bl, y_tr], c='b')


        # create grid and ticks
        xmajor_ticks = np.arange(0, self.n+1, 5)
        xminor_ticks = np.arange(0, self.n+1, 1)
        ymajor_ticks = np.arange(0, self.m+1, 5)
        yminor_ticks = np.arange(0, self.m+1, 1)

        ax.set_xticks(xmajor_ticks)
        ax.set_xticks(xminor_ticks, minor=True)
        ax.set_yticks(ymajor_ticks)
        ax.set_yticks(yminor_ticks, minor=True)

        ax.grid(which='both', zorder=-10.0)


        # plot exploration
        x0, y0 = (1, 1)
        s      = self.get_state_index((1, 1), 1)
        self.direction = 0

        #### TODO: FIND THE EXACT NUMBER OF ITERATIONS TO PERFORM ####

        # Trajectory simulator

        # for _ in tqdm(range(self.n * self.m * 10), desc="Plotting"):
        #     valid_actions = self.get_valid_actions((x0, y0))
        #     action = valid_actions[np.argmax([self.Q[s, a] for a in valid_actions])]
        #     x1, y1 = ACTION_MAP[action]((x0, y0))
        #     ax.arrow(x0, y0, (x1-x0)*0.8, (y1-y0)*0.8, width=0.1, head_width=0.3, color="r")

        #     x0, y0 = x1, y1
        #     s = self.get_state_index((x1, y1), 1)
        #     self.direction = action


        # Best action for each state

        # for x in range(1, self.n + 1):
        #     for y in range(1, self.m + 1):
        #         coord = (x, y)
        #         if coord in self.visited:
        #             s = self.get_state_index(coord, 1)
        #             valid_actions = self.get_valid_actions(coord)
        #             action = valid_actions[np.argmax([self.Q[s, a] for a in valid_actions])]
                    
        #             if action == 0:
        #                 ax.arrow(x, y-0.2, 0, 0.4, width=0.1, head_width=0.3, color="r")

        #             elif action == 1:
        #                 ax.arrow(x+0.2, y, -0.4, 0, width=0.1, head_width=0.3, color="r")

        #             elif action == 2:
        #                 ax.arrow(x-0.2, y, 0.4, 0, width=0.1, head_width=0.3, color="r")
                    
        #             elif action == 3:
        #                 ax.arrow(x, y+0.2, 0, -0.4, width=0.1, head_width=0.3, color="r")



        plt.show()

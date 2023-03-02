import numpy as np
import matplotlib.pyplot as plt
import sys


def move_up(x, y):
    return x, y + 1

def move_down(x, y):
    return x, y - 1

def move_left(x, y):
    return x - 1, y

def move_right(x, y):
    return x + 1, y


ACTION_MAP = {move_up: 1, move_down: 2, move_left: 3, move_right: 4}


class OptimalExploration():
    def __init__(self):
        """
        Defining the various attributes of our class
        """
        # Dimension of the grid
        self.N = 60
        self.M = 35

        # Discount factor
        self.discount = 0.1

        # Initialising the various penalties
        self.state_penalty = -30
        self.neighb_penalty = -100          
        self.obstacle_penalty = -1000     # (we never want to hit an obstacle)

        # Initial coordinates and state
        self.curr_x_coord = 1
        self.curr_y_coord = 1
        self.init_state = self.get_state(self.curr_x_coord, self.curr_y_coord)
        self.curr_state = self.init_state

        self.states_explored = [self.curr_state]

        # Reward and Utility matrices
        self.R = np.ones(self.N * self.M)
        self.U = np.ones(self.N * self.M)
        self.update_local_reward(self.init_state)

        self.state_coord_map = self.create_state_coord_map(self.N, self.M)

        self.actions = [move_up, move_down, move_left, move_right]

        self.obstacles = [[(10, 15), (20, 35)],             # bottom left and top right corner
                            [(50, 1), (60, 4)]]

        self.update_obstacles_rewards()


    def create_state_coord_map(self, N, M):
        """
        Defines a dictionary that uses states as keys and coordinate tuples as 
        values
        """
        state_coord_map = {}

        for m in range(1, self.M+1):
            for n in range(1, self.N+1):
                state = self.get_state(n, m)
                state_coord_map[state] = (n, m)

        return state_coord_map


    def get_state(self, x, y):
        """
        Returns the state linked with x and y coordinates
        """
        return (y - 1) * self.N + x


    def update_obstacles_rewards(self):
        """
        Updates the rewards matrix to take the obstacles in account
        """
        for obstacle in self.obstacles:
            (x_bl, y_bl) = obstacle[0]
            (x_tr, y_tr) = obstacle[1]

            for x in range(x_bl, x_tr+1):
                for y in range(y_bl, y_tr+1):
                    state = self.get_state(x,y)
                    self.R[state-1] = self.obstacle_penalty


    def get_valid_actions(self, x, y):
        """
        Gather list of possible actions taking the bounds of the matrix into account
        """
        state = self.get_state(x, y)
        valid_actions = []

        if state <= self.N * (self.M - 1):
            valid_actions.append(move_up)
        if state > self.N:
            valid_actions.append(move_down)
        if state % self.N != 0:
            valid_actions.append(move_right)
        if state % self.N != 1:
            valid_actions.append(move_left)

        return valid_actions


    def update_local_reward(self, state):
        """
        Updates the Reward matrix for the current state and the neighbours of
        the current state, assigning penalties to places that have already been
        explored
        """
        self.R[state-1] += self.state_penalty
        surr_states = self.get_surrounding_states(state)
        
        for state in surr_states:
            self.R[state-1] += self.neighb_penalty


    def get_surrounding_states(self, state):
        """
        Obtains the list of states that are being explored when the vehicle
        is in a state. This corresponds to the wingspan of the vehicle
        """
        # 9 cases: 4 vertices, 4 edges, central
        
        # Top row
        if state > self.N * (self.M - 1):
            if state % self.N == 1:             #top left corner
                surr_states = [state+1, state-self.N, state-self.N+1]
            elif state % self.N == 0:           #top right corner
                surr_states = [state-1, state-self.N, state-self.N-1]
            else:
                surr_states = [state+1, state-1, state-self.N, state-self.N+1, state-self.N-1]

        # Bottom row
        elif state < self.N:
            if state % self.N == 1:             #bottom left corner
                surr_states = [state+1, state+self.N, state+self.N+1]
            elif state % self.N == 0:           #bottom right corner
                surr_states = [state-1, state+self.N, state+self.N-1]
            else:
                surr_states = [state+1, state-1, state+self.N, state+self.N+1, state+self.N-1]

        # Right column
        elif state % self.N == 0:               #right edge but not corner
            surr_states = [state+self.N, state-self.N, state+self.N-1, state-self.N-1, state-1]

        # Left column
        elif state % self.N == 1:               #left edge but not corner
            surr_states = [state+self.N, state-self.N, state+self.N+1, state-self.N+1, state+1]

        # Centre
        else:
            surr_states = [state+1, state-1, state+self.N, state+self.N+1, state+self.N-1, state-self.N, state-self.N+1, state-self.N-1]

        return surr_states


    def simulation(self, N):
        """
        The simulation runs the RL algorithm. 
        """
        for _ in range(N):                              #### UPDATE TO WHILE WITH LOSS FUNCTION? ####
            # get valid actions for current state
            valid_actions = self.get_valid_actions(self.curr_x_coord, self.curr_y_coord)

            utilities = []
            # iterate over all 4 possible actions
            for action in self.actions:
                if action not in valid_actions:
                    utilities.append(-np.inf)
                else:
                    # get the coordinates corresponding to the next state after taking this action
                    new_x, new_y = action(self.state_coord_map[self.curr_state][0], self.state_coord_map[self.curr_state][1])
                    new_state = self.get_state(new_x, new_y)
                    utilities.append(self.R[new_state-1] + self.discount * self.U[new_state-1])

            # choose action that maximizes utility
            self.U[new_state-1] = np.amax(utilities)
            new_action_idx = np.argmax(utilities)

            new_action = self.actions[new_action_idx]

            # update coordinates to reflect chosen action
            self.curr_x_coord, self.curr_y_coord = new_action(self.curr_x_coord, self.curr_y_coord)
            self.curr_state = self.get_state(self.curr_x_coord, self.curr_y_coord)
            #print("new action: ", new_action)
            #print("current state: ", self.curr_state)
            #print()
            self.update_local_reward(self.curr_state)
            self.states_explored.append(self.curr_state)


    def show(self):
        """
        Plots the corresponding trajectory
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)

        # create contour
        ax.plot([0, self.N + 1], [0, 0], c='b')
        ax.plot([0, self.N + 1], [self.M + 1, self.M + 1], c='b')
        ax.plot([0, 0], [0, self.M + 1], c='b')
        ax.plot([self.N + 1, self.N + 1], [0, self.M + 1], c='b')

        # create obstacles
        for obstacle in self.obstacles:
            (x_bl, y_bl) = obstacle[0]
            (x_tr, y_tr) = obstacle[1]
            ax.plot([x_bl, x_tr], [y_bl, y_bl], c='b')
            ax.plot([x_bl, x_tr], [y_tr, y_tr], c='b')
            ax.plot([x_bl, x_bl], [y_bl, y_tr], c='b')
            ax.plot([x_tr, x_tr], [y_bl, y_tr], c='b')


        # create grid and ticks
        xmajor_ticks = np.arange(0, self.N+1, 5)
        xminor_ticks = np.arange(0, self.N+1, 1)
        ymajor_ticks = np.arange(0, self.M+1, 5)
        yminor_ticks = np.arange(0, self.M+1, 1)

        ax.set_xticks(xmajor_ticks)
        ax.set_xticks(xminor_ticks, minor=True)
        ax.set_yticks(ymajor_ticks)
        ax.set_yticks(yminor_ticks, minor=True)

        ax.grid(which='both', zorder=-10.0)


        # plot exploration
        for i, state in enumerate(self.states_explored):
            if i != 0:
                (x0, y0) = self.state_coord_map[self.states_explored[i - 1]]
                (x1, y1) = self.state_coord_map[state]
                # ax.scatter(x0, y0, c='r', s=5, zorder=10.0)
                ax.arrow(x0, y0, (x1-x0)*0.8, (y1-y0)*0.8, width=0.1, head_width=0.3, color="r")

        plt.show()


def solve():
    """
    Function to run the script
    """
    optimal_exploration = OptimalExploration()
    optimal_exploration.simulation(1500)
    optimal_exploration.show()


def main():
    if len(sys.argv) != 1:
        raise Exception("Usage: python3 final_project_test.py")

    solve()

if __name__ == '__main__':
    main()
















import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from scipy.spatial.distance import cdist

class Environment:
    """
    
    """

    def __init__(self) -> None:
        """
        
        """
        # Define the dimensions of the environment
        self.n = 50
        self.m = 30
        self.obstacle_reward = -100000
        self.distance_reward = -2
        self.orientation_reward = 2
        self.car_width = 2
        self.n_obstacles = 6

        self.obstacles = []
        self._generate_obstacles(self.n_obstacles)
        self.stops = self._generate_states()
        self.visited = []
        self.best_path = []

        self.borders = self._get_borders()

        self.s_to_coord, self.coord_to_s = self._get_dictionaries()

        self.n_stops = len(self.stops)

        self.Q = -cdist(self.stops, self.stops, 'euclidean')

        self.reset()
        
 
    
    def _generate_obstacles(self, n_obst) -> None:
        """
        
        """

        for _ in range(n_obst):
            dx = np.random.randint(2, self.n/5 + 1)
            dy = np.random.randint(2, self.m/5 + 1)
            x1 = np.random.randint(0, self.n + 1 - dx)
            y1 = np.random.randint(dy, self.m + 1)
            x2 = x1 + dx
            y2 = y1 - dy
            self.obstacles.append(((x1, y1), (x2, y2)))
        

    def _generate_states(self):
        x = list(np.arange(self.car_width, self.n, self.car_width))
        y = list(np.arange(self.car_width, self.m, self.car_width))
        if self.n - x[-1] < self.car_width:
            x = x[:-1]
            x.append(self.n - self.car_width)
        
        if self.m - y[-1] < self.car_width:
            y = y[:-1]
            y.append(self.m - self.car_width)
        
        temp_stops = [(i,j) for j in y for i in x]

        stops = []
        for stop in temp_stops:
            if not self._is_in_obstacle(stop):
                for obst in self.obstacles:
                    stop = self._adapt_states(stop, obst)
                stops.append(stop)
        return stops
    

    def _get_dictionaries(self):
        s_to_coord = dict()
        coord_to_s = dict()

        for i, stop in enumerate(self.stops):
            s_to_coord[i]    = stop
            coord_to_s[stop] = i
        
        return s_to_coord, coord_to_s


    def _is_in_obstacle(self, stop):
        x, y = stop
        for obst in self.obstacles:
            x1, y1 = obst[0]
            x2, y2 = obst[1]
            if (x2 >= x >= x1) and (y2 <= y <= y1):
                return True
        return False
    
    
    def _adapt_states(self, stop, obstacle):
        x, y = stop
        x1, y1 = obstacle[0]
        x2, y2 = obstacle[1]
        if np.abs(x - x1) < self.car_width and y >= y2 and y <= y1:
            stop2 = (x1 - self.car_width, y)
        elif np.abs(y - y1) < self.car_width and x >= x1 and x <= x2:
            stop2 = (x, y1 + self.car_width)
        elif np.abs(x - x2) < self.car_width and y >= y2 and y <= y1:
            stop2 = (x2 + self.car_width, y)
        elif np.abs(y - y2) < self.car_width and x >= x1 and x <= x2:
            stop2 = (x, y2 - self.car_width)
        else:
            stop2 = (x, y)
        return stop2

    
    def _intersects(self, state1, state2, obst):

        A = self.s_to_coord[state1]
        B = self.s_to_coord[state2]

        margin = self.car_width / 2

        x3, y3 = obst[0]
        x4, y4 = obst[1]

        x3 -= margin
        y3 += margin
        x4 += margin
        y4 -= margin

        borders = [((x3, y3), (x4, y3)), ((x4, y3), (x4, y4)),
                   ((x4, y4), (x3, y4)), ((x3, y4), (x3, y3))]

        for border in borders:
            C = border[0]
            D = border[1]

            if (self._ccw(A,C,D) != self._ccw(B,C,D)) and (self._ccw(A,B,C) != self._ccw(A,B,D)):

                xA, yA = A
                xB, yB = B
                xC, yC = C
                xD, yD = D
                
                if (xA-xB == xC-xD) or (yA-yB == yC-yD):
                    continue
                else:
                    return True
        return False        
    
    def _get_borders(self):
        borders = []
        borders.append(((0, 0),(0, self.m)))
        borders.append(((0, self.m),(self.n, self.m)))
        borders.append(((self.n, self.m),(self.n, 0)))
        borders.append(((self.n, 0),(0, 0)))

        for obst in self.obstacles:
            x3, y3 = obst[0]
            x4, y4 = obst[1]

            borders.append(((x3, y3), (x4, y3)))
            borders.append(((x4, y3), (x4, y4)))
            borders.append(((x4, y4), (x3, y4)))
            borders.append(((x3, y4), (x3, y3)))
        
        return borders



    def _ccw(self, A, B, C):
        xA, yA = A
        xB, yB = B
        xC, yC = C
        return (yC-yA)*(xB-xA) > (yB-yA)*(xC-xA)
    
    def _get_reward(self, state, destination):
        reward = self.Q[state, destination]

        for obst in self.obstacles:
            if self._intersects(state, destination, obst):
                reward += self.obstacle_reward
        
        return reward
    

    def transition(self, destination):
        cur_state = self.visited[-1]
        reward = self._get_reward(cur_state, destination)
        
        self.visited.append(destination)

        done = len(self.visited) == self.n_stops

        return reward, done

    
    def reset(self):
        self.visited = []
        first_step = np.random.randint(self.n_stops)
        self.visited.append(first_step)
    

    def show(self):
        fig = plt.figure(figsize=(self.n, self.m))
        ax = fig.add_subplot(111)

        # plot contours
        ax.plot([0, self.n], [0, 0], c='b')
        ax.plot([0, self.n], [self.m, self.m], c='b')
        ax.plot([0, 0], [0, self.m], c='b')
        ax.plot([self.n, self.n], [0, self.m], c='b')

        # plot stops
        ax.scatter(*zip(*self.stops), c='r')

        # plot obstacles
        for obst in self.obstacles:
            x1, y1 = obst[0]
            x2, y2 = obst[1]
            ax.plot([x1, x2], [y1, y1], c='b')
            ax.plot([x1, x2], [y2, y2], c='b')
            ax.plot([x1, x1], [y1, y2], c='b')
            ax.plot([x2, x2], [y1, y2], c='b')

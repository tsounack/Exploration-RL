import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from environment import *
from car         import *


def run_episode(env, car, lr, disc):
    env.reset()
    car.reset_memory()
    s = env.visited[0]
    
    max_step = env.n_stops
    reward = 0

    for _ in range(max_step):
        car.remember_state(s)
        a = car.take_action(s)
        r, done = env.transition(a)

        car.train(s, a, r, lr, disc)

        reward += r
        s = a

        if done:
            break

    return env, car, reward

def run_episodes(env, car, lr, disc, nb_episodes = 10000):
    rewards = []
    frames  = []

    for i in tqdm(range(nb_episodes)):
        env, car, reward = run_episode(env, car, lr, disc)
        rewards.append(reward)


def show_path(env, car, fname = "path.png"):
        fig = plt.figure(figsize=(env.n, env.m))
        ax = fig.add_subplot(111)

        # plot contours
        ax.plot([0, env.n], [0, 0], c='b')
        ax.plot([0, env.n], [env.m, env.m], c='b')
        ax.plot([0, 0], [0, env.m], c='b')
        ax.plot([env.n, env.n], [0, env.m], c='b')

        # plot stops
        ax.scatter(*zip(*env.stops), c='r')

        # plot obstacles
        for obst in env.obstacles:
            x1, y1 = obst[0]
            x2, y2 = obst[1]
            ax.plot([x1, x2], [y1, y1], c='b')
            ax.plot([x1, x2], [y2, y2], c='b')
            ax.plot([x1, x1], [y1, y2], c='b')
            ax.plot([x2, x2], [y1, y2], c='b')

        trajectory = list_trajectory(env, car)

        for i in range(len(trajectory) - 1):
            x0, y0 = trajectory[i]
            x1, y1 = trajectory[i + 1]
            ax.arrow(x0, y0, (x1-x0)*0.95, (y1-y0)*0.95, width=0.1, head_width=0.3, color="r")
        
        x0, y0 = x1, y1
        x1, y1 = trajectory[0]
        ax.arrow(x0, y0, (x1-x0)*0.95, (y1-y0)*0.95, width=0.1, head_width=0.3, color="k")
        plt.savefig(fname)


def list_trajectory(env, car):
    trajectory = []

    for s in env.visited:
        x, y = env.s_to_coord[s]
        trajectory.append((x, y))

    return trajectory




lr = 0.1
disc = 0.9

env = Environment()
car = Car(env)

run_episodes(env, car, lr, disc, 1000)
show_path(env, car, "path.png")


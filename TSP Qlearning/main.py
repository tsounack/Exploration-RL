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

def run_episodes(env, car, lr, disc, nb_episodes = 1000):
    rewards = []
    frames  = []
    max_rew = -np.inf

    for i in tqdm(range(nb_episodes)):
        env, car, reward = run_episode(env, car, lr, disc)
        rewards.append(reward)
        if reward > max_rew:
            max_rew = reward
            env.best_path = env.visited
    print("max reward: ", max_rew)
    print("final epsilon value: ", car.eps)
    plot_rewards(rewards)
    

def plot_rewards(rewards):
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)
    ax.plot(rewards)
    ax.set_title("Rewards over the episodes, max reward = {}, last reward = {}".format(round(max(rewards),2), round(rewards[-1], 2)))
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Reward")
    plt.savefig("rewards.png")


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

    #for s in env.visited:
    for s in env.best_path:
        x, y = env.s_to_coord[s]
        trajectory.append((x, y))

    return trajectory




lr = 0.1
disc = 0.9

env = Environment()
car = Car(env)

run_episodes(env, car, lr, disc, 50)
show_path(env, car, "path.png")


import numpy as np

class Car:
    def __init__(self, env):
        self.env = env
        self.n_stops = self.env.n_stops
        self.n_actions = self.n_stops
        self.Q = self.env.Q
        self.eps = 100.0
        self.eps_min = 0.01
        self.eps_decay = 0.99

        self.reset_memory()
        
    
    def reset_memory(self):
        self.states_memory = []
    
    def remember_state(self, state):
        self.states_memory.append(state)
    
    def take_action(self, state):
        q = np.copy(self.Q[state,:])

        q[self.states_memory] = -np.inf

        if np.random.rand() < self.eps:
            a = np.random.choice([x for x in range(self.n_actions) if x not in self.states_memory])
        else:
            a = np.argmax(q)
        
        return a

    def train(self, s, a, r, lr, disc):
        self.Q[s, a] += lr * (r + disc * np.max(self.Q[a, :]) - self.Q[s, a])

        self.eps *= self.eps_decay




import numpy as np
import time

from sarsalambda import SarsaLambda

grid = SarsaLambda(n = 60,
                   m = 35,
                   discount = 0.9,
                   learning = 0.9,
                   decay = 0.1,
                   epsilon = 0.3,
                   nb_events = 10000000,
                   initial_coord=(1, 1)
                   )
grid.simulate()
grid.show()

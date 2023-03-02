import numpy as np
import time

from sarsalambda import SarsaLambda

grid = SarsaLambda(n = 60,
                   m = 35,
                   discount = 0.8,
                   learning = 0.1,
                   decay = 0.5,
                   epsilon = 0.3,
                   nb_events = 1000000,
                   initial_coord=(1, 1)
                   )
grid.simulate()
grid.show()

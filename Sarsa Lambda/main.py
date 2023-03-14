from sarsalambda import SarsaLambda

grid = SarsaLambda(n = 60,
                   m = 35,
                   discount = 0.8,
                   learning = 0.1,
                   decay = 0.5,
                   epsilon = 0.3,
                   nb_events = 100,
                   initial_coord=(1, 1)
                   )
grid.run_simulations(100000)
grid.show()

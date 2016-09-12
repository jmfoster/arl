blocks = 1000
iterations = 40
workers = 40
runID = 17

addpath('Cache')
results = Game.main(blocks, iterations, workers, runID)

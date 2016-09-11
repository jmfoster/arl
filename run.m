blocks = 100
iterations = 32
workers = 32
runID = 7

addpath('Cache')
results = Game.main(blocks, iterations, workers, runID)

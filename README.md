prisoner_game.py runs the simulation. It lets you play matches between strategies, keeps track of rounds, and calculates total scores. You use it to see how well a strategy performs against a set of opponents.

strats.py defines example strategies like Always Cooperate, Always Defect, Tit-for-Tat, and also includes helper functions to generate random strategies and display them. You use it to get opponents for testing or to start from a base strategy.

To create the algorithm modules, you write a function that generates candidate strategies, tests them using prisoner_game.py against a set of opponents (from strats.py), and keeps track of the best scoring strategies. The functions should return the best strategy, its score, and any history you want to track. 

The algorithms will be called from experiments.py to run multiple trials and compare them against eachother.

The algorithms will be stored in the /algorithms directory. I've created an example algorithm for reference that uses random search optimization.

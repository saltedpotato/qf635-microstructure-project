from arch.bootstrap.multiple_comparison import *

class WhiteRealityCheck:
    def __init__(self, strategies, benchmark):
        """
        Inputs should be returns (not losses).
        - strategies: shape (T, N)
        - benchmark: shape (T,)
        """
        self.strategy_returns = strategies
        self.benchmark_returns = benchmark
        
    
    def run(self, reps=1000, block_size=None):
        """
        Run the White Reality Check using block bootstrap.

        Parameters:
        iterations: int, number of bootstrap iterations
        block_size: int, block size for dependent data
        """
        # Flipped the order of inputs as SPA function evaluates based on losses instead of returns. 
        # For more information, please read documentation: https://arch.readthedocs.io/en/stable/multiple-comparison/multiple-comparison_examples.html

        rc = SPA(benchmark=self.strategy_returns, models=self.benchmark_returns, reps=reps, block_size=block_size, seed=123)
        rc.compute()
        
        # Results
        print("White Reality Check Results:")
        print(f"p-value: {rc.pvalues}")
    
    def superior_strategies(self, size=0.05, reps=1000, block_size=None):
        stepm = StepM(benchmark=self.strategy_returns, models=self.benchmark_returns, size=size, reps=reps, block_size=block_size, seed=123)
        stepm.compute()
        print("Superior strategy indices:")
        print(stepm.superior_models)
        return stepm.superior_models

########### For Testing ###########
# import numpy as np
# import pandas as pd
# from arch.bootstrap.multiple_comparison import *

# # Simulate data: 250 days, 3 strategies
# np.random.seed(42)
# n = 250
# m = 3

# # Simulated benchmark and strategies
# benchmark = np.random.normal(0.001, 0.01, n)
# strategies = pd.DataFrame(
#     np.random.normal(0.005, 0.01, (n, 4)),
#     columns=["strat1", "strat2", "strat3", "strat4"],
# )

# wrc = WhiteRealityCheck(strategies=strategies, benchmark=benchmark)
# wrc.run()
# wrc.superior_strategies()
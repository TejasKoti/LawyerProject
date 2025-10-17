# -*- coding: utf-8 -*-

# Install & Imports
!pip install pymoo==0.5.0

import pandas as pd
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_termination
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# Load Dataset
file_path = "Dummy_Lawyers.csv"
df = pd.read_csv(file_path)

# Define Optimization Problem
class LawyerSelectionProblem(Problem):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        super().__init__(n_var=len(df), n_obj=4, n_constr=0, xl=0.0, xu=1.0)

    def _evaluate(self, X, out, *args, **kwargs):
        objs = []
        for weights in X:
            weights = np.clip(weights, 0, 1)
            weights /= np.sum(weights) + 1e-6  # normalize weights

            profile = (self.df[[
                "Years of active experience",
                "No of favoured settlements",
                "Client satisfaction (out of 10)",
                "Price"
            ]].T @ weights).values

            objs.append([
                -profile[0],  # maximize experience
                -profile[1],  # maximize settlements
                -profile[2],  # maximize satisfaction
                profile[3]    # minimize price
            ])
        out["F"] = np.array(objs)

# Run NSGA-2
def recommend_nsga2(domain, budget, seed=1, generations=100, pop_size=50):
    # Filter dataset
    filtered_df = df[(df["Domain"] == domain) & (df["Price"] <= budget)].reset_index(drop=True)
    if filtered_df.empty:
        print("No matching lawyers found for given filters.")
        return pd.DataFrame()

    # Define NSGA-II problem
    problem = LawyerSelectionProblem(filtered_df)
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    termination = get_termination("n_gen", generations)

    # Run optimization
    res = minimize(problem, algorithm, termination, seed=seed, verbose=False)

    # Get Pareto-optimal indices
    pareto_idx = NonDominatedSorting().do(res.F, only_non_dominated_front=True)

    # Collect lawyers corresponding to Pareto front
    pareto_lawyers = []
    for i in pareto_idx:
        weights = np.clip(res.X[i], 0, 1)
        weights /= np.sum(weights) + 1e-6
        idx = np.argmax(weights)
        lawyer = filtered_df.iloc[idx].to_dict()
        lawyer["Objective Vector"] = res.F[i]
        lawyer["Price Objective"] = res.F[i][3]
        pareto_lawyers.append(lawyer)

    # Create DataFrame
    result_df = pd.DataFrame(pareto_lawyers).drop_duplicates(subset=["Name"])
    result_df = result_df.sort_values(by="Price Objective").reset_index(drop=True)

    # Show relevant columns
    return result_df[[
        "Name", "Domain", "Price", "Years of active experience",
        "No of favoured settlements", "Client satisfaction (out of 10)", "Price Objective"
    ]]

# Example 1
print("=== Civil Law, Budget 20000 ===")
display(recommend_nsga2("Civil Law", 20000))

# Example 2
print("=== Tax Law, Budget 30000 ===")
display(recommend_nsga2("Tax Law", 30000))

# Example 3
print("=== Corporate Law, Budget 50000 ===")
display(recommend_nsga2("Corporate Law", 50000))
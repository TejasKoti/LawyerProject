# -*- coding: utf-8 -*-

import pandas as pd
import random

# === Load the dataset using file path ===
file_path = "/content/lawyers_with_estimated_price (1).csv"  # Replace with your actual path
df = pd.read_csv(file_path)

# === Genetic Algorithm Parameters ===
POPULATION_SIZE = 30
GENERATIONS = 50
MUTATION_RATE = 0.1
TOURNAMENT_SIZE = 5

# === Fitness Function ===
def compute_fitness(individual, df, budget):
    exp_norm = (individual["Years of active experience"] - df["Years of active experience"].min()) / \
               (df["Years of active experience"].max() - df["Years of active experience"].min() + 1e-6)
    fav_norm = (individual["No of favoured settlements"] - df["No of favoured settlements"].min()) / \
               (df["No of favoured settlements"].max() - df["No of favoured settlements"].min() + 1e-6)
    sat_norm = (individual["Client satisfaction (out of 10)"] - df["Client satisfaction (out of 10)"].min()) / \
               (df["Client satisfaction (out of 10)"].max() - df["Client satisfaction (out of 10)"].min() + 1e-6)
    price_prox_norm = 1 - ((budget - individual["Price"]) / (budget + 1e-6)) if individual["Price"] <= budget else 0

    fitness = 0.3 * exp_norm + 0.3 * fav_norm + 0.2 * sat_norm + 0.2 * price_prox_norm
    return fitness

# === Initialize Population ===
def initialize_population(candidates):
    unique_candidates = candidates.drop_duplicates(subset=["Name"])
    sampled = unique_candidates.sample(n=min(POPULATION_SIZE, len(unique_candidates)), replace=False)
    return list(sampled.to_dict(orient="records"))

# === Tournament Selection ===
def selection(population, df, budget):
    tournament = random.sample(population, min(TOURNAMENT_SIZE, len(population)))
    return max(tournament, key=lambda ind: compute_fitness(ind, df, budget))

# === Crossover ===
def crossover(parent1, parent2):
    child = {}
    for key in parent1:
        child[key] = parent1[key] if random.random() < 0.5 else parent2[key]
    return child

# === Mutation ===
def mutate(individual, candidates):
    if random.random() < MUTATION_RATE:
        mutation_candidate = random.choice(candidates.to_dict(orient="records"))
        mutation_field = random.choice([
            "Years of active experience",
            "No of favoured settlements",
            "Client satisfaction (out of 10)",
            "Price"
        ])
        individual[mutation_field] = mutation_candidate[mutation_field]
    return individual

# === Run the GA and Rank Results ===
def run_genetic_algorithm(domain, budget):
    # Filter by domain and budget
    candidates = df[(df["Domain"] == domain) & (df["Price"] <= budget)].copy()
    if candidates.empty:
        print("No matching candidates found.")
        return pd.DataFrame()

    # Initialize population (optional: use in further GA evolution)
    population = initialize_population(candidates)

    for _ in range(GENERATIONS):
        new_population = []
        while len(new_population) < POPULATION_SIZE:
            parent1 = selection(population, df, budget)
            parent2 = selection(population, df, budget)
            child = crossover(parent1, parent2)
            child = mutate(child, candidates)
            new_population.append(child)
        population = new_population

    # Evaluate all valid candidates for final display
    all_candidates = candidates.copy()
    all_candidates["Fitness Score"] = all_candidates.apply(lambda row: compute_fitness(row, df, budget), axis=1)

    # Sort by Fitness (descending), then Price (ascending)
    sorted_candidates = all_candidates.sort_values(
        by=["Fitness Score", "Price"], ascending=[False, True]
    ).drop_duplicates(subset="Name").reset_index(drop=True)

    return sorted_candidates

# === Run Demo ===
domain_input = "Tax Law"    # 👈 Change this
budget_input = 30000        # 👈 And this

top_lawyers = run_genetic_algorithm(domain_input, budget_input)

# === Display Results ===
top_lawyers[[
    "Name", "Domain", "Price", "Years of active experience",
    "No of favoured settlements", "Client satisfaction (out of 10)", "Fitness Score"
]]

!pip install pymoo

!pip install pymoo==0.5.0

!pip install pymoo --upgrade

import pandas as pd
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

# === Load your dataset ===
file_path = "/content/lawyers_with_estimated_price.csv"
df = pd.read_csv(file_path)

# === Filter dataset ===
domain_input = "Tax Law"
budget_input = 30000
filtered_df = df[(df["Domain"] == domain_input) & (df["Price"] <= budget_input)].reset_index(drop=True)

# === Define NSGA-II Optimization Problem ===
class LawyerSelectionProblem(Problem):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        super().__init__(n_var=len(df), n_obj=4, n_constr=0, xl=0.0, xu=1.0)

    def _evaluate(self, X, out, *args, **kwargs):
        objs = []
        for weights in X:
            weights = np.clip(weights, 0, 1)
            weights /= np.sum(weights) + 1e-6  # normalize weights

            # Weighted profile
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

# === Initialize and run NSGA-II ===
problem = LawyerSelectionProblem(filtered_df)

algorithm = NSGA2(
    pop_size=50,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)

termination = get_termination("n_gen", 100)

res = minimize(problem, algorithm, termination, seed=1, verbose=False)

# === Extract top individuals ===
top_lawyers = []
for i, weights in enumerate(res.X):
    weights = np.clip(weights, 0, 1)
    weights /= np.sum(weights) + 1e-6
    idx = np.argmax(weights)
    lawyer = filtered_df.iloc[idx].to_dict()
    lawyer["Objective Vector"] = res.F[i]
    lawyer["Price Objective"] = res.F[i][3]  # For sorting later
    top_lawyers.append(lawyer)

# === Display top results ===
result_df = pd.DataFrame(top_lawyers).drop_duplicates(subset=["Name"])
result_df = result_df.sort_values(by="Price Objective").reset_index(drop=True)

# === Show Relevant Columns ===
result_df[[
    "Name", "Domain", "Price", "Years of active experience",
    "No of favoured settlements", "Client satisfaction (out of 10)", "Price Objective"
]]

import pandas as pd
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

# === Load your dataset ===
file_path = "/content/lawyers_with_estimated_price.csv"
df = pd.read_csv(file_path)

# === Filter dataset ===
domain_input = "Tax Law"       # 👈 Change domain as needed
budget_input = 30000           # 👈 Set your budget limit
filtered_df = df[(df["Domain"] == domain_input) & (df["Price"] <= budget_input)].reset_index(drop=True)

# === Define NSGA-II Optimization Problem ===
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

# === Initialize and run NSGA-II ===
problem = LawyerSelectionProblem(filtered_df)

algorithm = NSGA2(
    pop_size=50,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)

termination = get_termination("n_gen", 100)

res = minimize(problem, algorithm, termination, seed=1, verbose=False)

# === Extract top individuals ===
top_lawyers = []
for i, weights in enumerate(res.X):
    weights = np.clip(weights, 0, 1)
    weights /= np.sum(weights) + 1e-6
    idx = np.argmax(weights)
    lawyer = filtered_df.iloc[idx].to_dict()
    lawyer["Objective Vector"] = res.F[i]
    lawyer["Price Objective"] = res.F[i][3]  # For sorting
    top_lawyers.append(lawyer)

# === Display all attributes ===
result_df = pd.DataFrame(top_lawyers).drop_duplicates(subset=["Name"])
result_df = result_df.sort_values(by="Price Objective").reset_index(drop=True)

# Show all columns in output
pd.set_option("display.max_columns", None)
display(result_df)

import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# Get the objective vectors (negated values were used for maximization)
F = res.F

# Extract two objectives to plot (e.g., experience vs. satisfaction)
x = -F[:, 0]  # experience (was negated to maximize)
y = -F[:, 2]  # satisfaction (was negated to maximize)

# Find Pareto-optimal solutions (non-dominated front)
nd_front = NonDominatedSorting().do(F, only_non_dominated_front=True)

# Plot all solutions
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='blue', s=10, label='Solution')

# Plot Pareto-optimal solutions
plt.scatter(x[nd_front], y[nd_front], color='red', s=40, marker='v', label='Pareto Solution')

plt.xlabel("Experience Score")
plt.ylabel("Client Satisfaction Score")
plt.legend()
plt.grid(True)
plt.title("NSGA-II Lawyer Selection: Experience vs Satisfaction")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# === Blue Dots: Raw Lawyers Data (within budget & domain) ===
x_blue = filtered_df["Price"].values
y_blue = filtered_df["Client satisfaction (out of 10)"].values

# === Red Dots: NSGA-II Pareto-optimal solutions ===
F = res.F
nd_indices = NonDominatedSorting().do(F, only_non_dominated_front=True)

# Extract price (minimize) and satisfaction (maximize, so we invert)
x_pareto = F[nd_indices, 3]       # Price (no negation — already to minimize)
y_pareto = -F[nd_indices, 2]      # Satisfaction (negated in obj, so invert back)

# Sort by price for smooth Pareto front line
sorted_idx = np.argsort(x_pareto)
x_pareto_sorted = x_pareto[sorted_idx]
y_pareto_sorted = y_pareto[sorted_idx]

# === Plotting ===
plt.figure(figsize=(8, 5))

# Raw filtered lawyers
plt.scatter(x_blue, y_blue, s=10, c='blue', label='Solution')

# NSGA-II Pareto front
plt.scatter(x_pareto_sorted, y_pareto_sorted, s=40, c='red', marker='v', label='Pareto Solution')

plt.xlabel("Price")
plt.ylabel("Client Satisfaction Score")
plt.title("NSGA-II Lawyer Selection: Price vs Satisfaction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
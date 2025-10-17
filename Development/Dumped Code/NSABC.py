# -*- coding: utf-8 -*-

# Cell 1: Imports & Data Load
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt

# Load dataset
df = pd.read_csv('./lawyer_dataset.csv')

# Prepare objectives matrix: [price, 10 - satisfaction]
data = df[['Price', 'Client satisfaction (out of 10)']].values
objectives = np.column_stack((data[:,0], 10 - data[:,1]))

import numpy as np
import random

class NSABC:
    def __init__(self, obj_vals, swarm_size=50, archive_size=50, max_iter=200, limit=20):
        self.n_solutions = obj_vals.shape[0]
        self.obj = obj_vals
        self.NP = swarm_size
        self.archive_size = archive_size
        self.MIC = max_iter
        self.limit = limit  # scout limit
        # initial swarm: random unique indices
        self.X = random.sample(range(self.n_solutions), self.NP)
        # trial counters for scout phase
        self.trial = {idx: 0 for idx in self.X}
        self.archive = []

    def _non_dominated_sort(self, idxs):
        objs = self.obj[idxs]
        n = len(idxs)
        S = [[] for _ in range(n)]
        front = [[]]
        domination_count = [0]*n

        for p in range(n):
            for q in range(n):
                if all(objs[p] <= objs[q]) and any(objs[p] < objs[q]):
                    S[p].append(q)
                elif all(objs[q] <= objs[p]) and any(objs[q] < objs[p]):
                    domination_count[p] += 1
            if domination_count[p] == 0:
                front[0].append(p)

        i = 0
        while front[i]:
            next_front = []
            for p in front[i]:
                for q in S[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        next_front.append(q)
            i += 1
            front.append(next_front)

        return [[idxs[i] for i in fr] for fr in front[:-1]]

    def _crowding_distance(self, front):
        l = len(front)
        if l == 0:
            return []
        dist = {idx: 0.0 for idx in front}
        for m in range(self.obj.shape[1]):
            sorted_list = sorted(front, key=lambda idx: self.obj[idx, m])
            dist[sorted_list[0]] = np.inf
            dist[sorted_list[-1]] = np.inf
            min_val = self.obj[sorted_list[0], m]
            max_val = self.obj[sorted_list[-1], m]
            if max_val == min_val:
                continue
            for i in range(1, l-1):
                prev_val = self.obj[sorted_list[i-1], m]
                next_val = self.obj[sorted_list[i+1], m]
                dist[sorted_list[i]] += (next_val - prev_val) / (max_val - min_val)
        return [dist[idx] for idx in front]

    def _update_archive(self, swarm_idxs):
        merged = list(set(self.archive + swarm_idxs))
        fronts = self._non_dominated_sort(merged)
        new_archive = []
        for front in fronts:
            if len(new_archive) + len(front) <= self.archive_size:
                new_archive.extend(front)
            else:
                cds = self._crowding_distance(front)
                ranked = sorted(zip(front, cds), key=lambda x: -x[1])
                need = self.archive_size - len(new_archive)
                new_archive.extend([idx for idx, _ in ranked[:need]])
                break
        self.archive = new_archive

    def _mutate_solution(self, idx):
        return random.randrange(self.n_solutions)

    def _employee_phase(self):
        new_X = []
        for idx in self.X:
            v = self._mutate_solution(idx)
            u = self._mutate_solution(idx)
            cands = [idx, v, u]
            fronts = self._non_dominated_sort(cands)
            chosen = random.choice(fronts[0])
            new_X.append(chosen)
            self.trial[idx] = 0 if chosen != idx else self.trial.get(idx,0) + 1
        self.X = new_X

    def _onlooker_phase(self):
        # compute crowding distances
        fronts = self._non_dominated_sort(self.X)
        distances = {}
        for front in fronts:
            cds = self._crowding_distance(front)
            for idx, cd in zip(front, cds):
                distances[idx] = cd
        # identify any infinite-distance elites
        inf_idxs = [idx for idx, d in distances.items() if np.isinf(d)]
        idx_list = list(self.X)
        # generate new swarm
        new_X = []
        for _ in range(self.NP):
            if inf_idxs:
                sel = random.choice(inf_idxs)
            else:
                # finite crowding: normalized weights
                dist_list = [distances.get(idx, 0.0) for idx in idx_list]
                total = sum(dist_list)
                if total > 0:
                    probs = [d/total for d in dist_list]
                else:
                    probs = [1/len(idx_list)]*len(idx_list)
                sel = random.choices(idx_list, weights=probs, k=1)[0]
            m = self._mutate_solution(sel)
            if all(self.obj[m] <= self.obj[sel]) and any(self.obj[m] < self.obj[sel]):
                new_X.append(m)
                self.trial[sel] = 0
            else:
                new_X.append(sel)
                self.trial[sel] = self.trial.get(sel,0) + 1
        self.X = new_X

    def _scout_phase(self):
        for i, idx in enumerate(self.X):
            if self.trial.get(idx, 0) > self.limit:
                new_idx = random.randrange(self.n_solutions)
                self.X[i] = new_idx
                self.trial[new_idx] = 0
                self.trial[idx] = 0

    def run(self):
        self._update_archive(self.X)
        for _ in range(self.MIC):
            self._employee_phase()
            self._onlooker_phase()
            self._scout_phase()
            self._update_archive(self.X)
        return self.archive

# Cell 3: Execute NSABC
nsabc = NSABC(objectives,
              swarm_size=100,
              archive_size=50,
              max_iter=300)
pareto_idxs = nsabc.run()

# Cell 4: Visualizing without annotations
import matplotlib.pyplot as plt

# All lawyers
prices = df['Price'].values
sats   = df['Client satisfaction (out of 10)'].values

# Pareto front
front_prices = prices[pareto_idxs]
front_sats   = sats[pareto_idxs]

plt.figure(figsize=(10,6))
plt.scatter(prices, sats, s=20, alpha=0.5, label='All lawyers')
plt.scatter(front_prices, front_sats, s=80, marker='*',
            color='orange', label='Pareto optimal')
plt.xlabel('Price (₹)')
plt.ylabel('Client satisfaction (out of 10)')
plt.title('Lawyer Recommendation: Price vs Satisfaction Trade-off')
plt.legend()
plt.show()

# Cell 5: Listing Top Recommendations
recommended = df.iloc[pareto_idxs].copy()
recommended['Tradeoff score'] = recommended['Price'] * (10 - recommended['Client satisfaction (out of 10)'])
recommended = recommended.sort_values('Tradeoff score')
recommended.reset_index(drop=True, inplace=True)
recommended.head(10)

# Cell 7: Parallel coordinates for the archive
from pandas.plotting import parallel_coordinates

pareto_df = df.iloc[pareto_idxs][[
    'Price', 'Years of active experience', 'No of cases fought',
    'No of cases settled','Client satisfaction (out of 10)', 'Age'
]].reset_index(drop=True)
pareto_df['ID'] = pareto_df.index.astype(str)
plt.figure(figsize=(10,6))
parallel_coordinates(pareto_df, 'ID', alpha=0.5)
plt.title('Parallel Coordinates Plot of Pareto Lawyers')
plt.legend([],[])  # hide legend if too many lines
plt.show()

# Cell 8: Histogram of trade-off score
scores = df['Price'] * (10 - df['Client satisfaction (out of 10)'])
pareto_scores = scores.iloc[pareto_idxs]

plt.figure(figsize=(8,4))
plt.hist(scores, bins=30, alpha=0.5, label='All lawyers')
plt.hist(pareto_scores, bins=30, alpha=0.8, label='Pareto lawyers')
plt.xlabel('Trade-off Score (Price × (10−Sat))')
plt.ylabel('Count')
plt.title('Distribution of Trade-off Scores')
plt.legend()
plt.show()

# Cell 6: Hypervolume over time (requires simple HV function)
def hypervolume(front, ref):
    hv = 0.0
    # e.g. sort by objective 0, integrate under curve in 2D
    pts = sorted(front, key=lambda x: x[0])
    prev_f2 = ref[1]
    for f1, f2 in pts:
        width = ref[0] - f1
        height = prev_f2 - f2
        hv += width * height
        prev_f2 = f2
    return hv

ref_point = [max(objectives[:,0])*1.1, max(objectives[:,1])*1.1]
hv_values = []
nsabc = NSABC(objectives, swarm_size=100, archive_size=50, max_iter=300)
for _ in range(nsabc.MIC):
    nsabc._employee_phase()
    nsabc._onlooker_phase()
    nsabc._scout_phase()
    nsabc._update_archive(nsabc.X)
    front = objectives[nsabc.archive]
    hv_values.append(hypervolume(front, ref_point))

plt.figure(figsize=(8,4))
plt.plot(hv_values)
plt.xlabel('Iteration')
plt.ylabel('Hypervolume')
plt.title('Hypervolume Progression')
plt.grid(True)
plt.show()

# Cell 5: Track archive size over time
archive_sizes = []
nsabc = NSABC(objectives, swarm_size=100, archive_size=50, max_iter=300)
for it in range(nsabc.MIC):
    nsabc._employee_phase()
    nsabc._onlooker_phase()
    nsabc._scout_phase()
    nsabc._update_archive(nsabc.X)
    archive_sizes.append(len(nsabc.archive))

plt.figure(figsize=(8,4))
plt.plot(range(1, nsabc.MIC+1), archive_sizes, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Archive (Pareto) Size')
plt.title('Convergence of Pareto Front Size')
plt.grid(True)
plt.show()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 10:18:04 2020

@author: jessicaruijsch
"""

from pymoo.util.misc import stack

def func_pf(flatten=True, **kwargs):
        f1_a = np.linspace(0.1**2, 0.4**2, 100)
        f2_a = (np.sqrt(f1_a) - 1)**2

        f1_b = np.linspace(0.6**2, 0.9**2, 100)
        f2_b = (np.sqrt(f1_b) - 1)**2

        a, b = np.column_stack([f1_a, f2_a]), np.column_stack([f1_b, f2_b])
        return stack(a, b, flatten=flatten)

def func_ps(flatten=True, **kwargs):
        x1_a = np.linspace(0.1, 0.4, 50)
        x1_b = np.linspace(0.6, 0.9, 50)
        x2 = np.zeros(50)

        a, b = np.column_stack([x1_a, x2]), np.column_stack([x1_b, x2])
        return stack(a,b, flatten=flatten)

import numpy as np
from pymoo.core.problem import Problem  # Before from pymoo.model.problem import Problem


# --------------------------------------------------
# define the problem
# --------------------------------------------------
class MyProblem(Problem):
    
    # by calling the super() function the problem properties are initialized 
    def __init__(self):
        super().__init__(n_var=2,                   # nr of variables
                         n_obj=2,                   # nr of objectives
                         n_constr=2,                # nr of constrains
                         xl=np.array([-2,-2]),      # lower boundaries
                         xu=np.array([2,2]))        # upper boundaries

    # the _evaluate function needs to be overwritten from the superclass 
    # the method takes two-dimensional NumPy array x with n rows and n columns as input
    # each row represents an individual and each column an optimization variable 
    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:,0]**2 + x[:,1]**2
        f2 = (x[:,0]-1)**2 + x[:,1]**2

        g1 = 2*(x[:, 0]-0.1) * (x[:, 0]-0.9) / 0.18
        g2 = - 20*(x[:, 0]-0.4) * (x[:, 0]-0.6) / 4.8

        # after doing the necessary calculations, 
        # the objective values have to be added to the dictionary out
        # with the key F and the constrains with key G 
        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2])


    # calculate pareto front
    def _calc_pareto_front(self, *args, **kwargs):
        return func_pf(**kwargs)

    def _calc_pareto_set(self, *args, **kwargs):
        return func_ps(**kwargs)

problem = MyProblem()


# --------------------------------------------------
# initialize the algorithm
# --------------------------------------------------
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

algorithm = NSGA2(
    pop_size=40,
    n_offsprings=10,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)

# --------------------------------------------------
# define the termination criterion
# --------------------------------------------------
from pymoo.termination import get_termination

termination = get_termination("n_gen", 40)

# --------------------------------------------------
# optimize
# --------------------------------------------------
from pymoo.optimize import minimize
 
res = minimize(problem,
               algorithm,
               termination,
               seed=None,
               #pf=problem.pareto_front(use_cache=False),
               save_history=True,
               verbose=True)

# --------------------------------------------------
# visualize
# --------------------------------------------------
import matplotlib.pyplot as plt

# Plot the design space 
f1, ax1 = plt.subplots(1)
ax1.scatter(res.X[:,0], res.X[:,1], s=30, fc='none', ec='r')
ax1.set_title('design space')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)
f1.savefig('design_space.png')

# Plot the objective space
f2, ax2 = plt.subplots(1)
ax2.scatter(res.F[:,0], res.F[:,1], s=30, fc='none', ec='k')
ax2.set_title('objective space')
ax2.set_xlabel('f1')
ax2.set_ylabel('f2') 
f2.savefig('objective_space.png')


# The same could be done with functions built-in to pymoo
"""
from pymoo.visualization.scatter import Scatter

# get the pareto-set and pareto-front for plotting
ps = problem.pareto_set(use_cache=False, flatten=False)
pf = problem.pareto_front(use_cache=False, flatten=False)

# Design Space
plot = Scatter(title = "Design Space", axis_labels="x")
plot.add(res.X, s=30, facecolors='none', edgecolors='r')
if ps is not None:
    plot.add(ps, plot_type="line", color="black", alpha=0.7)
plot.do()
plot.apply(lambda ax: ax.set_xlim(-0.5, 1.5))
plot.apply(lambda ax: ax.set_ylim(-2, 2))
plot.show()

# Objective Space
plot2 = Scatter(title = "Objective Space")
plot2.add(res.F)
if pf is not None:
    plot2.add(pf, plot_type="line", color="black", alpha=0.7)
plot2.show()
"""

"""
res.X design space values are
res.F objective spaces values
res.G constraint values
res.CV aggregated constraint violation
res.algorithm algorithm object
res.pop final population object
res.history history of algorithm object. (only if save_history has been enabled during the algorithm initialization)
res.time the time required to run the algorithm
"""

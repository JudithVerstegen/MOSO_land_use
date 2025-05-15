#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 09:58:17 2020

@author: jessicaruijsch

# --------------------------------------------------
# main script
# --------------------------------------------------  

Rules:
- A raster cell must have exactly one class (original data between 1 and 13)
- Water, urban area and no data (11, 12, 15) cannot be changed 
- Cerrado, forest, secondary vegetation (1, 3, 13), can be turned into agriculture, but not the other way around
- Combine different soy classes (5, 6, 7, 8, 9) into one class of soy (5)
- Cotton, pasture, soy, sugarcane (2, 4, 5, 10) can be interchanged and replace forest

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import pickle
from initial_population import initialize_spatial

default_directory = "C:/Users/verst137/OneDrive - Universiteit Utrecht/Documents/scripts/MOSO_land_use/input_data"



# --------------------------------------------------
# read data
# --------------------------------------------------

cell_area = 2.5 * 2.5 # in hectares

with open(default_directory + "/Objectives/sugarcane_potential_yield_example.pkl", 'rb') as output:
    sugarcane_pot_yield =  pickle.load(output)

with open(default_directory + "/Objectives/soy_potential_yield_example.pkl", 'rb') as output:
    soy_pot_yield =  pickle.load(output)

with open(default_directory + "/Objectives/cotton_potential_yield_example.pkl", 'rb') as output:
    cotton_pot_yield =  pickle.load(output)

with open(default_directory + "/Objectives/pasture_potential_yield_example.pkl", 'rb') as output:
    pasture_pot_yield =  pickle.load(output)

# --------------------------------------------------
# define the problem
# --------------------------------------------------

from pymoo.util.misc import stack
from pymoo.core.problem import Problem
from calculate_objectives import calculate_tot_yield, calculate_above_ground_biomass, calculate_landuse_patches


class MyProblem(Problem):
    
    # by calling the super() function the problem properties are initialized 
    def __init__(self):
        super().__init__(n_var=100,                   # nr of variables
                         n_obj=2,                   # nr of objectives
                         n_constr=0,                # nr of constrains
                         xl= 0,                   # lower boundaries
                         xu= 1)                  # upper boundaries

    # the _evaluate function needs to be overwritten from the superclass 
    # the method takes two-dimensional NumPy array x with n rows and n columns as input
    # each row represents an individual and each column an optimization variable 
    def _evaluate(self, X, out, *args, **kwargs):
        
        
        f1 = -calculate_tot_yield(X[:], sugarcane_pot_yield,soy_pot_yield,cotton_pot_yield,pasture_pot_yield,cell_area)
        f2 = -calculate_above_ground_biomass(X[:],cell_area)
        # f3 = calculate_landuse_patches(x)

        # after doing the necessary calculations, 
        # the objective values have to be added to the dictionary out
        # with the key F and the constrains with key G 
        out["F"] = np.column_stack([f1, f2])

Problem_def = MyProblem()

# --------------------------------------------------
# initialize the algorithm
# --------------------------------------------------

from pymoo.algorithms.moo.nsga2 import NSGA2
from spatial_sampling import SpatialSampling
from spatial_crossover import SpatialOnePointCrossover
from spatial_mutation import SpatialNPointMutation

     
algorithm = NSGA2(
    pop_size=70,
    n_offsprings=10,
    sampling = SpatialSampling(default_directory),
    crossover = SpatialOnePointCrossover(n_points=3),
    mutation = SpatialNPointMutation(prob = 0.001, point_mutation_probability = 0.015),
    eliminate_duplicates=False
    )
# algorithm.eleminate_duplicates = ElementwiseDuplicateElimination

# --------------------------------------------------
# define the termination criterion
# --------------------------------------------------

from pymoo.termination import get_termination

termination = get_termination("n_gen", 500)

# --------------------------------------------------
# optimize
# --------------------------------------------------

from pymoo.optimize import minimize
 
res = minimize(Problem_def,
               algorithm,
               termination,
               seed=None,
               pf=Problem_def.pareto_front(use_cache=False),
               save_history=True,
               verbose=True)

# --------------------------------------------------
# visualize pareto front
# --------------------------------------------------

from pymoo.visualization.scatter import Scatter

plt.scatter(-res.F[:,0],-res.F[:,1])
plt.title("Objective Space")
plt.xlabel('Total yield [tonnes]')
plt.ylabel('Above ground biomass [tonnes]')
plt.savefig(default_directory+"/outputs/objective_space.png",dpi=150)
plt.show()

# --------------------------------------------------
# visualize land use maps
# --------------------------------------------------

# np.argmax(-res.F[:,0], axis=0) --> optimized for f1
# np.argmax(-res.F[:,1], axis=0) --> optimized for f2

cmap = ListedColormap(["#10773e","#b3cc33", "#0cf8c1", "#a4507d","#877712",
                      "#be94e8","#eeefce","#1b5ee4","#614040","#00000000"])
legend_landuse = [mpatches.Patch(color="#10773e",label = 'Forest'),
          mpatches.Patch(color="#b3cc33",label = 'Cerrado'),
          mpatches.Patch(color="#0cf8c1",label = 'Secondary veg.'),
          mpatches.Patch(color="#a4507d",label = 'Soy'),
          mpatches.Patch(color="#877712",label = 'Sugarcane'),
          mpatches.Patch(color="#be94e8",label = 'Fallow/cotton'),
          mpatches.Patch(color="#eeefce",label = 'Pasture'),
          mpatches.Patch(color="#1b5ee4",label = 'Water'),
          mpatches.Patch(color="#614040",label = 'Urban'),
          mpatches.Patch(color="#00000000",label = 'No data')]

landuse_max_yield = res.X[np.argmax(-res.F[:,0], axis=0)]
landuse_max_biomass = res.X[np.argmax(-res.F[:,1], axis=0)]

plt.imshow(landuse_max_yield,interpolation='None',cmap=cmap,vmin=0.5,vmax=10.5)
plt.legend(handles=legend_landuse,bbox_to_anchor=(1.05, 1), loc=2, 
           borderaxespad=0.)
plt.title('Landuse map maximized total yield')
plt.xlabel('Column #')
plt.ylabel('Row #')
plt.savefig(default_directory+"/outputs/landuse_max_yield.png",dpi=150)
plt.show()

plt.imshow(landuse_max_biomass, interpolation='None',cmap=cmap,vmin=0.5,vmax=10.5)
plt.legend(handles=legend_landuse,bbox_to_anchor=(1.05, 1), loc=2, 
           borderaxespad=0.)
plt.title('Landuse map minimized CO2 emissions')
plt.xlabel('Column #')
plt.ylabel('Row #')
plt.savefig(default_directory+"/outputs/landuse_min_co2.png",dpi=150)
plt.show()

# --------------------------------------------------
# convergence
# --------------------------------------------------

# the objective space values in each generation
F = []

# iterate over the deepcopies of algorithms
for algorithm in res.history:
    # retrieve the optimum from the algorithm
    opt = algorithm.opt
    _F = opt.get("F")
    F.append(_F)

n_gen = np.array(range(1,len(F)+1))

    # --------------------------------------------------
    # maximum of objective values
    # --------------------------------------------------
    
# get maximum (extremes) of each generation for both objectives
obj_1 = []
obj_2 = []
for i in F:
    max_obj_1 = max(i[:,0])
    max_obj_2 = max(i[:,1])
    
    obj_1.append(max_obj_1)
    obj_2.append(max_obj_2)

# visualze the maximum objective 1 (total yield)
plt.plot(n_gen, -np.array(obj_1))
plt.title("Convergence")
plt.xlabel("Generations")
plt.ylabel("Maximum total yield [tonnes]")
plt.savefig(default_directory+"/outputs/max_tot_yield.png",dpi=150)
plt.show()

# visualze the maximum objective 2 (above ground biomass)
plt.plot(n_gen, -np.array(obj_2))
plt.title("Convergence")
plt.xlabel("Generations")
plt.ylabel("Above ground biomass [tonnes]")
plt.savefig(default_directory+"/outputs/max_biomass.png",dpi=150)
plt.show()
   
   
    # --------------------------------------------------
    # pareto front over generations
    # --------------------------------------------------

from pymoo.visualization.scatter import Scatter

for i in (0, 49, 99, 199, 299, 399, 499):
    plt.scatter(-F[i][:,0],-F[i][:,1])
plt.title("Objective Space")
plt.xlabel('Total yield [tonnes]')
plt.ylabel('Above ground biomass [tonnes]')
plt.legend(['gen 1','gen50','gen 100','gen 200','gen 300','gen 400','gen 500'])
plt.savefig(default_directory+"/outputs/objective_space_through_time.png",dpi=150)
plt.show()


#     # --------------------------------------------------
#     # hypervolume
#     # --------------------------------------------------

# import matplotlib.pyplot as plt
# from pymoo.indicators.hv.exact import ExactHypervolume  

# # make an array of the number of generations
# n_gen = np.array(range(1,len(F)+1))
# # set reference point
# ref_point = np.array([0.0, 0.0])
# # create the performance indicator object with reference point
# metric = ExactHypervolume(ref_point=ref_point)
# # calculate for each generation the HV metric
# hv = [metric.calc(f) for f in F]

# # visualze the convergence curve
# plt.plot(n_gen, hv, '-o', markersize=4, linewidth=2)
# plt.title("Convergence")
# plt.xlabel("Generations")
# plt.ylabel("Hypervolume")
# plt.ylim(0,1.5*10**11)
# plt.savefig(default_directory+"/outputs/hypervolume.png",dpi=150)
# plt.show()


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



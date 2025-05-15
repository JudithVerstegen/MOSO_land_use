#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 09:57:38 2020

@author: root

To do:
- include constrains
- 

"""
import numpy as np 
import random

from compute_genome import create_patch_ID_map
from pymoo.core.crossover import Crossover 
from pymoo.core.variable import Real, get
from pymoo.core.population import Population

#-------------------------------------------------------------------------------------  
# Spatial crossover
#-------------------------------------------------------------------------------------



class SpatialOnePointCrossover(Crossover):
    
    def __init__(self, n_points, **kwargs):
        super().__init__(2, 2, 1.0) # (n_parents,n_offsprings,probability)
        self.n_points = n_points #number of crossover points
        
    def do(self, problem, pop, parents=None, **kwargs):

        # if a parents with array with mating indices is provided -> transform the input first
        if parents is not None:
            pop = [pop[mating] for mating in parents]

        # get the dimensions necessary to create in and output
        n_parents, n_offsprings = self.n_parents, self.n_offsprings
        n_matings, n_var = len(pop), problem.n_var
        #print(n_parents)
        #print(n_matings)
        #print(n_var)

        # get the actual values from each of the parents
        X = np.swapaxes(np.array([[parent.get("X") for parent in mating] for mating in pop]), 0, 1)
        if self.vtype is not None:
            X = X.astype(self.vtype)

        # the array where the offsprings will be stored to
        Xp = np.empty(shape=(n_offsprings, n_matings, n_var, n_var), dtype=X.dtype)
        #print(Xp.shape)
        #print(Xp)

        # the probability of executing the crossover
        prob = get(self.prob, size=n_matings)

        # a boolean mask when crossover is actually executed
        cross = np.random.random(n_matings) < prob

        # the design space from the parents used for the crossover
        if np.any(cross):

            # we can not prefilter for cross first, because there might be other variables using the same shape as X
            Q = self._do(problem, X, **kwargs)
            assert Q.shape == (n_offsprings, n_matings, problem.n_var, problem.n_var), "Shape is incorrect of crossover impl."
            Xp[:, cross] = Q[:, cross]
            
           
        # now set the parents whenever NO crossover has been applied
        for k in np.flatnonzero(~cross):
            if n_offsprings < n_parents:
                s = np.random.choice(np.arange(self.n_parents), size=n_offsprings, replace=False)
            elif n_offsprings == n_parents:
                s = np.arange(n_parents)
            else:
                s = []
                while len(s) < n_offsprings:
                    s.extend(np.random.permutation(n_parents))
                s = s[:n_offsprings]

            Xp[:, k] = np.copy(X[s, k])
        
        # flatten the array to become a 2d-array
        Xp = Xp.reshape(10, 100, 100)
        # create a population object
        off = Population.new("X", Xp)
        

        return off

    def _do(self, problem, X, **kwargs):

        _, n_matings= X.shape[0],X.shape[1]
        # get genome of parents with CoMOLA functions
        do_crossover = np.full(X[0].shape, True)
        #save dimensions of landuse maps
        shape_landusemaps = [X[0][_].shape[0],X[0][_].shape[1]]
        child_landuse_maps1 = []
        child_landuse_maps2 = []
        for _ in range(n_matings):
            patches_parent1, genome_parent1 = create_patch_ID_map(X[0][_],0,[8,9],"True")
            patches_parent2, genome_parent2 = create_patch_ID_map(X[1][_],0,[8,9],"True")

            #-------------------------------------------------------------------------------------
            # Do crossover on genome
            #-------------------------------------------------------------------------------------

            # define number of cuts
            num_crossover_points = self.n_points
            num_cuts = min(len(genome_parent1)-1, num_crossover_points)

            # select random places to cut genome
            cut_points = random.sample(range(1,min(len(genome_parent1),len(genome_parent2))), num_cuts)
            cut_points.sort()

            # define initial genome of children
            genome_child1 = list(genome_parent1)
            genome_child2 = list(genome_parent2)

            # get parts of genome from parents to children
            j = 0
            for i in range(0,min(len(genome_parent1),len(genome_parent2))):
                if j < len(cut_points):
                    if i >= cut_points[j]:
                        j = j + 1
                # alternating parent 1 and 0
                if (j % 2) != 0:
                    genome_child1[i] = 0.
                # alternating 0 and parent 2
                if (j % 2) == 0:
                    genome_child2[i] = 0.
                # -------------------------------------------------------------------------------------
                # Create land use map children from genome
                # -------------------------------------------------------------------------------------

            rows = shape_landusemaps[0]
            cols = shape_landusemaps[1]

            # child 1: fill in genome in patches
            child1 = patches_parent1
            child2 = patches_parent2

            for x in range(0, cols):
                for y in range(0, rows):
                    if child1[x, y] != 0:
                        child1[x, y] = genome_child1[child1[x, y] - 1]
                        child2[x, y] = genome_child2[child2[x, y] - 1]

            child1 = np.where(child1 == 0, X[1][_], child1)
            child2 = np.where(child2 == 0, X[0][_], child2)
            child_landuse_maps1.append(child1)
            child_landuse_maps2.append(child2)
        return np.array([np.array(child_landuse_maps1),np.array(child_landuse_maps2)])

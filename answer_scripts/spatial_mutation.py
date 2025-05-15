import numpy as np
from pymoo.core.mutation import Mutation
from compute_genome import create_patch_ID_map 
#from compute_genome import getNbh, determine_patch_elements, create_patch_ID_map


def random_reset_mutation(genome_in, point_mutation_prob):
    genome = list(genome_in) 
    for i in range(1, len(genome)): 
        if np.random.uniform(0, 1)< point_mutation_prob:
            genome[i] = np.random.randint(low=3,high=8) 
    return (genome)

# class that performs the mutation
class SpatialNPointMutation(Mutation): 
    def __init__(self, prob=None,point_mutation_probability=0.01):
        super().__init__()
        self.prob = prob
        self.point_mutation_probability = point_mutation_probability 
        
    def _do(self, problem, X, **kwargs):
        shape_landusemaps = [X[0].shape[0], X[0].shape[1]]
        rows = shape_landusemaps[0]
        cols = shape_landusemaps[1]
        offspring = []
        # loop over individuals in population 
        for i in X:
        # performe mutation with certain probability 
            if np.random.uniform(0, 1) < self.prob:
                # get genome
                patches, genome = create_patch_ID_map(i, 0, [8, 9], "True")
                # perform mutation
                mutated_genome = random_reset_mutation(genome, self.point_mutation_probability)
                # go back to land use map
                mutated_individual = patches 
                for x in range(0, cols): 
                    for y in range(0, rows): 
                        if mutated_individual[x, y] != 0:
                            mutated_individual[x, y] = \
                            mutated_genome[mutated_individual[x, y] - 1] 
                        else:
                            mutated_individual[x, y] = i[x,y]
                mutated_individual = np.where(mutated_individual == 0, i,mutated_individual)
                offspring.append(mutated_individual)
            # if no mutation 
            else:
                offspring.append(i)
        offspring = np.array(offspring)
        return offspring    
    
'''        
    
def random_reset_mutation(genome_in, point_mutation_prob):
    genome = list(genome_in)
    for i in range(1, len(genome)):
        if np.random.uniform(0, 1)>point_mutation_prob:
            randomnumber = np.random.uniform(0, 1)
            if randomnumber < 0.2:
                genome[i] = 3.
            elif randomnumber < 0.4:
                genome[i] = 4.
            elif randomnumber < 0.6:
                genome[i] = 5.
            elif randomnumber < 0.8:
                genome[i] = 6.
            elif randomnumber < 1.0:
                genome[i] = 7.
    return (genome)


class SpatialNPointMutation(Mutation):

    def __init__(self, prob=None,point_mutation_probability=0.01):
        super().__init__()
        self.prob = prob
        self.point_mutation_probability = point_mutation_probability

    def _do(self, problem, X, **kwargs):
        
        #print(X[0])
        shape_landusemaps = [X[0].shape[0], X[0].shape[1]]
        
        rows = shape_landusemaps[0]
        cols = shape_landusemaps[1]
        offsprings = []
        
        # loop over individuals in population
        for i in X:
            # performe mutation with certain probability
            if np.random.uniform(0, 1) > self.prob:
                # get genome
                patches, genome = create_patch_ID_map(i, 0, [8, 9], "True")
                # perform mutation
                copy_genome_i = random_reset_mutation(genome,self.point_mutation_probability)
                # go back to land use map
                mutated_individual = patches
                for x in range(0, cols):
                    for y in range(0, rows):
                        if mutated_individual[x, y] != 0:
                            mutated_individual[x, y] = genome[mutated_individual[x, y] - 1]
                        else:
                            mutated_individual[x, y] = i[x,y]      
                mutated_individual = np.where(mutated_individual == 0, i, mutated_individual)
                offsprings.append(mutated_individual)
            # if no mutation
            else:
                offsprings.append(i)
        offsprings = np.array(offsprings)
        return offsprings
''' 
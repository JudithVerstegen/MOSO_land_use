import numpy as np
from pymoo.model.duplicate import NoDuplicateElimination
from pymoo.model.population import Population
from pymoo.model.repair import NoRepair


class InitializationSpatial:

    def __init__(self,
                 sampling,
                 repair=None,
                 eliminate_duplicates=None) -> None:

        super().__init__()
        self.sampling = sampling
        self.eliminate_duplicates = eliminate_duplicates if eliminate_duplicates is not None else NoDuplicateElimination()
        self.repair = repair if repair is not None else NoRepair()

    def do(self, problem, n_samples, **kwargs):

        # provide a whole population object - (individuals might be already evaluated)
        if isinstance(self.sampling, Population):
            pop = self.sampling

        else:
            if isinstance(self.sampling, np.ndarray):
                pop = Population.new(X=self.sampling)
            else:
                pop = self.sampling.do(problem, n_samples, **kwargs)

        # repair all solutions that are not already evaluated
        not_eval_yet = [k for k in range(len(pop)) if pop[k].F is None]
        if len(not_eval_yet) > 0:
            pop[not_eval_yet] = self.repair.do(problem, pop[not_eval_yet], **kwargs)

        # filter duplicate in the population
        pop = self.eliminate_duplicates.do(pop)

        return pop


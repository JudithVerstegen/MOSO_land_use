import numpy as np
from pymoo.operators.sampling.rnd import FloatRandomSampling
import initial_population



class SpatialSampling(FloatRandomSampling):
    """
    Randomly sample points in the real space by considering the lower and upper bounds of the problem.
    """

    def __init__(self, default_dir, var_type=float) -> None:
        super().__init__()
        self.default_dir = default_dir
        self.var_type = var_type

    def _do(self, problem, n_samples, **kwargs):
        landusemaps_np = initial_population.initialize_spatial(n_samples, self.default_dir)
        return np.array(landusemaps_np)




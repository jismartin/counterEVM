import numpy as np
import scipy.stats as stats
from typing import Tuple

def pert_sample(optimistic: float, most_probable: float, pessimistic: float, elapsed_time: float =0, complete_percent: float =0,
                size: int =1000, lamb: float =4) -> np.ndarray:
    """
    Generate random samples from a PERT-like distribution.

    Parameters:
        optimistic (float): Optimistic estimate (O)
        most_probable (float): Most Probable estimate (M)
        pessimistic (float): Pessimistic estimate (P)
        elapsed_time (float): Elapsed time since the start of the activity
        size (int): Number of samples to generate
        lamb (float): Shape parameter (lower values = skewed distribution)

    Returns:
        numpy array of sampled durations

    """

    # Planned estimates
    a=optimistic
    b=most_probable
    c=pessimistic

    # Update planned estimates based on real progress
    if complete_percent > 0:
        t=elapsed_time/complete_percent # Total time estimated
        dev_t=t-(a+4*b+c)/6 # Deviation from planned time
        a=a+dev_t # Update optimistic estimate
        b=b+dev_t # Update most probable estimate
        c=c+dev_t # Update pessimistic estimate

    # Compute alpha and Beta parameters
    alpha = 1 + lamb * (b-a) / (c-a)
    beta = 1 + lamb * (c-b) / (c-a)
   
    # Generate samples from Beta distribution
    samples = stats.beta.rvs(alpha, beta, size=size)
    
    # Scale samples to the feasible range
    a0 = elapsed_time if complete_percent > 0 else a
    return a0 + samples * (c-a0)

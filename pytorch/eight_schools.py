# %%
import torch as xp
from cuqi.distribution import JointDistribution
from cuqi.sampler import CWMH
from cuqipy_pytorch.distribution import Gaussian, LogGaussian
from cuqipy_pytorch.sampler import NUTS

# Observations
y_obs = xp.tensor([28, 8, -3,  7, -1, 1,  18, 12], dtype=xp.float32)
σ_obs = xp.tensor([15, 10, 16, 11, 9, 11, 10, 18], dtype=xp.float32)

# Bayesian model
μ     = Gaussian(0, 10**2)
τ     = LogGaussian(5, 1)
θp    = Gaussian(xp.zeros(8), 1)
θ     = lambda μ, τ, θp: μ+τ*θp
y     = Gaussian(θ, cov=σ_obs**2)

# Posterior sampling
joint = JointDistribution(μ, τ, θp, y)   # Define joint distribution 
posterior = joint(y=y_obs)               # Define posterior distribution
sampler = NUTS(posterior)                # Define sampling strategy
samples = sampler.sample(N=1000, Nb=500) # Sample from posterior

# Plot posterior samples
samples["θp"].plot_violin(); 
print(samples["μ"].mean()) # Average effect
print(samples["τ"].mean()) # Average variance

# %% Try CWMH (works ok, but not as good as NUTS)
sampler2 = CWMH(posterior._as_stacked())
samples2 = sampler2.sample_adapt(N=5000, Nb=2000)
samples2.plot_violin(xp.arange(2,10)); 

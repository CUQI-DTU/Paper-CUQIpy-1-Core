

# %%
import sys
sys.path.append('..')

# %%
# Setup
# -----
# We start by importing the necessary modules

import cuqi
import numpy as np
import matplotlib.pyplot as plt

do_save = False


#%%

def mysavefig(filename):
    plt.savefig(filename,
                bbox_inches='tight', 
                dpi=300)

s = 0.005

#%% First do the sinc case just to get same ylims
np.random.seed(4)
A, y_data, info = cuqi.testproblem.Deconvolution1D.get_components(
    phantom='sinc',
    noise_std=s
    )

x_true = info.exactSolution
y_true = info.exactData

x_true.plot(label="Exact solution")
y_true.plot(label="Exact data")
y_data.plot(label="Synthetic data")
(y_data-y_true).plot(label="Noise")
ax = plt.gca()
ylims = ax.get_ylim()

# %% First take a smooth signal and try with Gaussian prior
np.random.seed(4)

TP = cuqi.testproblem.Deconvolution1D(phantom='square', 
                                      noise_std=s)

#%%
A = TP.model
y_data = TP.data
x_true = TP.exactSolution
y_true = TP.exactData

x_true.plot(label="Exact solution")
y_true.plot(label="Exact data")
y_data.plot(label="Synthetic data")
(y_data-y_true).plot(label="Noise")
plt.ylim(ylims)
plt.title("Deconvolution1D test problem, phantom=\"square\"")
plt.legend()

if do_save:
    mysavefig('pics/testprob_square.png')

# %% Demonstrate forward and adjoint
yx = A @ x_true
xx = A.T(yx)

# %%
print(x_true.geometry)

# %%
x = cuqi.distribution.Gaussian(np.zeros(A.domain_dim), 
                           0.1)

TP.prior = x

# %%
samples = TP.UQ()

#%%
x_map = TP.MAP()

samples.plot_ci(exact=x_true)
if do_save:
    mysavefig('pics/square_gaussian_ci.png')

# %%

x_true.plot()
x_map.plot()
samples.plot(500)
plt.legend(("true","MAP","sample"))
if do_save:
    mysavefig('pics/square_gaussian_samples_map.png')



# %%
x = cuqi.distribution.Cauchy_diff(np.zeros(A.domain_dim), 
                           0.01)
TP.prior = x

#%%
samples = TP.UQ()

#%%
x_map = TP.MAP()

#%%
samples.plot_ci(exact=x_true)
if do_save:
    mysavefig('pics/square_cauchy_ci.png')

#%%
x_true.plot()
x_map.plot()
samples.plot(500)
plt.legend(("true","MAP","sample"))
if do_save:
    mysavefig('pics/square_cauchy_samples_map.png')

# %%
# Bayesian problem (Joint distribution)
# -------------------------------------
#
# After defining the prior and likelihood, we can now define the Bayesian problem. The
# Bayesian problem is defined by the joint distribution of the solution and the data.
# This can be seen when we print the Bayesian problem.


# %% Try laplace diff

x = cuqi.distribution.Laplace_diff(np.zeros(A.domain_dim), 
                           0.002)

#%%
TP.prior = x

#%%
samples = TP.UQ()

#%%
x_map = TP.MAP()

#%%
samples.plot_ci(exact=x_true)
if do_save:
    mysavefig('pics/square_laplace_ci.png')

#%%
x_true.plot()
x_map.plot()
samples.plot(500)
plt.legend(("true","MAP","sample"))
if do_save:
    mysavefig('pics/square_laplace_samples_map.png')
# %%

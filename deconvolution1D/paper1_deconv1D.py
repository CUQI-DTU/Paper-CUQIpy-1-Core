

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

do_save = True

##
np.random.seed(4)

def mysavefig(filename):
    plt.savefig(filename,
                bbox_inches='tight', 
                dpi=300)

s = 0.01

# %% First take a smooth signal and try with Gaussian prior

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
plt.title("Deconvolution1D test problem, phantom=\"sinc\"")
plt.legend()

if do_save:
    mysavefig('pics/testprob_sinc.png')

# %% Demonstrate forward and adjoint
yx = A @ x_true
xx = A.T(yx)

# %%
print(x_true.geometry)
# %%

v = cuqi.distribution.Gaussian(mean=np.zeros(A.domain_dim),
                               cov=0.1)
                              # geometry=A.domain_geometry )
sv = v.sample(3)
sv.plot()

if do_save:
    mysavefig('pics/distb_gaussian1D_samples.png')


# %% Conditional
w = cuqi.distribution.Gaussian(mean=np.zeros(A.domain_dim), cov=lambda q: q)

G = cuqi.distribution.Gamma(shape=1)
print(G)

# %%
G(rate=0.1).sample()

#%%
G = cuqi.distribution.Gamma(shape=1, rate=lambda R: R)
print(G)
G(R=0.1).sample()

# %%
print(info.infoString)
# %%

x = cuqi.distribution.Gaussian(mean=np.zeros(A.domain_dim), cov=0.1,
geometry=A.domain_geometry)

y = cuqi.distribution.Gaussian(A@x, s**2)

# %%
y(x=x_true).sample(5).plot()
if do_save:
    mysavefig('pics/sinc_multiple_data.png')

#%%
#x = cuqi.distribution.GMRF?
x = cuqi.distribution.GMRF(mean=np.zeros(A.domain_dim), 
                           prec=50)  # 100 and 200 quite good

#x = cuqi.distribution.Gaussian(mean=np.zeros(A.domain_dim), 
#                           cov=0.1)   #  0.1 good, but wide

sam_x = x.sample(3)
sam_x.plot()

if do_save:
    mysavefig('pics/distb_gmrf1D_samples.png')



# %%
# Bayesian problem (Joint distribution)
# -------------------------------------
#
# After defining the prior and likelihood, we can now define the Bayesian problem. The
# Bayesian problem is defined by the joint distribution of the solution and the data.
# This can be seen when we print the Bayesian problem.

BP = cuqi.problem.BayesianProblem(y, x)
print(BP)

# %%
# Setting the data (posterior)
# ----------------------------
#
# Now to set the data, we need to call the :func:`~cuqi.problem.BayesianProblem.set_data`

BP.set_data(y=y_data)
print(BP)

# %%
x_ml = BP.ML()
x_map = BP.MAP()

x_ml.plot()
x_map.plot()
x_true.plot()
y_data.plot()
plt.legend(("ML", "MAP", "true sig", "data"))

if do_save:
    mysavefig('pics/sinc_ML_MAP.png')

# %%
#
# We can then use the automatic sampling method to sample from the posterior distribution.

samples = BP.sample_posterior(1000)

# %%
# Plotting the results
# --------------------

samples.plot_ci(exact=info.exactSolution)

if do_save:
    mysavefig('pics/sinc_ci.png')


# %%

samplesUQ = BP.UQ(exact=x_true)


#%% Now hierarchical

s = cuqi.distribution.Gamma(1, 1e-4)

# %%
# Update likelihood with unknown noise level
# ------------------------------------------

y = cuqi.distribution.Gaussian(A @ x, prec=lambda s: s)

# %%
# Bayesian problem (Joint distribution)
# -------------------------------------

BP = cuqi.problem.BayesianProblem(y, x, s)

print(BP)

# %%
# Setting the data (posterior)
# ----------------------------
#

BP.set_data(y=y_data)

print(BP)

# %%
# Sampling from the posterior
# ---------------------------

samples = BP.sample_posterior(1000)


# %%
# Plotting the results
# --------------------
#
# Let is first look at the estimated noise level
# and compare it with the true noise level

samples["s"].plot_trace(lines=(("s", {}, 1/0.01**2),))

if do_save:
    mysavefig('pics/sinc_hier_s_trace.png')

# %%
# We see that the estimated noise level is close to the true noise level. Let's
# now look at the estimated solution


samples["x"].plot_ci(exact=info.exactSolution)
if do_save:
    mysavefig('pics/sinc_hier_ci.png')


# %%

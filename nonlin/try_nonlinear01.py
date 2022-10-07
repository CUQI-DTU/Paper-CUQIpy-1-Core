# %%
import cuqi

#%% Import numpy for "matlab-like" array operations,
import numpy

#%% Set up a vector to be median-filtered
tx = numpy.array([4, 6, 8, 9, 10, 1, 7.9, 3.1, 4.5, -1.2])
print(tx)

#%%
N = len(tx)

#%% Specify halfwidth of median filter (full width is 2*halfwidth+1)
halfwidth = 2

#%% Initialize output vector y to hold median filtered vector (here use same size as x but can be changed if desired)
ty = numpy.zeros(tx.shape)

#%%
# Loop through x, pick out the 2*halfwidth+1 subvectors, compute the median and store as element of y.
# Print subvectors and median values along the way.
# Note the in python slicing is similar to matlab: Where MATLAB you would do x(2:4) to
# get a 3-element vector, in python you do x[1:4]. That is, square brackets are used,
# counting starts from zero, and also importantly the last element is excluded which
# is why we need 1:4 and not 1:3, which would give two elements.
for i in range(halfwidth, len(tx)-halfwidth):
    subvec = tx[i-halfwidth:i+halfwidth+1]
    print(subvec)
    ty[i] = numpy.median(subvec)
    print(ty[i])

#%%
# Print the results
print(ty)

#%%
# We can define a function to do the same as above to an input vector x:
def rollingmedian(xx):
    yy = numpy.zeros(xx.shape)
    for i in range(halfwidth, len(xx)-halfwidth):
        subvec = xx[i-halfwidth:i+halfwidth+1]
        yy[i] = numpy.median(subvec)
    return yy

#%%
# We apply the function and see that it produces the same result as above
ty2 = rollingmedian(tx)
print(ty2)

# %% Load existing deconvolution test problem

model1, data1, probInfo1 = cuqi.testproblem.Deconvolution1D.get_components(dim=64, phantom="sinc")


# %%
x_true = probInfo1.exactSolution
x_true.plot()

# %%
N = len(x_true)

# %% Put inside a CUQIpy model
A = cuqi.model.Model(rollingmedian, range_geometry=N, domain_geometry=N)

# %% Generate clean data
y_true = A(x_true)

# %%
x_true.plot()
y_true.plot()

# %% Define prior
sig_x = 0.1
x = cuqi.distribution.GaussianCov(numpy.zeros(N), sig_x**2)

# %%  Data distribution
sig_y = 0.01

# Define data distributions, nonlinear model A
y = cuqi.distribution.GaussianCov(A(x), sig_y**2)

# %% Generate noisy data as sample of y given x=x_true
y_noisy = y(x=x_true).sample()

# %%
y_true.plot()
y_noisy.plot()

# %% First try non-expert interface
BP = cuqi.problem.BayesianProblem(y, x)
BP.set_data(y=y_noisy)

# %% Generate samples from automated sampler selection (pCN)
samples = BP.UQ()

# %%
samples.plot_mean()

# %%
samples.plot([1,2,-1,-2,-3])

# %% Go manual for more control
joint = cuqi.distribution.JointDistribution(y, x)
posterior = joint(y=y_noisy)

# %% Try the componentwise Metropolis Hasting sampler
mysampler = cuqi.sampler.CWMH(posterior)
samples2 = mysampler.sample_adapt(1000, 500)

# %%
samples2.plot_mean()
# %%
samples2.plot()
# %%
samples2.plot_std()
# %%

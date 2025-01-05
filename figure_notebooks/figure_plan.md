# Figure 1.
The purpose of Figure 1 is to present the probabilistic ruggedness metric. What is the heat kernal computing, what is happening to the spectral energies of the graph. 

### Panels A - B show that the spectrum of the graph Laplacian describe how the signal flows through the graph. The heat kernel can emphasize the smooth or non-smooth components of the flow as a function of the timestep, t. 

## Panel A
Simulated NK landscape network graph (do not show network graph). Cumulative sum of eigenvalues from the Laplacian decomposition.  

## Panel B
Using the same graph topology as Panel A, the cumulative sum of spectral energies computed under H0 (i.e. the heat kernel) at t = 0.1 - 100. As t increases, the contribution of low eigenmodes contributes less energy to the observed function. 

### Panel C - D are a repeat of panels A - B, however show the GFT (and spectral energy) instead of the laplacian eigenvalues.

## Panel C 
Repeat of Panel A, but plotting the cumulative sum of spectral energies instead of eigenvalues. 

## Panel D 
Repeat of Panel B, but plotting the cumulative sum of spectral energies instead of eigenvalues, after transformation against the heat kernel. 

## Panel E
Heatmap of the covariance matrix at different values of t. 

## Panel F
PCA projection of the signal over an abitrary (random, or very high K-valued NK) graph and a sample drawn from H0 at a set time point. 

# Figure 2.
Figure 2 reports data simulated with molecular evolution, laplacian eigenvectors and NK-landscapes. 

### Panels A - D report behaviour on smooth simulated fitness landscapes (laplacian eigenvector = 2). 

## Panel A
Network graph of a simulated landscape. The signal is the 2nd Laplacian eigenvector. 

## Panel B
Network graph of a sample drawn under H0. 

## Panel C
Network graph of the marginal variances according to the GMRF model. 

## Panel D
Log-likelihood distribution with empirical value shown.

### Panels E - H Repeat of panels A - D on rugged simulated landscape (laplacian eigenvector = 50)

## Panel I
Log-likelihood of simulated fitness as a function of the Laplacian eigenvector. 

## Panel J
Log-likelihood of simulated NK fitness landscape as a function of K. 


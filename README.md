# State-space model parameter estimation using particle marginal Metropolis-Hastings

This repo is an empirical implementation of the particle marginal Metropolis-Hastings algorithm [(Andrieu et al.)](https://www.stats.ox.ac.uk/~doucet/andrieu_doucet_holenstein_PMCMC.pdf) for a simple state-space model with 1-D latent states and 2-D observations.

The state-space model can be readily scaled to an arbitrary dimensionality of the observation space.

Directory `scripts` contains three Python scripts: 
- Data Generation
- Particle filter
- Particle marginal Metropolis-Hastings

Directory `notebooks` contains a single Jupyter notebook that implements the algorithms provided on `scripts` simply for demonstration purposes.

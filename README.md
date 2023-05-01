# PyTorchDistributionsExtended
A project to implement all 80+ `scipy` continuous probability distributions in PyTorch.

This project was undertaken to properly implement a Hamiltonian Monte Carlo in the `PyLevyProcess` library using `hamiltorch`. The existing `PyLevyProcess` implementation uses a direct Monte Carlo sampler. For more rigorous returns innovation sampling, Hamiltonian Monte Carlo is needed to ensure no/little autocorrelation among sampled returns, as needed for a Levy Process. As neither `scipy` nor `Jax` support gradients for all probability distributions, this project fills a vital niche.

# Examples
WIP

# References
WIP

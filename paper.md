---
title: 'GPUE: Graphics Processing Unit Gross-Pitaevskii Equation solver'
tags:
  - CUDA
  - physics
  - dynamics
  - quantum
  - nonlinear
  - Bose-Einstein condensate
authors:
  - name: James R. Schloss
    orcid: 
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Lee J. O'Riordan
    orcid: 0000-0002-6758-9433
    affiliation: 1
affiliations:
 - name: Okinawa Institute of Science and Technology Graduate University, Onna-son, Okinawa 904-0495, Japan.
   index: 1
date: 01 August 2033
bibliography: paper.bib
---

# Summary

Bose--Eintein Condensates (BECs) are superfluid systems that can be created in an expermental laboratory and allow for the the exploration of many interesting physical phenomenon, such as superfluid turbulence.
Simulations of BEC systems allow for fundamentally new discoveries that directly mimic what can be seen in experiments and are thus highly valuable for fundamental research.
In practice, almost all dynamics of BEC systems can be found by solving the non-linear Schr&ouml;dinger equation known as the Gross-Pitaevskii Equation (GPE):

$$
\frac{\partial\Psi(\mathbf{r},t)}{\partial t} = \left( -\frac{\hbar^2}{2m} \frac{\partial}{\partial\mathbf{r}^2} + V(\mathbf{r}) + g|\Psi(\mathbf{r},t)|^2\right)\Psi(\mathbf{r},t)
$$

Where $\Psi(\mathbf{r},t)$ is the many-body wavefunction of the quantum system, $m$ is the atomic mass, $V(\mathbf{r})$ is a potential to trap the atomic system, $g = \frac{4\pi\hbar^2a_s}{m}$ is a coupling factor, and $a_s$ is the scattering length of the atomic species.
Here, the GPE is shown in 1 dimension, but it can easily be extended to 2 or 3 dimensions, if necessary.
Though there are many methods to solve the GPE, one of the most straightforward is the split-operator method, which has previously been accelerated with GPU devices; however, there are no general software packages available using this method.
There are several software packages with similar goals, including GPELab and the Massively Parallel Trotter-Suzuki Solver.


GPUE is a GPU-based Gross-Pitaevskii Equation solver via the split-operator method for superfluid simulations of Bose-Einstein Condensates with an emphasis on vortex dynamics in 2 and 3 dimensions.
For this purpose, GPUE provides a number of important features:
1. Dynamic field generation of variables on the GPU device
2. Vortex tracking in 2D and vortex highlighting in 3D
3. Configurable gauge fields for the generation of artificial magnetic fields and corresponding vortex distributions.

All of these features are essential for usability and performance and have all been adequately described in the documentation.
GPUE provides a fast and accessible method to simulate superfluid physics for fundamental research in the area.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

# Figures

# Acknowledgements

# References

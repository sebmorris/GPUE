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
    orcid: 0000-0002-3243-8918
    affiliation: 1
  - name: Lee J. O'Riordan
    orcid: 0000-0002-6758-9433
    affiliation: 1
affiliations:
 - name: Okinawa Institute of Science and Technology Graduate University, Onna-son, Okinawa 904-0495, Japan.
   index: 1
date: 21 September 2018
bibliography: paper.bib
---

# Summary

Bose--Eintein Condensates (BECs) are superfluid systems consisting of bosonic atoms that have been cooled and condensed into a single, macroscopic ground state.
These systems can be created in an expermental laboratory, and allow for the the exploration of many interesting physical phenomenon, such as superfluid turbulence, chaotic dynamics, and as analogs of other quantum systems.
Numerical simulations of BECs allow for new discoveries that directly mimic what can be seen in experiments and are thus highly valuable for fundamental research.
In practice, almost all dynamics of BEC systems can be found by solving the non-linear Schr&ouml;dinger equation known as the Gross--Pitaevskii Equation (GPE):

\frac{\partial\Psi(x,t)}{\partial t} = \left( -\frac{\hbar^2}{2m} \frac{\partial}{\partial x^2} + V(\mathbf{r}) + g|\Psi(\mathbf{r},t)|^2\right)\Psi(x,t)
$$

Where $\Psi(x,t)$ is the one-dimensional many-body wavefunction of the quantum system, $m$ is the atomic mass, $V(\mathbf{r})$ is a potential to trap the atomic system, $g = \frac{4\pi\hbar^2a\_s}{m}$ is a coupling factor, and $a\_s$ is the scattering length of the atomic species.
Here, the GPE is shown in one dimension, but it can easily be extended to two or three dimensions, if necessary.
Though there are many methods to solve the GPE, one of the most straightforward is the split-operator method, which has previously been accelerated with GPU devices [@Ruf2009,@Bauke2011]; however, there are no generalized software packages available using this method that allow for user-configurable simulations and a variety of different system types. Even so, there are several software packages designed to simulate BECs with other methods, including GPELab [@Antoine2014] and the Massively Parallel Trotter-Suzuki Solver [@Wittek2013].

GPUE is a GPU-based Gross-Pitaevskii Equation solver via the split-operator method for superfluid simulations of both linear and non-linear Schr&ouml;dinger equations, with an emphasis on Bose--Einstein Condensates with vortex dynamics in 2 and 3 dimensions.
For this purpose, GPUE provides a number of unique features:
1. Dynamic field generation for trapping potentials and other variables on the GPU device
2. Vortex tracking in 2D and vortex highlighting in 3D
3. Configurable gauge fields for the generation of artificial magnetic fields and corresponding vortex distributions
4. Vortex manipulation via direct control of the wavefunction phase

All of these features are essential for usability and performance, and have all been adequately described in the documentation [@documentation].
GPUE provides a fast, robust, and accessible method to simulate superfluid physics for fundamental research in the area and has been used to simulate and manipulate large vortex lattices in two dimensions [@Oriordan2016, @Oriordan2016b] and spatial adiabatic passage in atom chips [@Oriordan2013], along with ongoing studies on vortex turbulence in two dimensions and vortex structures in three dimensions.

# Acknowledgements
This work has been supported by the Okinawa Institute of Science and Technology Graduate University and by JSPS KAKENHI Grant Number JP17J01488.
We would also like to thank Thomas Busch, Rashi Sachdeva, Tiantian Zhang, and Angela White for discussions on useful physical systems to simulate with the GPUE codebase, along with Peter Wittek and Tadhg Morgan for contributions to the code, itself.

# References

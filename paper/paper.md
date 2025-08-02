---
title: 'ChemFit: A Python package for fitting atomic potentials'
tags:
  - Python
  - Chemistry
  - Physics
  - Optimization
  - Molecular dynamics
authors:
  - name: Moritz Sallermann
    orcid: 0000-0002-2355-9355
    affiliation: "1, 2" # (Multiple affiliations must be quoted)

affiliations:
 - name: RWTH Aachen
   index: 1
 - name: University of Iceland
   index: 2

date: 02 08 2025
bibliography: paper.bib

---

# Summary

`ChemFit` is a Python package aimed at optimizing the "hyper-parameters" of chemical force fields by minimizing objective functions. `ChemFit` integrates with the Atomic Simulation environment ([ASE](https://wiki.fysik.dtu.dk/ase/)) and is able to optimize parameters for any ASE [calculator](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html).

On the one hand, `ChemFit` provides extremely flexible and extendable objective functions and on the other hand it has a unified interface to two different optimization backends.

# Statement of need

When not working from the "ab-initio" paradigm, it is a frequent task in physical chemistry to fit (semi-)empirical interaction potentials to data obtained from other sources.
Such data may be sourced from experiments or, indeed, the aforementioned ab-initio methods.

Especially in molecular dynamics, where classical force-fields are used to approximate interactions which can only be properly described at the quantum level of theory, some amount of "fitting" is unavoidable.

While numerical optimization libraries are *extremely* ubiquitous, they share the common need to set up objective functions (in a potentially restrictive way).

For physical-chemistry applications this can present a challenge, since the required book-keeping of input files and parameters can quickly get out of hand and these programs are usually made with 

`ChemFit` is designed to help in exactly such a scenario by providing tested and well designed interface to define such objective functions.
Since it integrates with `ASE`, which itself is an extremely flexible tool, it should be widely applicable.

# Features

- Extendable objective functions based on ASE. 
  - Simply derive from the abstract base class and implement `__call__(params)`
  - Lazy-loading mechanism to

- MPI parallelization over contributions. 
  - Run the fit on the main rank and use all other ranks to collect contributions in parallel

- Pre-defined objective functions for locally optimized structures (using the Kabsch algorithm to compare similarity) energy based objective functions

- Robust `Fitter` class, providing access to different optimization backends with 
  - per-parameter bounds
  - intelligent logging
  - robustness against exceptions and "bad" parameter values

# Availability

The source code of `ChemFit` is freely available under the MIT license on GitHub: [https://github.com/MSallermann/chemfit/](https://github.com/MSallermann/chemfit/) from where the newest develop version may be installed.
The documentation is hosted on ReadTheDocs: [https://chemfit.readthedocs.io/en/latest/](https://chemfit.readthedocs.io/en/latest/)
Lastly, a stable version may be installed from PyPI [https://pypi.org/project/chemfit/](https://pypi.org/project/chemfit/), e.g. with `pip install chemfit`.

<!-- 
Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

-->

# References
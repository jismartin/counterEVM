# Actionable decision-making in project management control using counterfactual explanations

This repository contains all code, data, and notebooks to reproduce the results of:

> Pereda, M., Santos, J. I., Ahedo, V. & Galán, J. M. (2025).
> *Actionable decision-making in project management control using counterfactual explanations*
> **X**(Y): ZZ–ZZ. [PDF](https://…)

## 🧠 Summary

This repository contains all code, data, and results supporting the paper on a novel framework for project management under uncertainty. The method combines stochastic simulation (Triad method), black-box predictive models, and counterfactual explanations to generate actionable, task-level replanning strategies. Unlike traditional explainable-AI tools, this approach goes beyond diagnostics to suggest minimal, targeted interventions—such as task duration adjustments or resource reallocations—to restore project performance. The included case study demonstrates how these prescriptive counterfactuals (CFs) can support on-time delivery with minimal disruption.

## 📁 Repository Structure

```
.
├── figures/                                # Pre-rendered figures for the manuscript
├── input_data/                             # Input dataset of the Fuel Tank Filter project
├── models/
│   └── gbc_best_model.pkl                  # Gradient Boosting model selected to generate CFs 
│   └── tf_best_model.keras                 # Tensor Flow model selected to generate CFs  
├── notebooks/
│   ├── cfs_generation.ipynb                # Generation and interpretation of CFs
│   ├── model_selection.ipynb               # Selection of models for CFs analysis
│   ├── monte_carlo_simulation.ipynb        # Generation of the simulation data at the EV
│   ├── pert_sample.ipynb                   # Checking the Pert beta distribution
│   ├── plots.ipynb                         # Generation of some figures
│   ├── selection_of_instance.ipynb         # Selection of instances for CFs analysis
│   ├── tools.py                            # Set of functions used in all notebooks

├── sim_data/                               # Simulation and other results
├── environment.yml                         # Conda environment specification
├── LICENSE                                 # MIT License
└── README.md                               # This file
```

## 🛠 Installation & Requirements

### Python Environment

* **Python Version:** 3.10 or higher
* **Environment Setup:**

  ```bash
  conda env create -f environment.yml
  conda activate counterevm-env
  ```

## 📝 License

This project is licensed under the **MIT License**. See the LICENSE file for details.

## 🧑‍🤝‍🧑 Acknowledgements

The authors would like to express their gratitude for the support and funding provided by the Ministry of Science and Innovation through its excellence network RED2022-134890-T, the project PID2020118906GB-I00, and the MOMENTUM program project MMT24-IMF-02. We acknowledge Santander Supercomputación support group at the University of Cantabria who provided access to the supercomputer Altamira Supercomputer at the Institute of Physics of Cantabria (IFCA-CSIC), member of the Spanish Supercomputing Network, for performing simulations/analyses.

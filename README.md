# **Posterior Consistency in Bayesian Factor Models**

This repository implements the findings and methodologies presented in the research paper: *On Posterior Consistency of Bayesian Factor Models in High Dimensions* by Yucong Ma and Jun S. Liu. It focuses on replicating the key analyses, simulations, and theoretical validations of posterior consistency in high-dimensional Bayesian factor models.

## **Overview**
Factor models are crucial tools for interpretable dimensionality reduction in high-dimensional data analysis, commonly used in fields like bioinformatics, social sciences, and economics. This project explores:
- Sparse Bayesian factor models.
- Theoretical and empirical analyses of posterior consistency.
- The phenomenon of "magnitude inflation" in loading matrices.
- Proposed solutions like the \(\sqrt{n}\)-orthonormal factor assumption.

## **Key Features**
1. **Bayesian Sparse Factor Model Implementation:**
   - Incorporates the Sparse Spike-and-Slab (SpSL) prior.
   - Supports the \(\sqrt{n}\)-orthonormal factor model for robust posterior consistency.

2. **Efficient Gibbs Sampling:**
   - Includes enhancements for posterior sampling, addressing magnitude inflation issues.

3. **Simulations and Real-World Data:**
   - Synthetic data generation for high-dimensional sparse scenarios.
   - Validation using benchmark datasets (e.g., gene expression data).

4. **Comparison of Approaches:**
   - Benchmarks against models like those by Ghosh and Dunson (2009) and Bhattacharya and Dunson (2011).

---

## **Repository Structure**
```plaintext
bayesian-factor-model/
├── src/
│   ├── models/            # Core Bayesian model definitions
│   ├── sampling/          # Gibbs sampler implementations
│   ├── simulations/       # Synthetic data and analysis scripts
│   ├── utils/             # Helper functions
│   └── experiments/       # Real-world data analyses
├── data/
│   ├── synthetic/         # Generated datasets
│   └── real-world/        # Processed real-world datasets
├── tests/                 # Unit and integration tests
├── results/               # Outputs from simulations and analyses
├── docs/                  # Documentation for the project
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
└── LICENSE                # License for the repository

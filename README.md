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

## **References**

Bhattacharya, A., & Dunson, D. B. (2011). Sparse Bayesian infinite factor models. Biometrika, 98(2), 291–306. https://doi.org/10.1093/biomet/asr013

Fruehwirth-Schnatter, S., & Lopes, H. F. (2018). Sparse Bayesian factor analysis when the number of factors is unknown. Bayesian Analysis, 13(1), 1–22. https://doi.org/10.1214/17-BA1063

Ghosh, J., & Dunson, D. B. (2009). Default prior distributions and efficient posterior computation in Bayesian factor models. Journal of Computational and Graphical Statistics, 18(2), 306–320. https://doi.org/10.1198/jcgs.2009.08101

Ročková, V., & George, E. I. (2016). Fast Bayesian factor analysis via automatic rotations to sparsity. Journal of the American Statistical Association, 111(516), 1608–1622. https://doi.org/10.1080/01621459.2016.1234131

Ma, Y., & Liu, J. S. (2021). On posterior consistency of Bayesian factor models in high dimensions. arXiv preprint arXiv:2006.01055. https://arxiv.org/abs/2006.01055

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

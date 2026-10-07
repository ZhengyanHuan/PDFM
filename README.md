# Primal-Dual Flow Matching for Sample-Wise Constrained Generation

Code for **Primal-Dual Flow Matching for Sample-Wise Constrained Generation**, by **Zhengyan Huan, Peter Y. Lu, and Shuchin Aeron**, accepted at **NeurIPS 2026**.

## Paper and method

The paper studies generative modeling with sample-wise constraints: given a training distribution and a user-specified target constraint satisfaction rate, the goal is to generate feasible samples while preserving distributional fidelity. Constraints are specified through a binary membership oracle that determines whether a sample is feasible. No differentiable constraint distance, projection, or convexity assumption is required by the practical method.

**Primal-Dual Flow Matching (PDFM)** combines flow matching with primal-dual optimization. A learned feasibility predictor turns binary oracle feedback into differentiable guidance for the velocity field, while a dual variable adapts the weight of the constraint term according to the gap between the measured satisfaction rate and the target. The paper develops constraint satisfaction guarantees under stated assumptions and evaluates the practical method on synthetic, PDE, fingerprint, and molecule generation tasks.

## Experiments

Each folder corresponds to an experiment in the paper. The main notebooks are linked below.

| Paper section | Experiment | Main notebook |
| --- | --- | --- |
| Sec. 5.1 | **Illustrative 2-D generation.** Samples from disconnected square regions are corrupted by Gaussian noise. This experiment compares PDFM with FM trained on the noisy data or its feasible subset, examining constraint satisfaction and distributional fidelity. | [main_5_1.ipynb](Sec_5_1/main_5_1.ipynb) |
| Sec. 5.2 | **Higher-dimensional ball constraints.** Gaussian-mixture samples are restricted to a unit Euclidean ball in 8 or 20 dimensions. The experiment studies feasibility and distribution matching even when all training samples satisfy the constraint. | [main_5_2.ipynb](Sec_5_2/main_5_2.ipynb) |
| Sec. 5.3 | **PDE-constrained generation.** Generate Kuramoto-Sivashinsky space-time patches from clean or degraded training data. A numerical solver checks dynamical consistency through a residual threshold; evaluation also considers summary-statistic histograms. | [main_5_3.ipynb](Sec_5_3/main_5_3.ipynb) |
| Appendix H.4 | **Fingerprint generation.** Generate SOCOFing fingerprint images with a ridge-connectivity constraint. The oracle thresholds and skeletonizes each image, then checks its number of connected ridge components. | [main_H4.ipynb](Sec_H4/main_H4.ipynb) |
| Appendix H.5 | **3-D molecule generation.** Generate QM9 molecules with a molecule-stability constraint: all atoms must satisfy valency rules after bond inference. Evaluation considers molecule stability and negative log-likelihood. | [main_H5.ipynb](Sec_H5/main_H5.ipynb) |



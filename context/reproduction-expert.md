# Reproduction Expert Context

## What Determines Reproduction Difficulty

### Paper-side Factors
- **Method completeness**: Does the paper describe all steps from data preprocessing to final evaluation? Papers that say "we follow standard practice" without specifying leave critical gaps.
- **Hyperparameter disclosure**: Are all hyperparameters explicitly stated? Missing learning rates, batch sizes, and optimizer settings are the #1 cause of reproduction failure.
- **Evaluation protocol clarity**: Are the metrics, datasets, and evaluation scripts clearly defined? Ambiguous evaluation leads to meaningless comparisons.
- **Algorithmic novelty**: Novel loss functions, custom layers, or non-standard training procedures require more effort and have higher failure risk than standard architectures.

### Implementation-side Factors
- **Framework compatibility**: Does the approach assume specific framework features (e.g., PyTorch-specific ops)?
- **Hardware requirements**: GPU memory requirements often determine whether the approach is feasible.
- **Data availability**: Proprietary or restricted datasets block reproduction entirely.
- **Random seed sensitivity**: Some methods are highly sensitive to initialization; papers rarely report this.

## Common Failure Modes in Reproduction

1. **"Details in the appendix" syndrome**: Critical implementation details buried in appendices or supplementary material that are easy to miss.
2. **Implicit preprocessing**: Data preprocessing steps assumed to be "standard" but varying significantly between implementations.
3. **Training instability**: Methods that require careful learning rate scheduling or gradient clipping that isn't fully documented.
4. **Evaluation mismatch**: Using different evaluation protocols (e.g., last checkpoint vs best checkpoint, different tokenizers).
5. **Codebase drift**: When official code exists but doesn't match the paper due to post-publication changes.

## MVP vs Full Plan: How to Split

### MVP Should Include
- The single most important experiment (usually Table 1 / main result)
- The core algorithm with minimal configuration
- A single dataset (the smallest or most accessible)
- One evaluation metric (the primary one)
- Hardcoded reasonable defaults for unclear parameters
- CPU or single-GPU feasibility

### MVP Should Exclude
- Ablation studies
- Multi-dataset evaluation
- Hyperparameter search
- Distributed training
- Baseline comparison (unless trivially available)
- Secondary metrics

### Full Plan Adds
- All experiments from the paper
- All datasets and metrics
- Complete hyperparameter configuration
- Baseline implementations or references
- Ablation studies
- Performance optimization (multi-GPU, mixed precision)

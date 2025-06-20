# Diffing

**Diffing** is a type of binary similarity task where the goal is to identify matching functions between two binaries compiled under different settings.

### Key Points:
- Let **A** and **B** be two sets of functions extracted from different binaries.
- Each function in A and B is represented as a VexIR2Vec embedding:
  - A = {a₁, a₂, ..., aₘ}
  - B = {b₁, b₂, ..., bₙ}
- The diffing module `Mdiff(A, B)` produces a list `L_diff` containing up to `min(m, n)` pairs of matched functions `(aᵢ, bⱼ)`.
- Each function from **A** and **B** appears **only once** in `L_diff`.
- The objective is to match every function in **A** to its closest or most equivalent counterpart in **B**, based on their VexIR2Vec representations.

## Inference

Usage:
```
bash ./v2v_wrapper_diffing.sh <path-to-diffing-ground-truth-directory> <suffix-to-groundtruth-project-directory> <path-to-vexir2vec.model> <path-to-x86-data-all | path-to-arm-data-all> <number-of-threads> <Projects-names> 
```
> [!NOTE]
>  `Projects` should be given as 'project1 project2'

Example
```bash ./v2v_wrapper_diffing.sh ~/diffing-groundtruth -ground-truth ~/vexir2vec.model ~/x86-data-all 10 'findutils diffutils'```

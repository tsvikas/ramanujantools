# Publications Relevance Analysis

Analysis of Ramanujan Machine publications for relevance to the ramanujantools package.

## Summary Table

| Paper | arXiv ID | Relevance | Priority | Key Connection |
|-------|----------|-----------|----------|----------------|
| Conservative Matrix Fields (2025) | 2507.08138 | **HIGH** | Must Include | Core CMF theory and algorithms |
| Nature Paper (2021) | 1907.00205 | **HIGH** | Must Include | Foundational paper for the project |
| PNAS Paper (2024) | 2308.11829 | **HIGH** | Must Include | CMF introduction and Apéry proof generalization |
| Conservative Matrix Field (2023) | 2303.09318 | **HIGH** | Must Include | Core CMF theoretical framework |
| Euler PCF (2023) | 2308.02567 | **HIGH** | Must Include | EulerSolver theoretical foundation |
| Arnold Journal (2021) | 2111.04468 | **HIGH** | Must Include | Irrationality measures and kamidelta algorithm |
| Euler2AI (2025) | 2502.17533 | **HIGH** | Recommended | Demonstrates package usage at scale |
| Unsupervised Discovery (2024) | 2412.16818 | **HIGH** | Recommended | Uses convergence-based methods from package |
| Ramanujan Library (2024) | 2412.12361 | **PARTIAL** | Optional | PSLQ method overlap |
| Integer Sequences (2022) | 2212.09470 | **PARTIAL** | Optional | Related work, different methodology |
| ASyMOB (2025) | 2505.23851 | **NO** | Exclude | AI benchmarking, unrelated |

---

## Detailed Analysis

### Papers to INCLUDE (Must-Have References)

#### 1. **Nature Paper: "The Ramanujan Machine" (2021)**
- **arXiv**: 1907.00205
- **DOI**: 10.1038/s41586-021-03229-4
- **Relevance**: Foundational paper introducing the Ramanujan Machine project
- **Connection**: ramanujantools is the official implementation library for this research
- **Implementation**:
  - PCF structures for discovering formulas
  - Matrix representations
  - Linear recurrences
- **Recommendation**: Primary citation, add to README.md and CITATION.cff

#### 2. **PNAS: "Algorithm-assisted Discovery..." (2024)**
- **arXiv**: 2308.11829
- **DOI**: 10.1073/pnas.2321440121
- **Relevance**: Introduces Conservative Matrix Fields as core framework
- **Connection**: CMF module implements this theory directly
- **Implementation**:
  - `CMF` class with conservation validation
  - Apéry's ζ(3) proof implementation
  - Irrationality measure computations
- **Recommendation**: Main theoretical reference for CMF module

#### 3. **arXiv: "Conservative Matrix Fields: Continuous Asymptotics" (2025)**
- **arXiv**: 2507.08138
- **Relevance**: Extends CMF theory to higher dimensions
- **Connection**: Describes algorithms already implemented
- **Implementation**:
  - D-finite CMF framework
  - Coboundary transformations
  - Poincaré-Perron asymptotics (kamidelta)
- **Recommendation**: Advanced CMF reference

#### 4. **arXiv: "The Conservative Matrix Field" (2023)**
- **arXiv**: 2303.09318
- **Author**: Ofir David (Ramanujan Machine group)
- **Relevance**: Core CMF theoretical paper
- **Connection**: Defines exact mathematical structure in package
- **Implementation**:
  - Conservation property validation
  - FFbar construction methods
  - Path-independent work calculations
- **Recommendation**: Primary CMF theory reference, cite in cmf.py docstring

#### 5. **arXiv: "On Euler Polynomial Continued Fractions" (2023)**
- **arXiv**: 2308.02567v2
- **Author**: Ofir David
- **Relevance**: Direct implementation in EulerSolver
- **Connection**: Author committed code one month before arXiv submission
- **Implementation**:
  - `EulerSolver.solve_for(a, b)`
  - Transformation to infinite sum form
  - Algorithm matches paper exactly
- **Recommendation**: Reference in euler_solver.py and solvers/README.md

#### 6. **Arnold Journal: "Irrationality Measures and PCFs" (2021)**
- **arXiv**: 2111.04468
- **DOI**: 10.1007/s40598-024-00250-z
- **Relevance**: Theoretical foundation for irrationality measures
- **Connection**: kamidelta algorithm implementation
- **Implementation**:
  - `Matrix.kamidelta()` prediction algorithm
  - `Limit.delta(L)` irrationality measure
  - Eigenvalue analysis for convergence rates
- **Recommendation**: Reference in delta-related methods

---

### Papers to INCLUDE (Recommended)

#### 7. **NeurIPS: "From Euler to AI" (2025)**
- **arXiv**: 2502.17533
- **Conference**: NeurIPS 2025
- **Relevance**: Demonstrates ramanujantools at scale
- **Connection**: Uses package for formula unification (360+ formulas)
- **Usage**:
  - Coboundary equivalence checking
  - Euler form transformations
  - Symbolic verification pipeline
- **Recommendation**: Showcase real-world research impact

#### 8. **NeurIPS: "Unsupervised Discovery of Formulas" (2024)**
- **arXiv**: 2412.16818
- **Conference**: NeurIPS 2024
- **Relevance**: Uses convergence-based metrics from package
- **Connection**: kamidelta algorithm for pattern detection
- **Scale**: 1.7M+ formulas analyzed
- **Recommendation**: Demonstrates computational capabilities

---

### Papers to INCLUDE (Optional Context)

#### 9. **ICLR: "The Ramanujan Library" (2024)**
- **arXiv**: 2412.12361
- **Conference**: ICLR 2025
- **Relevance**: Shares PSLQ algorithm implementation
- **Connection**: `Limit.identify()` uses same method
- **Distinction**: Creates database/API vs. computational tools
- **Recommendation**: Optional related work citation

#### 10. **arXiv: "Integer Sequences" (2022)**
- **arXiv**: 2212.09470
- **Relevance**: Related work from same research group
- **Connection**: Complementary ESMA algorithm (not implemented)
- **Distinction**: Discovery via pattern recognition vs. analysis tools
- **Recommendation**: Optional background reference

---

### Papers to EXCLUDE

#### 11. **ASyMOB Benchmark (2025)**
- **arXiv**: 2505.23851
- **Relevance**: None
- **Reason**: AI/ML benchmarking paper, completely different domain
- **Recommendation**: Do not include

---

## Implementation Recommendations

### 1. Update CITATION.cff

Add a `references` section with key papers:

```yaml
references:
  - type: article
    title: "Generating conjectures on fundamental constants with the Ramanujan Machine"
    authors:
      - family-names: "Raayoni"
        given-names: "Gal"
      # ... additional authors
    journal: "Nature"
    volume: 590
    year: 2021
    start: 67
    end: 73
    identifiers:
      - type: doi
        value: "10.1038/s41586-021-03229-4"
      - type: other
        value: "arXiv:1907.00205"

  - type: article
    title: "Algorithm-assisted discovery of an intrinsic order among mathematical constants"
    authors:
      - family-names: "Elimelech"
        given-names: "Rotem"
      # ... additional authors
    journal: "Proceedings of the National Academy of Sciences"
    volume: 121
    year: 2024
    identifiers:
      - type: doi
        value: "10.1073/pnas.2321440121"
      - type: other
        value: "arXiv:2308.11829"
```

### 2. Create docs/references.md

```markdown
# References

## Foundational Papers

### The Ramanujan Machine (2021)
Raayoni G., Gottlieb S., Manor Y., et al. "Generating conjectures on fundamental
constants with the Ramanujan Machine." *Nature* 590, 67-73 (2021).
- [Nature](https://doi.org/10.1038/s41586-021-03229-4)
- [arXiv:1907.00205](https://arxiv.org/abs/1907.00205)

Introduces the Ramanujan Machine project and discovery algorithms. ramanujantools
provides the mathematical infrastructure for analyzing discovered formulas.

### Conservative Matrix Fields (2024)
Elimelech R., David O., et al. "Algorithm-assisted discovery of an intrinsic order
among mathematical constants." *PNAS* 121, e2321440121 (2024).
- [PNAS](https://doi.org/10.1073/pnas.2321440121)
- [arXiv:2308.11829](https://arxiv.org/abs/2308.11829)

Introduces Conservative Matrix Fields (CMFs) as the unifying framework. The
`ramanujantools.cmf` module implements this theory.

## Theoretical Foundations

### CMF Theory (2023)
David O. "The conservative matrix field." arXiv:2303.09318 (2023).
- [arXiv](https://arxiv.org/abs/2303.09318)

Formalizes the CMF mathematical structure with conservation property validation
and path-independent calculations.

### Euler PCF (2023)
David O. "On Euler polynomial continued fractions." arXiv:2308.02567 (2023).
- [arXiv](https://arxiv.org/abs/2308.02567)

Describes the EulerSolver algorithm for transforming PCFs to infinite sum form.

### Irrationality Measures (2021)
Ben David N., et al. "On the Connection Between Irrationality Measures and
Polynomial Continued Fractions." *Arnold Mathematical Journal* (2021).
- [Journal](https://doi.org/10.1007/s40598-024-00250-z)
- [arXiv:2111.04468](https://arxiv.org/abs/2111.04468)

Theoretical foundation for the kamidelta algorithm and delta prediction methods.

## Applications and Extensions

### Euler2AI (2025)
Raz T., et al. "From Euler to AI: Unifying Formulas for Mathematical Constants."
*NeurIPS* (2025).
- [arXiv:2502.17533](https://arxiv.org/abs/2502.17533)

Demonstrates ramanujantools usage for large-scale formula unification (360+ formulas).

### Unsupervised Discovery (2024)
Shalyt M., et al. "Unsupervised Discovery of Formulas for Mathematical Constants."
*NeurIPS* (2024).
- [arXiv:2412.16818](https://arxiv.org/abs/2412.16818)

Uses convergence-based metrics from ramanujantools for pattern detection in 1.7M+ formulas.
```

### 3. Update README.md

Add a "Publications" section:

```markdown
## Publications

This package implements mathematical frameworks developed by the Ramanujan Machine research group:

**Core Theory:**
- [Nature 2021](https://doi.org/10.1038/s41586-021-03229-4): Foundational paper introducing the Ramanujan Machine
- [PNAS 2024](https://doi.org/10.1073/pnas.2321440121): Conservative Matrix Fields framework
- [arXiv:2303.09318](https://arxiv.org/abs/2303.09318): CMF mathematical theory

**Algorithms:**
- [arXiv:2308.02567](https://arxiv.org/abs/2308.02567): Euler PCF solver
- [arXiv:2111.04468](https://arxiv.org/abs/2111.04468): Irrationality measures

See [docs/references.md](docs/references.md) for complete bibliography.
```

### 4. Add Docstring References

In specific modules:

**ramanujantools/cmf/cmf.py:**
```python
class CMF:
    r"""
    Represents a Conservative Matrix Field (CMF).

    A CMF is defined by a set of axes and their relevant matrices that satisfy
    the conservation property: for every two axes x and y,
    $Mx(x, y) \cdot My(x+1, y) = My(x, y) \cdot Mx(x, y+1)$.

    References:
        - David O. (2023). "The conservative matrix field." arXiv:2303.09318
        - Elimelech R., et al. (2024). "Algorithm-assisted discovery..."
          PNAS 121, e2321440121
    """
```

**ramanujantools/solvers/euler_solver.py:**
```python
class EulerSolver:
    r"""
    A solver for identifying Euler polynomial continued fractions.

    Given polynomials a(x), b(x), finds h₁(x), h₂(x), f(x) such that:
        b(x) = -h₁(x)h₂(x)
        f(x)a(x) = f(x-1)h₁(x) + f(x+1)h₂(x+1)

    References:
        David O. (2023). "On Euler polynomial continued fractions."
        arXiv:2308.02567
    """
```

---

## Summary

**Must Include (6 papers):**
1. Nature 2021 (foundational)
2. PNAS 2024 (CMF introduction)
3. arXiv:2507.08138 (CMF extensions)
4. arXiv:2303.09318 (CMF theory)
5. arXiv:2308.02567 (Euler solver)
6. arXiv:2111.04468 (irrationality measures)

**Recommended (2 papers):**
7. arXiv:2502.17533 (NeurIPS 2025, scale demonstration)
8. arXiv:2412.16818 (NeurIPS 2024, convergence methods)

**Optional (2 papers):**
9. arXiv:2412.12361 (PSLQ overlap)
10. arXiv:2212.09470 (related work)

**Exclude (1 paper):**
11. arXiv:2505.23851 (unrelated)

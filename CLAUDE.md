# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`ramanujantools` is a Python research library for exploring Polynomial Continued Fractions (PCFs), Linear Recurrences, and Conservative Matrix Fields (CMFs). The library is used to discover new mathematical conjectures for constants like e, π, Catalan's constant, Euler's constant, and Riemann Zeta function values.

## Development Commands

### Environment Setup
```bash
# Install package with dev dependencies
pip install .[dev]
```

### Testing
```bash
# Run all tests (discovers *_test.py files in ramanujantools/)
pytest

# Run a specific test file
pytest ramanujantools/matrix_test.py

# Run a specific test function
pytest ramanujantools/matrix_test.py::test_function_name
```

### Linting
```bash
# Check for syntax errors and undefined names
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Check all linting rules (warnings only)
flake8 . --count --exit-zero --max-line-length=127 --statistics
```

### Building
```bash
# Build package (uses hatchling backend)
pip install build
python -m build
```

## Code Architecture

### Core Mathematical Objects

The library is organized around several core mathematical abstractions:

1. **Matrix** (`matrix.py`): Extends sympy.Matrix with symbolic manipulation capabilities. All matrix operations support substitution via `subs()` which internally uses `xreplace()` for performance. Matrices can be evaluated numerically using the `walk()` method with trajectories and starting positions.

2. **Position** (`position.py`): Represents a point in multi-dimensional space, used as substitution dictionaries for matrix evaluation.

3. **LinearRecurrence** (`linear_recurrence.py`): Represents linear recurrences of the form Σa_i(n)p(n-i) = 0. Can be constructed from coefficient lists or companion matrices. The `normalize()` method removes common factors.

4. **Limit** (`limit.py`): Encapsulates convergence calculation results from matrix walks. Stores current and previous matrices along with initial values and final projection to extract p/q limit values.

### Domain-Specific Modules

#### PCF (Polynomial Continued Fractions)
- `pcf/pcf.py`: Main PCF class representing continued fractions with polynomials a_n and b_n
- `pcf/hypergeometric.py`: Hypergeometric limit specializations (1F1, 2F1)
- PCFs can be constructed from (a_n, b_n) pairs, LinearRecurrence objects, or 2x2 matrices
- The `deflate()` method removes common factors from a_n and b_n

#### CMF (Conservative Matrix Fields)
- `cmf/cmf.py`: Main CMF class for multi-dimensional matrix fields satisfying Mx(x,y)·My(x+1,y) = My(x,y)·Mx(x,y+1)
- `cmf/ffbar.py`: FFbar construction using f(x,y) and f̄(x,y) functions
- `cmf/d_finite.py`: D-finite sequence support
- `cmf/pfq.py`: Generalized hypergeometric functions
- `cmf/meijer_g.py`: Meijer G-function representations
- `cmf/known_cmfs.py`: Pre-defined CMFs for constants (e, π, ζ(3), etc.)
- CMFs validate conservation automatically during construction unless `validate=False`

#### Solvers
- `solvers/euler_solver.py`: Finds Euler family PCF representations that can be transformed into infinite sum expansions
- `solvers/coboundary_solver.py`: Finds coboundary equivalence U(x) satisfying M1(x)·U(x+1) = U(x)·M2(x)
- `solvers/ffbar_solver.py`: Solves FFbar conditions and converts PCFs to FFbar constructions

### FLINT Integration

The `flint_core/` module provides high-performance polynomial arithmetic via python-flint:

- **FlintContext** (`context.py`): Context managers for fmpz (integer) or fmpq (rational) polynomial rings
- **SymbolicMatrix** and **NumericMatrix** (`symbolic_matrix.py`, `numeric_matrix.py`): Fast matrix operations using FLINT
- Use `flint_from_sympy()` and `flint_to_sympy()` for conversion between sympy and FLINT representations
- Matrix equality checks and CMF validation use FLINT for performance

### Important Conventions

- The symbol `n` is reserved for PCF/recurrence operations and should not be used as a CMF axis
- Matrix substitution uses `xreplace()` internally (not `subs()`) for performance
- Test files follow the pattern `*_test.py`
- Matrices are validated for square dimensions and conservation properties where applicable
- Symbolic computations use sympy; numeric evaluations use mpmath for high precision

### Performance Utilities

- `utils/batched.py`: Batching support for parallel computations via the `Batchable` protocol
- Matrix walks support batched evaluation for performance
- Use `@cached_property` and `@lru_cache` decorators for expensive computations

## Python Version

Requires Python >= 3.11

## Key Dependencies

- sympy: Symbolic mathematics
- mpmath: High-precision arithmetic
- python-flint: Fast polynomial arithmetic
- gmpy2: Multiple precision arithmetic
- numpy: Numerical arrays
- tqdm: Progress bars for long computations

## Research Context

This package is the computational engine of the **Ramanujan Machine** research project at Technion. Understanding this context is crucial for working with the code effectively.

### Project Role

- **NOT** a discovery engine—it's the **analysis and verification infrastructure**
- Discovery algorithms (MITM, Gradient Descent) find patterns → ramanujantools analyzes structures
- Used in peer-reviewed research (Nature 2021, PNAS 2024, NeurIPS 2024/2025, ICLR 2025)
- Has processed millions of formulas at scale

### Key Theoretical Concepts

**Conservative Matrix Fields (CMFs)**:
- Multi-dimensional matrix fields with path-independent multiplication
- Unifies thousands of different-looking PCF formulas as 1D slices of higher-dimensional structures
- Generalizes Apéry's 1979 proof technique for ζ(3) irrationality

**Irrationality Measures (Delta)**:
- Quantifies approximation quality: |p/q - L| = 1/q^(1+δ)
- The **kamidelta algorithm** predicts δ from eigenvalue analysis WITHOUT knowing target constant L
- Used for unsupervised clustering of 1.7M+ formulas

**Euler Family PCFs**:
- Special PCFs transformable to infinite sum representations
- Conditions: b(x) = -h₁(x)h₂(x) and f(x)a(x) = f(x-1)h₁(x) + f(x+1)h₂(x+1)
- EulerSolver finds these transformations

**Coboundary Equivalence**:
- Two PCFs are equivalent if related by gauge transformation U
- M₁(n)·U(n+1) = U(n)·M₂(n)
- Used to prove different formulas are actually the same

### Important Algorithms

**kamidelta()** - Predict irrationality measure:
- Uses Poincaré characteristic polynomial for eigenvalue analysis
- Performs GCD slope fitting on sequence denominators
- Returns delta prediction: δ = -1 + error/slope
- Found in: Matrix.kamidelta(), PCF.kamidelta(), LinearRecurrence.kamidelta()

**walk()** - Matrix trajectory multiplication:
- Computes ∏M(start + i·trajectory) for i=0..iterations
- Automatic routing to FLINT (SymbolicMatrix) or mpmath (NumericMatrix) for performance
- Supports batched iterations for efficiency

**identify()** - PSLQ-based constant identification:
- Uses Limit.identify(L) to find integer relations
- Searches for vectors a,b where a·p/b·p ≈ L
- Core method for validating discovered formulas

### Code Organization Principles

**Dual Representation**:
- Objects maintain both symbolic (sympy/FLINT) and numeric (mpmath) forms
- Automatic selection based on input types

**Conservation Validation**:
- CMFs validate Mx(x,y)·My(x+1,y) = My(x,y)·Mx(x,y+1) at construction
- Uses FLINT for performance (10-100x faster than sympy)
- Can disable with validate=False for performance

**Batching Pattern**:
- Methods decorated with @batched accept scalar OR list inputs
- walk([10, 20, 30]) reuses intermediate computations
- Essential for performance at scale

## Documentation Files

For deeper understanding:
- **docs/theoretical-foundations.md**: Mathematical theory, research context, algorithm explanations
- **docs/package-overview.md**: Comprehensive API and architecture reference
- **docs/publications-analysis.md**: Related papers and their relevance
- **ramanujantools/solvers/README.md**: Detailed solver algorithms with mathematical definitions

## Common Patterns

**Creating a PCF and analyzing convergence**:
```python
from ramanujantools.pcf import PCF
pcf = PCF(a_n=n, b_n=-(n-1)*n)
pcf = pcf.deflate_all()  # Remove common factors
limits = pcf.limit([10, 20, 30], start=1)  # Batched evaluation
delta = pcf.kamidelta()  # Predict irrationality measure
```

**Working with CMFs**:
```python
from ramanujantools.cmf import CMF
from ramanujantools import Matrix
cmf = CMF({x: Mx, y: My})  # Validates conservation automatically
work = cmf.work(start={x:0, y:0}, end={x:5, y:3})  # Path-independent
traj_matrix = cmf.trajectory_matrix(trajectory={x:1, y:1}, start={x:0, y:0})
```

**Solving for Euler form**:
```python
from ramanujantools.solvers import EulerSolver
solutions = EulerSolver.solve_for(a, b)
for sol in solutions:
    print(f"h1={sol.h_1}, h2={sol.h_2}, f={sol.f}")
```

## Development Notes

- **Performance critical code**: Matrix walks, CMF validation, eigenvalue computation
- **Testing pattern**: Each module has *_test.py with pytest-compatible tests
- **Benchmarking**: matrix_benchmark.py and cmf_benchmark.py for performance testing
- **Known constants**: cmf/known_cmfs.py contains pre-computed CMFs (e, π, ζ(3), etc.)
- **Authors**: Code committed by Ofir David matches papers (2303.09318, 2308.02567)

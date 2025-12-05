# Ramanujantools Tutorial: From Basics to Research Applications


A hands-on guide to learning the ramanujantools package from scratch.


## Table of Contents

1. [Introduction](#introduction)
2. [Module 1: Setup and First Steps](#module-1-setup-and-first-steps)
3. [Module 2: Matrix Fundamentals](#module-2-matrix-fundamentals)
4. [Module 3: Polynomial Continued Fractions](#module-3-polynomial-continued-fractions)
5. [Module 4: Irrationality Measures](#module-4-irrationality-measures)
6. [Module 5: Conservative Matrix Fields](#module-5-conservative-matrix-fields)
7. [Module 6: Complete Analysis Workflow](#module-6-complete-analysis-workflow)
8. [Quick Reference](#quick-reference)


## Introduction


### What is ramanujantools?

Ramanujantools is a Python research library developed by the Ramanujan Machine project at Technion. It provides tools for:

- **Polynomial Continued Fractions (PCFs)**: Discovering formulas for mathematical constants like e, π, ζ(3)
- **Conservative Matrix Fields (CMFs)**: Multi-dimensional generalizations of PCFs
- **Linear Recurrences**: Sequences satisfying polynomial recurrence relations
- **Irrationality Measures**: Analyzing convergence quality and proving irrationality

### Who should use this tutorial?

- Mathematicians exploring continued fractions
- Researchers in number theory and special functions
- Anyone interested in computational mathematics
- Students learning about mathematical constants

### Prerequisites:

- Basic Python knowledge
- Familiarity with mathematical notation (matrices, continued fractions helpful but not required)
- SymPy awareness is helpful


editor notes start and end with <<<, you shold fix them
eveloper notes start and end with ^^^, you should keep them in the docs - they are question to the developer team


## Module 1: Setup and First Steps


### Installation


First, install the package:

<!-- #region -->
<<<
editor notes: encorage the user to install the package in a venv, managed with one of the common tools today (list them)
than give examples with uv
<<<

```bash 
pip install ramanujantools
```

For development with test dependencies: <<< this was changed, will not work anymore. fix here and in the README/CLAUDE file <<<

```bash
pip install ramanujantools[dev]
```
<!-- #endregion -->

### Basic Imports


Import the core classes you'll use throughout:

```python
# Core imports - you'll use these constantly
from ramanujantools import Position, Matrix, Limit, LinearRecurrence, PCF, CMF, known_cmfs

import mpmath

# Sympy for symbolic math
import sympy as sp
from sympy.abc import x, y, z, n

# Set up nice printing
sp.init_printing()

print(f"Sympy version: {sp.__version__}")
```

<<< we need a short ibtro to sympy and to mpf.


### Lesson 1.1: Position Objects


Position objects represent points in multi-dimensional space. They're used throughout the package for matrix substitutions.


#### Basic Operations:

```python
# Create a simple position
p1 = Position({x: 5, y: 3})
print(p1)
print(repr(p1))
p1
```

```python
# Vector addition
p2 = Position({x: 2, y: 7})
p_sum = p1 + p2
print("Sum:", p_sum)  # Position({x: 7, y: 10})
```

```python
# Scalar multiplication
p_scaled = 3 * p1
print("Scaled:", p_scaled)  # Position({x: 15, y: 9})
```

```python
Position({x: 5, y: 3, z: -6}).longest()  # 6
```

```python
Position({x: 5, y: 3, z: -6}).shortest()  # 3
```

```python
Position({x: 5, y: -3, z: 0}).signs()  # {x: 1, y: -1, z: 0}
```

```python
# equlaity works
Position({x:1}) == Position({x:1})
```

```python
# notice that this doesnt work. ^^^ should we fix it? or at least verify that the keys are sympy symbols? ^^^
Position(x=1) == Position({x:1})
```

#### Symbolic Positions:

```python
# Positions can contain symbolic expressions
p_symbolic = Position({x: n**2 + 3, y: 2*n})
p_symbolic
```

```python
# Check if it's polynomial
p_symbolic.is_polynomial()  # True
```

```python
# Check if it's integer
p_integer = Position({x: 5, y: -7})
p_integer.is_integer()  # True
```

**Key Takeaway:** Positions are like dictionaries that support mathematical operations. They're the foundation for matrix substitutions.


## Module 2: Matrix Fundamentals


### Lesson 2.1: Matrix Basics


The Matrix class extends sympy.Matrix with specialized functionality for continued fractions and convergence analysis.


<<<
OK, so what do I need to know about sp.Matrix? 
some of it is below - we need to be clear what is from sp and what did we add
<<<

^^^why extend and not use directly? what is this adding to us?^^^


#### Creating Matrices:

```python
# Basic matrix
m = Matrix([[1, 2], [3, 4]])
print(m)
m
```

```python
# Matrix properties
print(f"Is square? {m.is_square()}")
print(f"Determinant: {m.det()}")
```

#### Symbolic Matrices:

```python
# Create a symbolic matrix
m_symbolic = Matrix([[x, x**2], [1, x + 1]])
m_symbolic
```

```python
m_symbolic.det()
```

```python
# Substitute values (this uses fast xreplace internally)
m_symbolic.subs({x: 5})

```

```python
# Try different values
for val in [1, 2, 10]:
    result = m_symbolic.subs({x: val})
    print(f"At x={val}: det = {result.det()}")
```

### Lesson 2.2: The Walk Operation


**This is the most important operation in the entire package.**

The `walk()` method computes the product of matrices along a trajectory:

**M(start) · M(start+trajectory) · M(start+2·trajectory) · ... · M(start+n·trajectory)**


#### Single Walk Example:

```python
# Simple 1D example with variable n
m = Matrix([[0, n], [1, n]])  # This matrix is crucial for PCFs!

# Walk with trajectory n→n+1, starting at n=1, for 5 steps
# This computes: M(1) · M(2) · M(3) · M(4) · M(5)
m.walk(
    trajectory={n: 1},    # Increment n by 1 each step
    iterations=5,         # Do 5 multiplications
    start={n: 1}          # Start at n=1
)
#^^^ why not call it start-step-(count?)
```

#### Batched Walks (Efficient!):


Instead of calling `walk()` multiple times, use a list of iterations. This reuses intermediate results and is **much faster**.

```python
m = Matrix([[0, n], [1, n]])
depths = [5, 10, 20, 50]

# Evaluate at depths 5, 10, 20, 50 all at once
results = m.walk(
    trajectory={n: 1},
    iterations=depths,
    start={n: 1}
)
{d: r for d, r in zip(depths, results)}
```

```python
# <<< any way to calulate the ratio with mpf for perscision?
{d: sp.N(r[1,1]/r[0,1]) for d, r in zip(depths, results)}
# <<< it can be proven that this series relates to the PCF `...` and converge to e. 
# <<< other important known Matrices are ...
```

**Multi-dimensional Walks:**

```python
# Matrices can depend on multiple variables
m = Matrix([[x + y, x*y], [1, x]])

# Walk along x-axis only (y fixed)
result_x = m.walk({x: 1, y: 0}, 3, {x: 0, y: 5})
print("Walk along x-axis (y=5 fixed):")
print(result_x)

# Walk along y-axis only (x fixed)
result_y = m.walk({x: 0, y: 1}, 3, {x: 2, y: 0})
print("\nWalk along y-axis (x=2 fixed):")
print(result_y)

# Diagonal walk (both increase together)
result_diag = m.walk({x: 1, y: 1}, 3, {x: 0, y: 0})
print("\nDiagonal walk (both increase):")
print(result_diag)
```

**Key Takeaway:** The `walk()` operation is how we compute continued fractions and analyze convergence. Always use batched evaluation with lists for efficiency.


## Module 3: Polynomial Continued Fractions


PCFs are the primary research objects in ramanujantools. They represent continued fractions where the terms are polynomials.


### Mathematical Background


A **Polynomial Continued Fraction (PCF)** has the form:

```
a₀ + b₁/(a₁ + b₂/(a₂ + b₃/(a₃ + ...)))
```
<<< this should be a nice latex

where $a_n$ and $b_n$ are polynomials in $n$.


### Lesson 3.1: The Golden Ratio


The simplest possible PCF is $a_n = b_n = 1$ i.e.:

```python
# we represent this PCF with PCF(1, 1)
golden = PCF(1, 1)
golden
```

since the limit fulfill $φ = 1 + \cfrac{1}{φ}$ it converges to the golden ratio φ.

```python
golden.a_n, golden.b_n
```

```python
# Evaluate at increasing depths
depths = [5, 10, 20, 50]
limits = golden.limit(iterations=depths, start=1)
{d: limit.as_float() for d, limit in zip(depths, limits)}
```

```python
# Compare with actual golden ratio
(1 + 5**0.5) / 2
```

### Lesson 3.2: The Famous e Formula


One of the most beautiful continued fractions converges to Euler's number e.

```python
pcf_e = PCF(n, n)
pcf_e
```

```python
# Evaluate with batched iterations
depths = [10, 20, 50, 100]
limits = pcf_e.limit(depths, start=1)
{d: f"{1/limit.as_float()} ({limit.precision()} digits accurate)" for d, limit in zip(depths, limits)}
```

```python
1/limits[-1].as_float() + 1 - mpmath.e 
```

```python
print("\nConvergence to e:")
for i, depth in enumerate(depths):
    value = float(limits[i].as_float())
    precision = limits[i].precision()
    print(f"Depth {depth:3d}: {value:.15f} ({precision:.1f} digits accurate)")

# Compare with actual e
actual_e = float(mpmath.e)
print(f"\nActual e:   {actual_e:.15f}")

# Check error at depth 100
error = abs(float(limits[-1].as_float()) - actual_e)
print(f"Error at depth 100: {error:.2e}")
```

### Lesson 3.3: More Complex PCFs

Explore different polynomial patterns:

```python
# Example 1: Linear in numerator, quadratic in denominator
pcf1 = PCF(a_n=2*n + 1, b_n=n**2)
print("PCF(2n+1, n²):")
limit1 = pcf1.limit(100, start=1)
print(f"Converges to: {limit1.as_float()}")

# Example 2: Different pattern
pcf2 = PCF(a_n=6*n - 1, b_n=-n*(2*n - 1))
print("\nPCF(6n-1, -n(2n-1)):")
limit2 = pcf2.limit(100, start=1)
print(f"Converges to: {limit2.as_float()}")

# Check the degrees
print(f"\nDegrees of pcf1: {pcf1.degrees()}")  # (deg(a_n), deg(b_n))
print(f"Degrees of pcf2: {pcf2.degrees()}")
```

### Lesson 3.4: Deflation - Cleaning Up PCFs

PCFs often have common factors that should be removed for canonical form.

```python
# Create a PCF with common factors
pcf_messy = PCF(a_n=2*n**2 + 4*n, b_n=2*n**3)
print("Before deflation:")
print(f"a_n = {pcf_messy.a_n}")
print(f"b_n = {pcf_messy.b_n}")

# Remove common factors
pcf_clean = pcf_messy.deflate_all()
print("\nAfter deflate_all():")
print(f"a_n = {pcf_clean.a_n}")
print(f"b_n = {pcf_clean.b_n}")

# They converge to the same value
limit_messy = pcf_messy.limit(50, start=1)
limit_clean = pcf_clean.limit(50, start=1)

print(f"\nMessy converges to: {limit_messy.as_float()}")
print(f"Clean converges to: {limit_clean.as_float()}")
print("Same value! ✓")
```

### Lesson 3.5: Understanding PCF Components

```python
# PCFs have several useful methods and properties
pcf = PCF(a_n=n**2 + n, b_n=-n**3)

# Get the underlying matrix representation
M = pcf.M()
print("Matrix representation M(n):")
print(M)

# Get the initial matrix A (for a₀ term)
A = pcf.A()
print("\nInitial matrix A:")
print(A)

# Find singular points (where things might break)
singular = pcf.singular_points()
print(f"\nSingular points: {singular}")

# The matrix [[0, b_n], [1, a_n]] is fundamental to PCF theory
```

**Key Takeaway:** Always use `deflate_all()` before analysis to get the canonical form of a PCF.

---

## Module 4: Irrationality Measures

### Mathematical Background

The **irrationality measure** δ quantifies how well a number can be approximated by rationals:

**|p/q - L| ≈ 1/q^(1+δ)**

- Higher δ means better approximations
- Can be used to prove irrationality
- δ = 0 for rational numbers
- δ > 0 for irrational numbers

### Lesson 4.1: Calculate Delta

When you know the target constant, you can calculate δ directly:

```python
import mpmath
mpmath.mp.dps = 100  # Set precision for calculations

# Use our e formula
pcf_e = PCF(a_n=n, b_n=n)

# Calculate delta when we KNOW the target (e)
depth = 500
delta_value = pcf_e.delta(depth=depth, L=mpmath.e, start=1)

print(f"Irrationality measure δ for PCF(n,n) → e:")
print(f"δ ≈ {delta_value}")
print(f"\nInterpretation: The denominators q_n grow exponentially")
print(f"and error decays like 1/q_n^(1+{delta_value:.3f})")
```

### Lesson 4.2: The Kamidelta Algorithm (Blind Prediction!)

This is one of the unique features of ramanujantools: predict δ **without knowing** what constant the PCF converges to!

**How it works:**
1. Analyzes eigenvalues of the companion matrix
2. Measures GCD growth of denominators
3. Predicts δ from the ratio

```python
# Create a PCF (we don't know what it converges to)
pcf = PCF(a_n=2*n + 1, b_n=n**2)

# Predict delta using only the algebraic structure
delta_predicted = pcf.kamidelta(depth=100)

print("Kamidelta prediction (no target constant needed!):")
print(f"Predicted δ values: {delta_predicted}")
print(f"\nThis predicts convergence quality from eigenvalues alone!")

# Compare with actual limit
limit = pcf.limit(200, start=1)
print(f"\nActual convergence value: {limit.as_float()}")
```

**Why This Matters:**

The kamidelta algorithm was used in research to classify 1.7 million formulas by convergence patterns without needing to know their target constants. This is published in NeurIPS 2024.

### Lesson 4.3: PSLQ - Identify Unknown Constants

Given a PCF that converges to something unknown, use the PSLQ algorithm to identify it:

```python
# This converges to ln(2)
pcf_mystery = PCF(a_n=2*n - 1, b_n=-n**2)

# Evaluate to high precision
limit = pcf_mystery.limit(1000, start=1)
value = limit.as_float()

print(f"Mystery PCF converges to: {value}")

# Try to identify it as ln(2)
import mpmath
result = limit.identify(mpmath.log(2))

if result is not None:
    print(f"\n✓ Identified as related to ln(2)!")
    print(f"Relation: {result}")
else:
    print("\nCouldn't identify (try higher precision)")
```

**Key Takeaway:** The kamidelta algorithm is unique to this package and represents cutting-edge research in automated mathematical discovery.

---

## Module 5: Conservative Matrix Fields

### Mathematical Background

A **Conservative Matrix Field (CMF)** is a multi-dimensional matrix field where matrices satisfy a conservation property:

**M_x(x,y) · M_y(x+1,y) = M_y(x,y) · M_x(x,y+1)**

This means matrix multiplication is **path-independent** - you get the same result regardless of which path you take through the grid.

**Why This Matters:**
- PCFs are 1D slices of higher-dimensional CMFs
- Thousands of different-looking formulas are actually different trajectories through the same CMF
- Generalizes Apéry's 1979 proof technique for ζ(3) irrationality
- Published in PNAS 2024

### Lesson 5.1: Pre-defined CMFs

The easiest way to start is with pre-defined CMFs for famous constants:

```python
# CMF for e
cmf_e = known_cmfs.e()
print("CMF for e:")
print(f"Dimensions: {cmf_e.axes()}")
print(f"Rank (matrix size): {cmf_e.rank()}")

# CMF for π
cmf_pi = known_cmfs.pi()
print("\nCMF for π:")
print(f"Dimensions: {cmf_pi.axes()}")

# CMF for ζ(3) - Apéry's famous formula!
cmf_zeta3 = known_cmfs.zeta3()
print("\nCMF for ζ(3) (Apéry):")
print(f"Dimensions: {cmf_zeta3.axes()}")
```

### Lesson 5.2: Evaluate a CMF

CMFs are multi-dimensional, so we specify a trajectory through the space:

```python
cmf = known_cmfs.e()

# Define a trajectory (which way to walk in the grid)
trajectory = {x: 1, y: 1}  # Walk diagonally

# Define starting point
start = {x: 0, y: 0}

# Calculate limit with batched iterations
limits = cmf.limit(
    trajectory=trajectory,
    iterations=[50, 100, 200],  # Batched!
    start=start
)

print("CMF for e evaluated along (1,1) trajectory:")
for i, depth in enumerate([50, 100, 200]):
    value = limits[i].as_float()
    print(f"Depth {depth:3d}: {value}")
```

### Lesson 5.3: Apéry's ζ(3) Formula

This is the **famous formula that proved ζ(3) is irrational in 1979!**

```python
cmf_zeta3 = known_cmfs.zeta3()

# Get the trajectory matrix (reduces 2D CMF to 1D PCF)
trajectory = {x: 1, y: 1}
start = {x: 0, y: 0}

# This extracts the 1D continued fraction
traj_matrix = cmf_zeta3.trajectory_matrix(trajectory, start)

# Convert to PCF
pcf_apery = PCF(traj_matrix).deflate_all()

print("Apéry's ζ(3) formula:")
print(f"a_n = {pcf_apery.a_n}")
print(f"b_n = {pcf_apery.b_n}")

# This should give: a_n = 34n³ + 51n² + 27n + 5, b_n = -n⁶

# Evaluate it
limit = pcf_apery.limit(200, start=1)
print(f"\nConverges to: {limit.as_float()}")

# Compare with actual ζ(3)
import mpmath
actual_zeta3 = float(mpmath.zeta(3))
print(f"Actual ζ(3):  {actual_zeta3}")

print("\n✓ This PCF proved ζ(3) is irrational (Apéry, 1979)")
print("✓ CMF framework generalizes this to multiple dimensions (PNAS 2024)")
```

**Historical Context:**

Apéry's proof was a breakthrough after decades without progress. The CMF framework shows his formula is actually a trajectory through a 2D matrix field, and there are many related formulas from different trajectories.

### Lesson 5.4: Different Trajectories, Different Formulas

```python
# Same CMF, different trajectories give different formulas
cmf = known_cmfs.zeta3()

# Trajectory 1: Original Apéry direction
traj1 = {x: 1, y: 1}
pcf1 = PCF(cmf.trajectory_matrix(traj1, {x: 0, y: 0})).deflate_all()
print("Trajectory (1,1):")
print(f"a_n = {pcf1.a_n}")

# Trajectory 2: Different direction
traj2 = {x: 2, y: 1}
pcf2 = PCF(cmf.trajectory_matrix(traj2, {x: 0, y: 0})).deflate_all()
print("\nTrajectory (2,1):")
print(f"a_n = {pcf2.a_n}")

# Both converge to related values involving ζ(3)
print("\nBoth are valid formulas from the same CMF!")
```

**Key Takeaway:** CMFs unify thousands of formulas. What look like different formulas are often just different paths through the same mathematical structure.

---

## Module 6: Complete Analysis Workflow

This section shows a complete analysis workflow, combining all the techniques you've learned.

### Full PCF Analysis

```python
# Create a candidate PCF
pcf = PCF(a_n=5 + 10*n, b_n=1 - 9*n**2)

print("="*50)
print("COMPLETE PCF ANALYSIS")
print("="*50)

# Step 1: Clean it up
print("\n1. DEFLATION")
pcf_clean = pcf.deflate_all()
print(f"   Original: a_n={pcf.a_n}, b_n={pcf.b_n}")
print(f"   Cleaned:  a_n={pcf_clean.a_n}, b_n={pcf_clean.b_n}")

# Step 2: Evaluate convergence
print("\n2. CONVERGENCE")
limits = pcf_clean.limit([50, 100, 200, 500], start=1)
for i, depth in enumerate([50, 100, 200, 500]):
    val = float(limits[i].as_float())
    prec = limits[i].precision()
    print(f"   Depth {depth:3d}: {val:.10f} ({prec:.1f} digits)")

# Step 3: Predict irrationality measure
print("\n3. KAMIDELTA (blind prediction)")
try:
    delta_pred = pcf_clean.kamidelta(depth=200)
    print(f"   Predicted δ: {delta_pred}")
except:
    print("   (kamidelta failed - might need higher depth)")

# Step 4: Check degrees
print("\n4. STRUCTURE")
deg_a, deg_b = pcf_clean.degrees()
print(f"   Degree(a_n) = {deg_a}")
print(f"   Degree(b_n) = {deg_b}")
print(f"   Degree pattern: ({deg_a}, {deg_b})")

# Step 5: Try to identify the constant
print("\n5. IDENTIFICATION")
final_value = limits[-1].as_float()
print(f"   Converges to: {final_value}")

# Try common constants
import mpmath
candidates = {
    'π': mpmath.pi,
    'e': mpmath.e,
    'ln(2)': mpmath.log(2),
    'γ': mpmath.euler,
    'ζ(3)': mpmath.zeta(3)
}

for name, const in candidates.items():
    result = limits[-1].identify(const)
    if result is not None:
        print(f"   ✓ Related to {name}!")
        break

print("\n" + "="*50)
```

### Research-Level Analysis

For research applications, you would also:

1. **Check for Euler family membership:**
```python
from ramanujantools.solvers import EulerSolver

solutions = EulerSolver.solve_for(pcf_clean.a_n, pcf_clean.b_n)
if solutions:
    print("✓ PCF is in Euler family!")
    for sol in solutions:
        print(f"  h₁={sol.h_1}, h₂={sol.h_2}, f={sol.f}")
```

2. **Find FFbar construction:**
```python
from ramanujantools.solvers import FFbarSolver

ffbars = FFbarSolver.from_pcf(pcf_clean)
if ffbars:
    print("✓ Found FFbar representation!")
    for ffbar in ffbars:
        print(f"  f={ffbar.f}, f̄={ffbar.fbar}")
```

3. **Check coboundary equivalence with known formulas:**
```python
from ramanujantools.solvers import CoboundarySolver

# Compare with another PCF
pcf_other = PCF(a_n=..., b_n=...)
U = CoboundarySolver.find_coboundary(
    pcf_clean.M(),
    pcf_other.M(),
    max_deg=5
)
if U is not None:
    print("✓ PCFs are coboundary equivalent!")
```

---

## Quick Reference

### Essential Commands

```python
# POSITION (coordinates)
Position({x: 5, y: 3})           # Create position
p1 + p2, 3 * p1                  # Vector operations
p.longest(), p.shortest()        # Find extremes

# MATRIX (core object)
Matrix([[a, b], [c, d]])         # Create matrix
m.subs({x: 5})                   # Substitute values
m.walk(traj, iters, start)       # CORE OPERATION
m.limit(traj, iters, start)      # Get convergence

# PCF (continued fractions)
PCF(a_n, b_n)                    # Create PCF
pcf.limit(depth, start)          # Evaluate convergence
pcf.limit([10,20,50], start)     # Batched evaluation
pcf.deflate_all()                # Clean up factors
pcf.kamidelta(depth)             # Predict δ (blind!)
pcf.delta(depth, L)              # Calculate δ with target

# LIMIT (convergence results)
limit.as_float()                 # Get numeric value
limit.as_rational()              # Get exact fraction
limit.precision()                # Digits of accuracy
limit.delta(target)              # Irrationality measure
limit.identify(constant)         # PSLQ identification

# CMF (multi-dimensional)
known_cmfs.e(), .pi(), .zeta3()  # Pre-defined CMFs
cmf.limit(traj, iters, start)    # Evaluate CMF
cmf.trajectory_matrix(traj, st)  # Extract 1D slice
cmf.work(start, end)             # Path-independent
```

### Key Patterns

1. **Always use LISTS for iterations:** `[10, 20, 50]` for batched evaluation
2. **walk(trajectory, iterations, start)** appears everywhere
3. **deflate_all()** before analysis to get canonical form
4. **as_float()** to see numeric results
5. **Symbol `n` is reserved** for PCF operations

### Common Pitfalls

1. Symbol `n` is reserved for PCF/recurrence operations - don't use it in CMFs
2. Substitution uses `xreplace` internally (not full sympy `subs`)
3. Validation can be slow - use `validate=False` for experimentation
4. Matrix dimensions must be square for most operations
5. Starting positions must include all required axes

---

## Research Applications

### How This Package is Used

ramanujantools has been used in peer-reviewed research:

**Nature 2021:** Foundational Ramanujan Machine paper introducing algorithmic discovery of formulas

**PNAS 2024:** Conservative Matrix Fields framework showing thousands of formulas are related

**NeurIPS 2024:** Analyzed 1.7 million formulas using kamidelta for unsupervised clustering

**NeurIPS 2025:** Harvested 385 formulas from 455,050 arXiv papers, proved 360 equivalences

**Arnold Mathematical Journal 2024:** Irrationality measures and eigenvalue analysis

### The Discovery Pipeline

1. **Algorithmic Search** (MITM, Gradient Descent): Find numerical patterns
2. **ramanujantools Analysis:** Verify, deflate, analyze structure
3. **Theoretical Understanding:** Prove why it works, find equivalences
4. **Publication:** Contribute to mathematical knowledge

### What Makes This Special

- **Not just numerical:** Symbolic manipulation at scale
- **Unique algorithms:** Kamidelta, CMF validation, coboundary solving
- **Research infrastructure:** Used to analyze millions of formulas
- **Bridge:** Connects computational discovery with rigorous proof

---

## Next Steps

### Learning Path

1. **Start Simple:** Work through Modules 1-3 with simple PCFs
2. **Build Intuition:** Experiment with different polynomials in PCFs
3. **Explore CMFs:** Try pre-defined CMFs and different trajectories
4. **Read the Papers:** Understand the theoretical foundations
5. **Contribute:** Apply to your own research or contribute improvements

### Additional Resources

**Documentation:**
- `docs/package-overview.md` - Comprehensive API reference
- `docs/theoretical-foundations.md` - Mathematical background
- `docs/publications-analysis.md` - Research context
- `ramanujantools/solvers/README.md` - Solver algorithms

**Test Files for Examples:**
- `position_test.py` - Simple examples
- `matrix_test.py` - Core operations
- `pcf_test.py` - Main research objects
- `cmf_test.py` - Advanced structures
- `known_cmfs_test.py` - Pre-defined examples

**Key Papers:**
- [Nature 2021](https://doi.org/10.1038/s41586-021-03229-4): Ramanujan Machine introduction
- [PNAS 2024](https://doi.org/10.1073/pnas.2321440121): CMF framework
- [arXiv:2111.04468](https://arxiv.org/abs/2111.04468): Irrationality measures
- [arXiv:2303.09318](https://arxiv.org/abs/2303.09318): CMF mathematical theory

### Community

- **GitHub:** [RamanujanMachine/ramanujantools](https://github.com/RamanujanMachine/ramanujantools)
- **Website:** [ramanujanmachine.com](https://www.ramanujanmachine.com)
- **Issues:** Report bugs or request features on GitHub

---

## Exercises

Try these exercises to test your understanding:

**Exercise 1:** Find a PCF that converges to ln(2)
- Hint: Try simple polynomials and evaluate at different depths

**Exercise 2:** Extract Apéry's formula from the zeta3() CMF
- Use a different trajectory than (1,1)
- Compare the polynomials you get

**Exercise 3:** Analyze a PCF from literature
- Choose a formula from a paper
- Deflate it, evaluate convergence, predict delta

**Exercise 4:** Explore the e CMF
- Try trajectories (1,0), (0,1), (1,1), (2,1)
- See what different formulas you get

**Exercise 5:** Use kamidelta for classification
- Create 10 random PCFs
- Predict their delta values
- Group them by convergence quality

---

## Conclusion

You now have the foundation to:
- Create and analyze PCFs for mathematical constants
- Use the crucial `walk()` operation
- Calculate and interpret irrationality measures
- Work with multi-dimensional CMFs
- Understand how this package contributes to research

The ramanujantools package represents a new approach to mathematics: combining algorithmic discovery, symbolic analysis, and rigorous proof. You're now equipped to participate in this exciting field!

**Happy exploring!**

---

*Tutorial created for ramanujantools - The official symbolic and numeric research tools of the Ramanujan Machine project.*

*For questions, issues, or contributions, visit: https://github.com/RamanujanMachine/ramanujantools*

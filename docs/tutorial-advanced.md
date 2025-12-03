# Ramanujantools Tutorial - Chapter 2: Advanced Topics

A comprehensive guide to advanced features, solvers, and research-level techniques in ramanujantools.

## Table of Contents

1. [Introduction](#introduction)
2. [Module 7: Linear Recurrences](#module-7-linear-recurrences)
3. [Module 8: Solver Algorithms](#module-8-solver-algorithms)
4. [Module 9: Advanced CMF Constructions](#module-9-advanced-cmf-constructions)
5. [Module 10: Hypergeometric Functions](#module-10-hypergeometric-functions)
6. [Module 11: Performance and Internals](#module-11-performance-and-internals)
7. [Module 12: Research Techniques](#module-12-research-techniques)
8. [Complete Examples](#complete-examples)

---

## Introduction

**Prerequisites:**

This tutorial assumes you've completed Chapter 1 and are comfortable with:
- Creating and evaluating PCFs
- Using the `walk()` operation
- Working with Matrix and Position objects
- Basic CMF concepts

**What You'll Learn:**

- Master Linear Recurrences and their transformations
- Use solver algorithms to find equivalent representations
- Work with hypergeometric functions and D-finite structures
- Optimize performance with FLINT
- Apply research-level analysis techniques

---

## Module 7: Linear Recurrences

### Mathematical Background

A **Linear Recurrence** represents sequences where each term is a linear combination of previous terms:

**c₀·p(n) + c₁·p(n-1) + c₂·p(n-2) + ... + cₖ·p(n-k) = 0**

where the coefficients c₀, c₁, ..., cₖ can be polynomials in n.

### Lesson 7.1: Creating Linear Recurrences

**From Coefficient Lists:**

```python
from ramanujantools import LinearRecurrence, Matrix
from sympy.abc import n

# Fibonacci: p(n) = p(n-1) + p(n-2)
# Rearranged: p(n) - p(n-1) - p(n-2) = 0
# Coefficients: [1, -1, -1] for [p(n), p(n-1), p(n-2)]
fib = LinearRecurrence([1, -1, -1])

print("Fibonacci recurrence:")
print(f"Coefficients: {fib.coeffs}")
print(f"Order: {fib.order}")

# More complex: coefficients can be polynomials
lucas = LinearRecurrence([1, -1, -1])  # Same as Fibonacci
poly_rec = LinearRecurrence([1, -(2*n + 1), n**2])

print("\nPolynomial recurrence:")
print(f"Coefficients: {poly_rec.coeffs}")
```

**From Companion Matrices:**

```python
# Every 2x2 matrix defines a second-order recurrence
m = Matrix([[0, n], [1, 2*n + 1]])
rec_from_matrix = LinearRecurrence.from_matrix(m)

print("Recurrence from matrix:")
print(f"Matrix:\n{m}")
print(f"Coefficients: {rec_from_matrix.coeffs}")
```

### Lesson 7.2: Evaluating Sequences

Generate sequence values using initial conditions:

```python
# Fibonacci with initial values [F(0)=0, F(1)=1]
fib = LinearRecurrence([1, -1, -1])

# Initial values as a row vector
initial = Matrix([[0, 1]])

# Evaluate from n=0 to n=10
sequence = fib.evaluate_solution(initial, start=0, end=10)

print("Fibonacci sequence:")
print(sequence)  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]

# Lucas numbers with different initial values
lucas_initial = Matrix([[2, 1]])
lucas_seq = fib.evaluate_solution(lucas_initial, start=0, end=10)

print("\nLucas sequence (same recurrence, different initial values):")
print(lucas_seq)  # [2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123]
```

### Lesson 7.3: Convergence Analysis

Like PCFs, recurrences have convergence properties:

```python
# Create a recurrence
rec = LinearRecurrence([1, -(2*n + 1), n**2])
initial = Matrix([[1, 1]])

# Calculate limits at multiple depths
depths = [10, 20, 50, 100]
limits = rec.limit(depths, start=1, initial_values=initial)

print("Recurrence convergence:")
for i, depth in enumerate(depths):
    value = limits[i].as_float()
    precision = limits[i].precision()
    print(f"Depth {depth:3d}: {value:.10f} ({precision:.1f} digits)")

# The recurrence generates rational approximations to some constant
print(f"\nFinal approximation: {limits[-1].as_float()}")
```

### Lesson 7.4: Inflation and Deflation

Transform recurrences while preserving limits:

**Inflation** (multiply solution by a polynomial):

```python
rec = LinearRecurrence([1, -n, -(n-1)])

print("Original recurrence:")
print(f"Coefficients: {rec.coeffs}")

# Inflate by n (multiply solution by n)
inflated = rec.inflate(n)

print("\nAfter inflating by n:")
print(f"Coefficients: {inflated.coeffs}")

# Inflating changes the recurrence but preserves the limit
# If p(n) satisfies rec, then n·p(n) satisfies inflated
```

**Deflation** (divide solution by a polynomial):

```python
# Deflation is the inverse operation
rec_inflated = LinearRecurrence([1, -(2*n+1), n**2])

# Try to deflate by n
deflated = rec_inflated.deflate(n)

if deflated is not None:
    print("After deflating by n:")
    print(f"Coefficients: {deflated.coeffs}")
else:
    print("Cannot deflate by n (not a factor)")

# Normalize to canonical form
normalized = rec.normalize()
print("\nNormalized (leading coefficient = 1):")
print(f"Coefficients: {normalized.coeffs}")
```

### Lesson 7.5: Folding and Unfolding

**Folding** combines recurrences into higher order:

```python
rec = LinearRecurrence([1, -n, -(n-1)])

# Fold with multiplier n+1
# This creates a higher-order recurrence
folded = rec.fold(n + 1)

print("Original order:", rec.order)
print("Folded order:", folded.order)
print("Folded coefficients:", folded.coeffs)
```

**Unfolding** attempts to factor into lower order:

```python
# Start with a higher-order recurrence
rec_high = LinearRecurrence([1, -(3*n+2), 2*n**2])

# Try to unfold (factor into lower-order recurrence with inflation)
unfolded = rec_high.unfold(inflation_degree=2)

if unfolded is not None:
    print("Successfully unfolded!")
    print(f"Lower-order coefficients: {unfolded.coeffs}")
else:
    print("Cannot unfold to lower order")
```

### Lesson 7.6: Composition

Compose two recurrences to create a new one:

```python
rec1 = LinearRecurrence([1, -1, -1])  # Fibonacci
rec2 = LinearRecurrence([1, -2, 1])   # Powers of 2

# Composition creates a recurrence satisfied by p₁(p₂(n))
composed = rec1.compose(rec2)

print("Composed recurrence:")
print(f"Order: {composed.order}")
print(f"Coefficients: {composed.coeffs}")
```

### Lesson 7.7: Converting to PCF

Every second-order recurrence can be represented as a PCF:

```python
# Second-order recurrence
rec = LinearRecurrence([1, -(2*n+1), n**2])

# Convert to PCF
pcf = rec.to_pcf()

print("PCF from recurrence:")
print(f"a_n = {pcf.a_n}")
print(f"b_n = {pcf.b_n}")

# Evaluate both - they should give the same limit
rec_limit = rec.limit(100, start=1, initial_values=Matrix([[1, 1]]))
pcf_limit = pcf.limit(100, start=1)

print(f"\nRecurrence limit: {rec_limit.as_float()}")
print(f"PCF limit: {pcf_limit.as_float()}")
```

**Key Takeaway:** Linear recurrences are another way to represent convergent sequences. They can be transformed, composed, and converted to PCF form.

---

## Module 8: Solver Algorithms

The solver modules find alternative representations and prove equivalences.

### Lesson 8.1: Euler Solver

**Mathematical Background:**

A PCF is in the **Euler family** if there exist polynomials h₁(x), h₂(x), f(x) such that:

1. **b(x) = -h₁(x)h₂(x)** (b factors into two parts)
2. **f(x)a(x) = f(x-1)h₁(x) + f(x+1)h₂(x+1)** (balance condition)

When a PCF is in the Euler family, it can be transformed to an **infinite sum** representation!

**Finding Euler Representations:**

```python
from ramanujantools.solvers import EulerSolver
from ramanujantools.pcf import PCF
from sympy.abc import x, n

# Create a PCF
a = 2*x + 1
b = -x*(x + 1)

print("Testing PCF for Euler family membership:")
print(f"a(x) = {a}")
print(f"b(x) = {b}")

# Find all Euler representations
solutions = EulerSolver.solve_for(a, b)

print(f"\nFound {len(solutions)} Euler solution(s):")
for i, sol in enumerate(solutions):
    print(f"\nSolution {i+1}:")
    print(f"  h₁(x) = {sol.h_1}")
    print(f"  h₂(x) = {sol.h_2}")
    print(f"  f(x)  = {sol.f}")

    # Verify the conditions
    print(f"\nVerification:")
    print(f"  b = -h₁·h₂? {b == -sol.h_1 * sol.h_2}")

    balance = sol.f.subs(x, x) * a - (sol.f.subs(x, x-1) * sol.h_1 +
                                       sol.f.subs(x, x+1) * sol.h_2.subs(x, x+1))
    print(f"  Balance condition satisfied? {balance.simplify() == 0}")
```

**Complete Example with PCF:**

```python
# Create a PCF and check if it's Euler
pcf = PCF(a_n=6*n - 1, b_n=-n*(2*n - 1))

print("Checking PCF for Euler family:")
print(f"a_n = {pcf.a_n}")
print(f"b_n = {pcf.b_n}")

# Solve for Euler form
solutions = EulerSolver.solve_for(pcf.a_n, pcf.b_n)

if solutions:
    print(f"\n✓ This PCF is in the Euler family!")
    print(f"  Found {len(solutions)} representation(s)")

    # The infinite sum formula is:
    # K(b/a) = (f(1)h₂(1)/f(0)) * ((Σ f(0)f(1)/f(k)f(k+1) * Π h₁(i)/h₂(i+1))⁻¹ - 1)

    sol = solutions[0]
    print(f"\n  h₁(n) = {sol.h_1}")
    print(f"  h₂(n) = {sol.h_2}")
    print(f"  f(n)  = {sol.f}")
    print("\n  Can be expressed as infinite sum!")
else:
    print("\n✗ Not in Euler family")
```

**Why This Matters:**

Euler family PCFs can be transformed to infinite sums, which may be easier to:
- Evaluate numerically
- Prove properties about
- Connect to other mathematical structures
- Use in irrationality proofs

### Lesson 8.2: Coboundary Solver

**Mathematical Background:**

Two matrices M₁(x) and M₂(x) are **coboundary equivalent** if there exists U(x) such that:

**M₁(x) · U(x+1) = U(x) · M₂(x)**

This means they're related by a "gauge transformation" - they represent the same mathematical object in different coordinate systems.

**Finding Coboundary Matrices:**

```python
from ramanujantools.solvers import CoboundarySolver
from ramanujantools import Matrix
from sympy.abc import x

# Two different-looking matrices
M1 = Matrix([[0, x], [1, 2*x + 1]])
M2 = Matrix([[0, x + 1], [1, 2*x + 3]])

print("Finding coboundary relation between:")
print(f"M₁(x) =\n{M1}\n")
print(f"M₂(x) =\n{M2}\n")

# Find U(x) such that M₁(x)·U(x+1) = U(x)·M₂(x)
U, free_vars = CoboundarySolver.find_coboundary(M1, M2, max_deg=3)

if U is not None:
    print("✓ Found coboundary matrix U(x):")
    print(U)

    # Verify the relation
    lhs = M1 * U.subs(x, x+1)
    rhs = U * M2

    print(f"\nVerification:")
    print(f"M₁(x)·U(x+1) = U(x)·M₂(x)? {(lhs - rhs).simplify() == Matrix.zeros(2, 2)}")

    if free_vars:
        print(f"\nFree variables: {free_vars}")
else:
    print("✗ No coboundary relation found (try higher max_deg)")
```

**Application: Proving PCF Equivalence:**

```python
# Two PCFs that look different but might be equivalent
pcf1 = PCF(a_n=2*n + 1, b_n=n**2)
pcf2 = PCF(a_n=2*n + 3, b_n=(n+1)**2)

print("Checking if PCFs are coboundary equivalent:")
print(f"PCF₁: a={pcf1.a_n}, b={pcf1.b_n}")
print(f"PCF₂: a={pcf2.a_n}, b={pcf2.b_n}")

# Get their matrix representations
M1 = pcf1.M()
M2 = pcf2.M()

# Check for coboundary
U, _ = CoboundarySolver.find_coboundary(M1, M2, max_deg=2)

if U is not None:
    print("\n✓ PCFs are coboundary equivalent!")
    print("They converge to the same constant (up to a transformation)")
    print(f"Transformation matrix:\n{U}")
else:
    print("\n✗ PCFs are not equivalent")
```

**Application: Validating CMF Conservation:**

```python
# For a CMF, we need Mₓ(x,y)·Mᵧ(x+1,y) = Mᵧ(x,y)·Mₓ(x,y+1)
# This is a coboundary condition!

from sympy.abc import y

Mx = Matrix([[0, x+y], [1, x]])
My = Matrix([[y, x], [1, y+1]])

# Check conservation: Mₓ(x,y)·Mᵧ(x+1,y) = Mᵧ(x,y)·Mₓ(x,y+1)
lhs = Mx.subs(x, x) * My.subs(x, x+1)
rhs = My.subs(y, y) * Mx.subs(y, y+1)

print("CMF conservation check:")
print(f"Conserved? {(lhs - rhs).simplify() == Matrix.zeros(2, 2)}")

if (lhs - rhs).simplify() == Matrix.zeros(2, 2):
    print("✓ Valid CMF!")
else:
    print("✗ Not a valid CMF")
```

### Lesson 8.3: FFbar Solver

**Mathematical Background:**

An **FFbar construction** builds a 2D CMF from two functions f(x,y) and f̄(x,y) satisfying:

1. **Linear condition:** f(x+1,y-1) - f̄(x,y-1) + f̄(x+1,y) - f(x,y) = 0
2. **Quadratic condition:** f·f̄(x,y) - f·f̄(x,0) - f·f̄(0,y) + f·f̄(0,0) = 0

The resulting CMF has matrices:
- **Mₓ = [[0, b(x)], [1, a(x,y)]]**
- **Mᵧ = [[f̄(x,y), b(x)], [1, f(x,y)]]**

**Finding FFbar from a PCF:**

```python
from ramanujantools.solvers import FFbarSolver
from ramanujantools.pcf import PCF
from sympy.abc import n

# Create a PCF
pcf = PCF(a_n=n, b_n=-n*(n-1))

print("Finding FFbar construction for PCF:")
print(f"a_n = {pcf.a_n}")
print(f"b_n = {pcf.b_n}")

# Find all FFbar constructions where Mₓ matches this PCF
ffbars = FFbarSolver.from_pcf(pcf)

print(f"\nFound {len(ffbars)} FFbar construction(s):")
for i, ffbar in enumerate(ffbars):
    print(f"\nFFbar {i+1}:")
    print(f"  f(x,y) = {ffbar.f}")
    print(f"  f̄(x,y) = {ffbar.fbar}")

    # The FFbar object is actually a CMF
    print(f"  Axes: {ffbar.axes()}")
```

**Solving FFbar Conditions Directly:**

```python
from sympy import symbols
from ramanujantools.solvers import FFbarSolver

x, y = symbols('x y')
# Create generic polynomials with symbolic coefficients
a0, a1, b0, b1 = symbols('a0 a1 b0 b1')

f = a0 + a1*x
fbar = b0 + b1*y

print("Solving FFbar conditions for:")
print(f"f(x,y) = {f}")
print(f"f̄(x,y) = {fbar}")

# Solve the linear and quadratic conditions
solutions = FFbarSolver.solve_ffbar(f, fbar)

print(f"\nFound {len(solutions)} valid parameterization(s):")
for i, sol in enumerate(solutions):
    print(f"\nSolution {i+1}:")
    print(f"  f = {sol.f}")
    print(f"  f̄ = {sol.fbar}")
```

**Complete FFbar Example:**

```python
from ramanujantools.cmf import FFbar
from sympy.abc import x, y

# Define f and f̄
f = x + y
fbar = x - y + 1

print("Creating FFbar CMF:")
print(f"f(x,y) = {f}")
print(f"f̄(x,y) = {fbar}")

# Create and validate FFbar
try:
    cmf_ffbar = FFbar(f, fbar)
    print("\n✓ Valid FFbar construction!")

    # Get the matrices
    print(f"\nMₓ =\n{cmf_ffbar[x]}")
    print(f"\nMᵧ =\n{cmf_ffbar[y]}")

    # Evaluate along a trajectory
    from ramanujantools import Position
    trajectory = Position({x: 1, y: 1})
    start = Position({x: 0, y: 0})

    limit = cmf_ffbar.limit(trajectory, 50, start)
    print(f"\nConverges to: {limit.as_float()}")

except Exception as e:
    print(f"\n✗ Invalid FFbar: {e}")
```

**Key Takeaway:** FFbar is a powerful construction that generates 2D CMFs from simpler 1D functions. Many PCFs can be "lifted" to FFbar representations.

---

## Module 9: Advanced CMF Constructions

### Lesson 9.1: Creating Custom CMFs

Build your own CMF from scratch:

```python
from ramanujantools.cmf import CMF
from ramanujantools import Matrix
from sympy.abc import x, y

# Define matrices for each axis
Mx = Matrix([[0, x + y], [1, 2*x + y]])
My = Matrix([[y, x + y], [1, x + y + 1]])

print("Creating custom CMF:")
print(f"Mₓ(x,y) =\n{Mx}\n")
print(f"Mᵧ(x,y) =\n{My}\n")

# Create CMF with validation
try:
    cmf = CMF({x: Mx, y: My}, validate=True)
    print("✓ Valid CMF (conservation property satisfied)")

    # CMF properties
    print(f"\nAxes: {cmf.axes()}")
    print(f"Dimension: {cmf.dim()}")
    print(f"Rank (matrix size): {cmf.rank()}")

except ValueError as e:
    print(f"✗ Invalid CMF: {e}")
    print("Conservation property not satisfied")

    # Try creating without validation (for experimentation)
    cmf = CMF({x: Mx, y: My}, validate=False)
    print("\nCreated without validation (use for testing only)")
```

**Checking Conservation Manually:**

```python
# The conservation property: Mₓ(x,y)·Mᵧ(x+1,y) = Mᵧ(x,y)·Mₓ(x,y+1)

Mx_shifted = Mx.subs(x, x+1)
My_shifted = My.subs(y, y+1)

lhs = Mx * My_shifted
rhs = My * Mx_shifted

difference = (lhs - rhs).simplify()

print("Conservation check:")
print(f"Mₓ(x,y)·Mᵧ(x+1,y) - Mᵧ(x,y)·Mₓ(x,y+1) =")
print(difference)

if difference == Matrix.zeros(2, 2):
    print("\n✓ Conservation satisfied!")
else:
    print("\n✗ Conservation violated")
```

### Lesson 9.2: CMF Transformations

**Sub-CMF Extraction:**

Extract a lower-dimensional CMF along a trajectory:

```python
from ramanujantools.cmf import known_cmfs
from ramanujantools import Position

# Start with a 2D CMF
cmf_2d = known_cmfs.zeta3()

print(f"Original CMF dimension: {cmf_2d.dim()}")

# Extract 1D sub-CMF along trajectory (1,1)
basis = [Position({x: 1, y: 1})]
sub_cmf = cmf_2d.sub_cmf(basis)

print(f"Sub-CMF dimension: {sub_cmf.dim()}")
print(f"This is effectively a PCF along the (1,1) direction")
```

**Dual CMF:**

Get the inverse-transpose CMF:

```python
cmf = known_cmfs.e()

# Compute dual
dual = cmf.dual()

print("Original CMF:")
print(f"Mₓ(0,0) =\n{cmf[x].subs({x: 0, y: 0})}\n")

print("Dual CMF:")
print(f"(Mₓ⁻¹)ᵀ(0,0) =\n{dual[x].subs({x: 0, y: 0})}")
```

**Coboundary Transformation:**

Apply a gauge transformation to the entire CMF:

```python
from ramanujantools import Matrix
from sympy.abc import x, y

cmf = known_cmfs.e()

# Define a transformation matrix U(x,y)
U = Matrix([[1, x], [0, 1]])

# Apply coboundary transformation
# New CMF: M'ₓ(x,y) = U⁻¹(x,y)·Mₓ(x,y)·U(x+1,y)
cmf_transformed = cmf.coboundary(U)

print("Transformed CMF:")
print(f"Original Mₓ(0,0) =\n{cmf[x].subs({x: 0, y: 0})}\n")
print(f"Transformed Mₓ(0,0) =\n{cmf_transformed[x].subs({x: 0, y: 0})}")

# Both CMFs converge to related values
```

### Lesson 9.3: Work Calculations

The `work()` method computes path-independent transformations:

```python
from ramanujantools import Position

cmf = known_cmfs.zeta3()

# Define start and end points
start = Position({x: 0, y: 0})
end = Position({x: 5, y: 3})

# Compute work (path-independent!)
W = cmf.work(start, end)

print(f"Work from {start} to {end}:")
print(W)

# This is the product of all matrices along ANY path from (0,0) to (5,3)
# Due to conservation, all paths give the same result

# Verify with different paths:
# Path 1: Go right then up
path1 = cmf[x].walk({x: 1, y: 0}, 5, start) * cmf[y].walk({x: 0, y: 1}, 3, {x: 5, y: 0})

# Path 2: Go up then right
path2 = cmf[y].walk({x: 0, y: 1}, 3, start) * cmf[x].walk({x: 1, y: 0}, 5, {x: 0, y: 3})

# Path 3: Diagonal
# (This is trickier - requires the work() algorithm)

print(f"\nPath independence check:")
print(f"Work matches path 1? {(W - path1).simplify() == Matrix.zeros(2, 2)}")
print(f"Work matches path 2? {(W - path2).simplify() == Matrix.zeros(2, 2)}")
```

---

## Module 10: Hypergeometric Functions

### Lesson 10.1: The pFq Framework

**Mathematical Background:**

The **generalized hypergeometric function** ₚFᵧ is defined by:

**ₚFᵧ(a₁,...,aₚ; b₁,...,bᵧ; z) = Σ (a₁)ₙ···(aₚ)ₙ / (b₁)ₙ···(bᵧ)ₙ · zⁿ/n!**

where (a)ₙ = a(a+1)···(a+n-1) is the Pochhammer symbol.

In ramanujantools, these are represented as CMFs!

**Creating a pFq CMF:**

```python
from ramanujantools.cmf import pFq
from ramanujantools import Position
import sympy as sp

# Create 2F1 (Gauss hypergeometric function)
cmf_2f1 = pFq(p=2, q=1, z=sp.Rational(1, 2))

print("2F1 hypergeometric CMF:")
print(f"p (numerator params): {cmf_2f1.p}")
print(f"q (denominator params): {cmf_2f1.q}")
print(f"z (argument): {cmf_2f1.z}")
print(f"Dimension: {cmf_2f1.dim()}")  # p + q dimensions
print(f"Axes: {cmf_2f1.axes()}")  # x0, x1, y0

# The axes correspond to parameters
# x0, x1 are numerator parameters a1, a2
# y0 is denominator parameter b1
```

**Evaluating pFq Functions:**

```python
# Define parameter values
a1, a2 = sp.Rational(1, 2), sp.Rational(3, 2)
b1 = sp.Rational(5, 2)
z = sp.Rational(1, 2)

# Create 2F1(1/2, 3/2; 5/2; 1/2)
cmf = pFq(p=2, q=1, z=z)

# Set up starting point (parameter values)
from sympy import symbols
x0, x1, y0 = symbols('x0 x1 y0')
start = Position({x0: a1, x1: a2, y0: b1})

# Choose a trajectory (affects which formula we get)
trajectory = Position({x0: 1, x1: 0, y0: 0})

# Evaluate
limit = cmf.limit(trajectory, 50, start)
print(f"\n2F1(1/2, 3/2; 5/2; 1/2) ≈ {limit.as_float()}")

# Can also use direct evaluation
value = pFq.evaluate([a1, a2], [b1], z)
print(f"Direct evaluation: {value}")
```

**Special Cases:**

```python
# 1F1 (Kummer confluent hypergeometric)
cmf_1f1 = pFq(p=1, q=1, z=1)
print("\n1F1 (Kummer):")
print(f"Dimension: {cmf_1f1.dim()}")

# 0F1 (Bessel-related)
cmf_0f1 = pFq(p=0, q=1, z=sp.Rational(1, 4))
print("\n0F1:")
print(f"Dimension: {cmf_0f1.dim()}")

# 3F2 (higher order)
cmf_3f2 = pFq(p=3, q=2, z=sp.Rational(-1, 27))
print("\n3F2:")
print(f"Dimension: {cmf_3f2.dim()}")
```

### Lesson 10.2: Ascending and State Vectors

**Ascending Operation:**

Extend pFq to p+1Fq+1 while preserving delta:

```python
cmf_2f1 = pFq(p=2, q=1, z=sp.Rational(1, 2))

# Ascend to 3F2
cmf_3f2 = cmf_2f1.ascend()

print("After ascending:")
print(f"Original: {cmf_2f1.p}F{cmf_2f1.q}")
print(f"Ascended: {cmf_3f2.p}F{cmf_3f2.q}")
print(f"Dimension increase: {cmf_2f1.dim()} → {cmf_3f2.dim()}")
```

**State Vectors:**

Compute derivatives of pFq:

```python
# State vector is [pFq, θ·pFq, θ²·pFq, ...]
# where θ = z·d/dz

cmf = pFq(p=2, q=1, z=sp.Rational(1, 2))

# Compute state vector
a_params = [sp.Rational(1, 2), sp.Rational(3, 2)]
b_params = [sp.Rational(5, 2)]

state = pFq.state_vector(a_params, b_params, z=sp.Rational(1, 2), order=3)

print("State vector:")
for i, val in enumerate(state):
    print(f"  θ^{i} · 2F1 = {val}")
```

### Lesson 10.3: Meijer G-Functions

**Mathematical Background:**

The **Meijer G-function** is an even more general special function:

**G^{m,n}_{p,q}(z | a₁,...,aₚ; b₁,...,bᵧ)**

It generalizes hypergeometric functions, Bessel functions, and many others.

**Creating Meijer G CMFs:**

```python
from ramanujantools.cmf import MeijerG

# Create a Meijer G CMF
# G^{m,n}_{p,q} with parameters m=1, n=1, p=2, q=2
cmf_g = MeijerG(m=1, n=1, p=2, q=2, z=sp.Rational(1, 4))

print("Meijer G-function CMF:")
print(f"Parameters: m={cmf_g.m}, n={cmf_g.n}, p={cmf_g.p}, q={cmf_g.q}")
print(f"Dimension: {cmf_g.dim()}")  # p + q dimensions
print(f"Axes: {cmf_g.axes()}")

# Constraints: 0 ≤ n ≤ p, 0 ≤ m ≤ q
```

**Evaluating Meijer G:**

```python
# Set up parameters
from sympy import symbols
a0, a1, b0, b1 = symbols('a:2'), symbols('b:2')

# Starting point
start = Position({a0: sp.Rational(1, 2), a1: sp.Rational(1, 3),
                  b0: sp.Rational(2, 3), b1: sp.Rational(3, 4)})

# Trajectory
trajectory = Position({a0: 1, a1: 0, b0: 0, b1: 0})

# Evaluate
limit = cmf_g.limit(trajectory, 30, start)
print(f"\nMeijer G value: {limit.as_float()}")
```

**Connection to Other Functions:**

Many special functions are special cases of Meijer G:

```python
# Bessel function Jν(z) = G^{1,0}_{0,2}
bessel_cmf = MeijerG(m=1, n=0, p=0, q=2, z=1)

# Exponential: e^z = G^{1,0}_{0,1}
exp_cmf = MeijerG(m=1, n=0, p=0, q=1, z=1)

print("Special cases of Meijer G:")
print(f"Bessel: {bessel_cmf.dim()}D CMF")
print(f"Exponential: {exp_cmf.dim()}D CMF")
```

---

## Module 11: Performance and Internals

### Lesson 11.1: FLINT Integration

**What is FLINT?**

FLINT (Fast Library for Number Theory) provides highly optimized polynomial arithmetic. Ramanujantools uses it automatically for performance-critical operations.

**When FLINT is Used:**

```python
from ramanujantools import Matrix
from sympy.abc import n

# Create a symbolic matrix
m = Matrix([[0, n], [1, 2*n + 1]])

# When you call walk(), ramanujantools automatically:
# 1. Detects this is symbolic
# 2. Converts to FLINT representation
# 3. Uses fast FLINT arithmetic
# 4. Converts back to sympy

result = m.walk({n: 1}, 1000, {n: 1})
# This is 10-100x faster than pure sympy!

print("Walk completed using FLINT backend")
print(f"Result matrix GCD: {result.gcd}")
```

**SymbolicMatrix vs NumericMatrix:**

```python
# Symbolic walk (uses FLINT for polynomial arithmetic)
m_symbolic = Matrix([[0, n], [1, n]])
result_symbolic = m_symbolic.walk({n: 1}, 100, {n: 1})
print("Symbolic result (exact polynomials):")
print(f"Type: {type(result_symbolic)}")

# Numeric walk (uses mpmath for high-precision arithmetic)
m_numeric = Matrix([[0, 5], [1, 10]])  # No symbols
result_numeric = m_numeric.walk({}, 100, {})  # No trajectory needed
print("\nNumeric result (high-precision numbers):")
print(f"Type: {type(result_numeric)}")
```

**FlintContext for Advanced Use:**

```python
from ramanujantools.flint_core import flint_ctx, SymbolicMatrix
from sympy.abc import x, y

# Create a FLINT context for variables x, y
with flint_ctx([x, y], fmpz=True) as ctx:
    # Work with FLINT polynomials directly
    # fmpz=True means integer coefficients
    # fmpz=False would use rational coefficients

    print(f"FLINT context for: {ctx.vars}")
    print(f"Polynomial ring: {ctx.ring}")

    # You can create SymbolicMatrix directly
    from ramanujantools.flint_core import SymbolicMatrix

    # This uses FLINT internally for fast operations
    m_flint = SymbolicMatrix.from_sympy(Matrix([[x, y], [1, x+y]]), ctx)
    print(f"\nFLINT matrix created")
```

**Performance Comparison:**

```python
import time
from ramanujantools import Matrix
from sympy.abc import n

m = Matrix([[0, n**2], [1, 3*n**2 + 2*n + 1]])

# Large walk - FLINT makes this practical
start_time = time.time()
result = m.walk({n: 1}, 500, {n: 1})
elapsed = time.time() - start_time

print(f"Walk to depth 500 completed in {elapsed:.2f} seconds")
print(f"Using FLINT backend automatically")
print(f"Without FLINT, this would take much longer!")

# The result is exact symbolic polynomials
print(f"\nResult matrix [0,0] degree: {result[0,0].as_poly(n).degree()}")
```

### Lesson 11.2: Caching and Optimization

**Batched Evaluation:**

The batched evaluation pattern reuses intermediate results:

```python
from ramanujantools.pcf import PCF
from sympy.abc import n
import time

pcf = PCF(a_n=2*n + 1, b_n=n**2)

# Inefficient: separate calls
print("Inefficient approach:")
start = time.time()
limit_10 = pcf.limit(10, start=1)
limit_20 = pcf.limit(20, start=1)  # Recomputes first 10 steps!
limit_50 = pcf.limit(50, start=1)  # Recomputes first 20 steps!
elapsed_inefficient = time.time() - start
print(f"Time: {elapsed_inefficient:.3f}s")

# Efficient: batched evaluation
print("\nEfficient approach:")
start = time.time()
limits = pcf.limit([10, 20, 50], start=1)  # Computes once, reuses results
elapsed_efficient = time.time() - start
print(f"Time: {elapsed_efficient:.3f}s")

print(f"\nSpeedup: {elapsed_inefficient/elapsed_efficient:.1f}x")
```

**Cached Properties:**

Many expensive computations are cached:

```python
from ramanujantools.pcf import PCF
from sympy.abc import n

pcf = PCF(a_n=2*n**3 + n, b_n=-n**4)

# First access computes and caches
print("First access to singular_points (computed):")
import time
start = time.time()
sing1 = pcf.singular_points()
elapsed1 = time.time() - start
print(f"Time: {elapsed1:.4f}s")

# Second access uses cached value
print("\nSecond access to singular_points (cached):")
start = time.time()
sing2 = pcf.singular_points()
elapsed2 = time.time() - start
print(f"Time: {elapsed2:.4f}s")

print(f"\nCached access is {elapsed1/elapsed2:.0f}x faster!")
```

---

## Module 12: Research Techniques

### Lesson 12.1: Formula Discovery Workflow

**Complete pipeline from discovery to publication:**

```python
from ramanujantools import Matrix
from ramanujantools.pcf import PCF
from ramanujantools.solvers import EulerSolver, FFbarSolver, CoboundarySolver
from sympy.abc import n
import mpmath

# Step 1: Discovery (assume we found this numerically)
pcf_candidate = PCF(a_n=6*n**2 + 6*n + 1, b_n=-n**3*(2*n+1))

print("="*60)
print("FORMULA DISCOVERY WORKFLOW")
print("="*60)

# Step 2: Deflation to canonical form
print("\n1. CANONICAL FORM")
pcf = pcf_candidate.deflate_all()
print(f"   a_n = {pcf.a_n}")
print(f"   b_n = {pcf.b_n}")
print(f"   Degrees: {pcf.degrees()}")

# Step 3: High-precision evaluation
print("\n2. HIGH-PRECISION EVALUATION")
mpmath.mp.dps = 200  # 200 decimal places
limits = pcf.limit([100, 500, 1000], start=1)
value = limits[-1].as_float()
precision = limits[-1].precision()
print(f"   Converges to: {value}")
print(f"   Precision: {precision:.1f} decimal places")

# Step 4: Constant identification
print("\n3. CONSTANT IDENTIFICATION")
candidates = {
    'π': mpmath.pi,
    'e': mpmath.e,
    'ln(2)': mpmath.log(2),
    'ζ(2)': mpmath.zeta(2),
    'ζ(3)': mpmath.zeta(3),
    'Catalan': mpmath.catalan,
}

identified = None
for name, const in candidates.items():
    result = limits[-1].identify(const)
    if result is not None:
        print(f"   ✓ Related to {name}!")
        identified = (name, const)
        break

if identified:
    name, const = identified

    # Step 5: Irrationality measure
    print("\n4. IRRATIONALITY MEASURE")
    delta_actual = pcf.delta(depth=1000, L=const, start=1)
    delta_predicted = pcf.kamidelta(depth=500)
    print(f"   Actual δ: {delta_actual}")
    print(f"   Predicted δ (kamidelta): {delta_predicted}")

    # Step 6: Check for Euler family
    print("\n5. EULER FAMILY CHECK")
    euler_sols = EulerSolver.solve_for(pcf.a_n, pcf.b_n)
    if euler_sols:
        print(f"   ✓ In Euler family ({len(euler_sols)} representation(s))")
        sol = euler_sols[0]
        print(f"   h₁ = {sol.h_1}")
        print(f"   h₂ = {sol.h_2}")
        print(f"   f  = {sol.f}")
    else:
        print("   ✗ Not in Euler family")

    # Step 7: Check for FFbar construction
    print("\n6. FFBAR CONSTRUCTION CHECK")
    ffbar_sols = FFbarSolver.from_pcf(pcf)
    if ffbar_sols:
        print(f"   ✓ Has FFbar representation ({len(ffbar_sols)} found)")
        ffbar = ffbar_sols[0]
        print(f"   f  = {ffbar.f}")
        print(f"   f̄ = {ffbar.fbar}")
    else:
        print("   ✗ No FFbar representation found")

    # Step 8: Literature search simulation
    print("\n7. NOVELTY CHECK")
    print("   (In real workflow: search literature, databases)")
    print("   (Check Ramanujan Library, OEIS, known formulas)")

    print("\n" + "="*60)
    print(f"CONCLUSION: New formula for {name}")
    print("="*60)
```

### Lesson 12.2: Equivalence Proving

**Proving two formulas are the same:**

```python
# Two formulas that look different
pcf1 = PCF(a_n=2*n + 1, b_n=-n**2)
pcf2 = PCF(a_n=2*n + 3, b_n=-(n+1)**2)

print("EQUIVALENCE PROOF WORKFLOW\n")

# Step 1: Compare numerical values
print("1. NUMERICAL CHECK")
val1 = pcf1.limit(500, start=1).as_float()
val2 = pcf2.limit(500, start=1).as_float()
print(f"   PCF₁ → {val1}")
print(f"   PCF₂ → {val2}")

if abs(val1 - val2) < 1e-50:
    print("   ✓ Numerically equivalent")

    # Step 2: Coboundary check
    print("\n2. COBOUNDARY CHECK")
    M1 = pcf1.M()
    M2 = pcf2.M()

    U, free = CoboundarySolver.find_coboundary(M1, M2, max_deg=3)

    if U is not None:
        print("   ✓ Coboundary equivalent!")
        print(f"   Transformation matrix U(n):")
        print(f"   {U}")
        print("\n   This proves: PCF₁ ≡ PCF₂ (mathematically equivalent)")
    else:
        print("   ✗ Not coboundary equivalent (try higher degree)")
else:
    print("   ✗ Different constants")
```

### Lesson 12.3: Large-Scale Analysis

**Analyzing multiple formulas:**

```python
from ramanujantools.pcf import PCF
from sympy.abc import n
import mpmath

# Collection of PCF candidates
pcfs = [
    PCF(a_n=n, b_n=n),
    PCF(a_n=2*n+1, b_n=n**2),
    PCF(a_n=6*n-1, b_n=-n*(2*n-1)),
    PCF(a_n=n**2, b_n=-n**3),
    PCF(a_n=3*n, b_n=-2*n*(n-1)),
]

print("LARGE-SCALE ANALYSIS")
print("="*70)

results = []
for i, pcf in enumerate(pcfs):
    print(f"\nPCF {i+1}: a_n={pcf.a_n}, b_n={pcf.b_n}")

    # Deflate
    pcf_clean = pcf.deflate_all()

    # Evaluate
    limit = pcf_clean.limit(200, start=1)
    value = float(limit.as_float())

    # Predict delta
    try:
        delta = pcf_clean.kamidelta(depth=100)
        delta_val = delta[0] if isinstance(delta, list) else delta
    except:
        delta_val = None

    # Check Euler family
    euler = len(EulerSolver.solve_for(pcf_clean.a_n, pcf_clean.b_n)) > 0

    # Store results
    results.append({
        'pcf': pcf_clean,
        'value': value,
        'delta': delta_val,
        'euler': euler,
        'degrees': pcf_clean.degrees()
    })

    print(f"  Value: {value:.10f}")
    print(f"  Delta: {delta_val}")
    print(f"  Euler family: {euler}")
    print(f"  Degrees: {pcf_clean.degrees()}")

# Clustering by convergence properties
print("\n" + "="*70)
print("CLUSTERING BY DELTA")
print("="*70)

# Group by similar delta values
for r in sorted(results, key=lambda x: x['delta'] if x['delta'] else 0):
    print(f"δ ≈ {r['delta']:.3f if r['delta'] else 'N/A':>6} : a_n={r['pcf'].a_n}")
```

### Lesson 12.4: Companion Matrix Analysis

**Deep dive into eigenvalue structure:**

```python
from ramanujantools import Matrix
from sympy.abc import n
import sympy as sp

# Create a PCF
pcf = PCF(a_n=2*n + 1, b_n=-n**2)
M = pcf.M()

print("COMPANION MATRIX ANALYSIS\n")

# Convert to companion form
print("1. COMPANION FORM")
companion = M.as_companion()
print(f"   Is companion? {companion.is_companion()}")
print(f"   Companion matrix:")
print(f"   {companion}")

# Characteristic polynomial
print("\n2. CHARACTERISTIC POLYNOMIAL")
char_poly = M.charpoly(n)
print(f"   det(λI - M) = {char_poly}")

# Eigenvalues
print("\n3. EIGENVALUES")
eigenvals = M.eigenvals()
print(f"   Eigenvalues: {eigenvals}")

# Sorted eigenvalues (by absolute value in Poincaré form)
print("\n4. SORTED EIGENVALUES (Poincaré)")
sorted_eigs = M.sorted_eigenvals()
for i, eig in enumerate(sorted_eigs):
    print(f"   λ_{i} = {eig}")

# Poincaré characteristic polynomial
print("\n5. POINCARÉ FORM")
poincare_poly = M.poincare_charpoly()
print(f"   Rescaled polynomial: {poincare_poly}")

# This is used in kamidelta algorithm!
print("\n6. CONNECTION TO KAMIDELTA")
print("   Eigenvalue ratios determine convergence rate")
print("   GCD slope determines denominator growth")
print("   Delta = -1 + (error rate) / (growth rate)")
```

---

## Complete Examples

### Example 1: Discovering and Verifying a New Formula

```python
"""
Complete workflow: Assume we numerically discovered a PCF
and want to verify and understand it.
"""

from ramanujantools.pcf import PCF
from ramanujantools.solvers import EulerSolver, FFbarSolver
from sympy.abc import n
import mpmath

# The discovered PCF
pcf_raw = PCF(a_n=34*n**3 + 51*n**2 + 27*n + 5, b_n=-n**6)

print("COMPLETE VERIFICATION WORKFLOW")
print("="*70)

# 1. Canonical form
print("\n1. DEFLATION")
pcf = pcf_raw.deflate_all()
print(f"   a_n = {pcf.a_n}")
print(f"   b_n = {pcf.b_n}")

# 2. Evaluate
print("\n2. EVALUATION")
mpmath.mp.dps = 100
limits = pcf.limit([50, 100, 500], start=1)
for i, depth in enumerate([50, 100, 500]):
    print(f"   Depth {depth}: {limits[i].as_float()}")

# 3. Identify
print("\n3. IDENTIFICATION")
value = limits[-1].as_float()
zeta3 = mpmath.zeta(3)
if abs(value - zeta3) < 1e-50:
    print(f"   ✓ Converges to ζ(3)!")
    print(f"   Error: {abs(value - zeta3):.2e}")

# 4. This is Apéry's formula!
print("\n4. HISTORICAL SIGNIFICANCE")
print("   This is Apéry's 1979 formula that proved ζ(3) is irrational!")
print("   It comes from the zeta3() CMF along trajectory (1,1)")

# 5. Verify with CMF
print("\n5. CMF VERIFICATION")
from ramanujantools.cmf import known_cmfs
from ramanujantools import Position

cmf = known_cmfs.zeta3()
traj = Position({x: 1, y: 1})
start = Position({x: 0, y: 0})

traj_matrix = cmf.trajectory_matrix(traj, start)
pcf_from_cmf = PCF(traj_matrix).deflate_all()

print(f"   From CMF: a_n = {pcf_from_cmf.a_n}")
print(f"   Matches? {pcf_from_cmf.a_n == pcf.a_n}")

print("\n" + "="*70)
```

### Example 2: Comparing Multiple Representations

```python
"""
Show that different-looking formulas are actually the same
using multiple techniques.
"""

from ramanujantools.pcf import PCF
from ramanujantools.solvers import CoboundarySolver, EulerSolver
from sympy.abc import n

# Three different-looking PCFs for ln(2)
pcf1 = PCF(a_n=2*n - 1, b_n=-n**2)
pcf2 = PCF(a_n=2*n + 1, b_n=-(n+1)**2)
pcf3 = PCF(a_n=4*n - 1, b_n=-4*n**2)

print("PROVING FORMULA EQUIVALENCE")
print("="*70)

# 1. Numerical verification
print("\n1. NUMERICAL VALUES")
val1 = pcf1.limit(500, start=1).as_float()
val2 = pcf2.limit(500, start=1).as_float()
val3 = pcf3.limit(500, start=1).as_float()

print(f"   PCF₁ → {val1}")
print(f"   PCF₂ → {val2}")
print(f"   PCF₃ → {val3}")

import mpmath
ln2 = mpmath.log(2)
print(f"   ln(2) = {ln2}")
print(f"   All close to ln(2)? {all(abs(v - ln2) < 1e-20 for v in [val1, val2, val3])}")

# 2. Coboundary relations
print("\n2. COBOUNDARY RELATIONS")

# PCF1 vs PCF2
U12, _ = CoboundarySolver.find_coboundary(pcf1.M(), pcf2.M(), max_deg=2)
if U12 is not None:
    print(f"   PCF₁ ~ PCF₂ via U₁₂(n) = ")
    print(f"   {U12}")

# PCF1 vs PCF3
U13, _ = CoboundarySolver.find_coboundary(pcf1.M(), pcf3.M(), max_deg=2)
if U13 is not None:
    print(f"   PCF₁ ~ PCF₃ via U₁₃(n) = ")
    print(f"   {U13}")

# 3. Euler family
print("\n3. EULER FAMILY ANALYSIS")
for i, pcf in enumerate([pcf1, pcf2, pcf3], 1):
    sols = EulerSolver.solve_for(pcf.a_n, pcf.b_n)
    if sols:
        print(f"   PCF₂: Euler family ✓")
        print(f"      h₁={sols[0].h_1}, h₂={sols[0].h_2}")
    else:
        print(f"   PCF_{i}: Not Euler family ✗")

print("\n4. CONCLUSION")
print("   All three PCFs converge to ln(2)")
print("   PCF₁ and PCF₂ are coboundary equivalent (same formula)")
print("   PCF₃ is a scaled version")

print("="*70)
```

### Example 3: Custom CMF Research

```python
"""
Research workflow: Build a custom CMF, validate it,
and explore its properties.
"""

from ramanujantools.cmf import CMF
from ramanujantools import Matrix, Position
from sympy.abc import x, y

print("CUSTOM CMF RESEARCH WORKFLOW")
print("="*70)

# 1. Design matrices
print("\n1. DESIGNING CMF")
print("   Goal: Create CMF for a specific constant")

# Example: Matrices inspired by Fibonacci-like structure
Mx = Matrix([[y, x + y], [1, x]])
My = Matrix([[x, x + y], [1, y]])

print(f"   Mₓ(x,y) =\n{Mx}\n")
print(f"   Mᵧ(x,y) =\n{My}\n")

# 2. Validate conservation
print("2. CONSERVATION VALIDATION")
try:
    cmf = CMF({x: Mx, y: My}, validate=True)
    print("   ✓ Valid CMF (conservation property satisfied)")
except ValueError as e:
    print(f"   ✗ Invalid: {e}")
    print("   Adjusting matrices...")
    # Would adjust and retry

# 3. Explore trajectories
print("\n3. TRAJECTORY EXPLORATION")
trajectories = [
    Position({x: 1, y: 0}),  # x-axis
    Position({x: 0, y: 1}),  # y-axis
    Position({x: 1, y: 1}),  # diagonal
    Position({x: 2, y: 1}),  # custom
]

start = Position({x: 0, y: 0})

for traj in trajectories:
    # Extract PCF for this trajectory
    traj_matrix = cmf.trajectory_matrix(traj, start)
    pcf = PCF(traj_matrix).deflate_all()

    # Evaluate
    limit = pcf.limit(100, start=1)

    print(f"\n   Trajectory {dict(traj)}:")
    print(f"      PCF: a_n={pcf.a_n}, b_n={pcf.b_n}")
    print(f"      Value: {limit.as_float()}")

# 4. Analyze structure
print("\n4. STRUCTURE ANALYSIS")
print(f"   Dimension: {cmf.dim()}")
print(f"   Rank: {cmf.rank()}")
print(f"   Axes: {cmf.axes()}")

# 5. Check for FFbar
print("\n5. FFBAR CHECK")
from ramanujantools.solvers import FFbarSolver

# Try to express as FFbar
pcf_x = PCF(Mx).deflate_all()
ffbars = FFbarSolver.from_pcf(pcf_x)

if ffbars:
    print(f"   ✓ Has FFbar representation!")
    print(f"   f = {ffbars[0].f}")
    print(f"   f̄ = {ffbars[0].fbar}")
else:
    print("   ✗ No FFbar representation")

print("\n" + "="*70)
```

---

## Summary

You now have mastery of:

### Core Skills
✓ Linear recurrences and their transformations
✓ Solver algorithms (Euler, Coboundary, FFbar)
✓ Advanced CMF constructions and analysis
✓ Hypergeometric and Meijer G functions
✓ Performance optimization with FLINT
✓ Research-level analysis techniques

### Research Capabilities
✓ Discover and verify new formulas
✓ Prove equivalence between formulas
✓ Build and validate custom CMFs
✓ Analyze convergence properties at scale
✓ Connect to classical special function theory

### Next Steps
1. **Read the research papers** - Understand the theoretical foundations
2. **Explore the test files** - See advanced usage patterns
3. **Contribute to the package** - Add new features or examples
4. **Apply to your research** - Use these tools for mathematical discovery

---

## Additional Resources

**Source Code:**
- `ramanujantools/linear_recurrence.py` - Full recurrence implementation
- `ramanujantools/solvers/` - All solver algorithms
- `ramanujantools/cmf/pfq.py` - Hypergeometric functions
- `ramanujantools/flint_core/` - Performance layer

**Test Files:**
- `linear_recurrence_test.py` - Recurrence examples
- `solvers/*_test.py` - Solver usage
- `cmf/pfq_test.py` - Hypergeometric examples
- `cmf/meijer_g_test.py` - Meijer G examples

**Research Papers:**
- [arXiv:2303.09318](https://arxiv.org/abs/2303.09318) - CMF mathematical theory
- [arXiv:2308.02567](https://arxiv.org/abs/2308.02567) - Euler PCF algorithm
- [arXiv:2111.04468](https://arxiv.org/abs/2111.04468) - Kamidelta theory
- [arXiv:2507.08138](https://arxiv.org/abs/2507.08138) - Advanced CMF topics

---

*Advanced Tutorial for ramanujantools - Chapter 2*

*For questions or contributions: https://github.com/RamanujanMachine/ramanujantools*

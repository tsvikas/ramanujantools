# Theoretical Foundations of ramanujantools

This document explains the mathematical theory and research context behind the ramanujantools package, synthesized from the Ramanujan Machine publications.

## Table of Contents
1. [Research Context](#research-context)
2. [The Central Problem](#the-central-problem)
3. [Mathematical Frameworks](#mathematical-frameworks)
4. [Key Algorithms and Their Origins](#key-algorithms-and-their-origins)
5. [The Kamidelta Algorithm Explained](#the-kamidelta-algorithm-explained)
6. [From Discovery to Proof](#from-discovery-to-proof)
7. [Real-World Impact](#real-world-impact)

---

## Research Context

### The Ramanujan Machine Project

The ramanujantools package is the **computational engine** of the Ramanujan Machine, a research initiative at Technion – Israel Institute of Technology that aims to automate mathematical discovery using algorithms.

**Historical Inspiration**: Named after Srinivasa Ramanujan, the legendary mathematician who produced thousands of formulas through intuition, the project asks: "Can we algorithmically generate mathematical conjectures the way Ramanujan did?"

**Project Goals**:
1. **Automated Discovery**: Generate conjectures for fundamental constants (π, e, Catalan's constant, ζ values)
2. **Systematic Organization**: Understand relationships between thousands of formulas
3. **Proof Generation**: Transform conjectures into rigorous mathematical proofs
4. **Unification**: Show that seemingly different formulas are actually equivalent

### The Package's Role

ramanujantools is **NOT** the discovery engine—it's the **analysis and verification infrastructure**. The relationship is:

- **Discovery Algorithms** (MITM, Gradient Descent, ESMA): Find numerical patterns → Generate conjectures
- **ramanujantools**: Analyze structures → Transform representations → Prove equivalences → Verify conjectures

Think of it as: Discovery algorithms are the **telescope** that spots new mathematical objects; ramanujantools is the **laboratory** that studies them.

---

## The Central Problem

### What Are We Trying to Do?

Given a fundamental mathematical constant like π or e, we want to find **continued fraction formulas** that converge to it:

$$\alpha = a_0 + \cfrac{b_1}{a_1 + \cfrac{b_2}{a_2 + \cfrac{b_3}{a_3 + \ddots}}}$$

where $a_n$ and $b_n$ are **polynomials in n** (hence "Polynomial Continued Fraction" or PCF).

### Why This Matters

1. **Irrationality Proofs**: PCFs with specific properties can prove a number is irrational
   - **Apéry's 1979 breakthrough**: Used a PCF to prove ζ(3) is irrational
   - This was the first new irrationality proof in decades

2. **Efficient Approximations**: PCFs generate rational approximations p/q with controlled error

3. **Hidden Structure**: Different-looking formulas may be related through transformations

4. **Computational Discovery**: We can search for PCFs numerically before understanding them theoretically

### The Scale of the Problem

The Ramanujan Machine has:
- Generated **millions** of PCF candidates through algorithmic search
- Validated **thousands** of formulas for various constants
- Discovered **hundreds** of new formulas not in mathematical literature
- Proven **equivalence** between 360+ formulas (94% of validated formulas)
- Analyzed **1.7 million** formulas in unsupervised clustering studies

---

## Mathematical Frameworks

### 1. Polynomial Continued Fractions (PCFs)

**Definition**: A continued fraction where $a_n$ and $b_n$ are polynomials in $n$.

**Matrix Representation**: Every PCF can be represented as a product of 2×2 matrices:

$$M(n) = \begin{pmatrix} 0 & b_n \\ 1 & a_n \end{pmatrix}$$

The convergent p_k/q_k at depth k is extracted from:

$$\begin{pmatrix} p_k \\ q_k \end{pmatrix} = \prod_{i=1}^{k} M(i) \cdot \begin{pmatrix} 1 \\ a_0 \end{pmatrix}$$

**Package Implementation**:
- `ramanujantools.pcf.PCF(a_n, b_n)`
- Can be constructed from polynomials, matrices, or linear recurrences
- Supports deflation (removing common factors) to canonical form

### 2. Conservative Matrix Fields (CMFs)

**The Key Insight**: In 2024, researchers realized that thousands of discovered PCFs were actually **1-dimensional slices** of higher-dimensional structures.

**Definition**: A CMF is a multi-dimensional matrix field satisfying a **conservation property**:

$$M_x(x,y) \cdot M_y(x+1,y) = M_y(x,y) \cdot M_x(x,y+1)$$

This means **matrix multiplication is path-independent**—you get the same result walking from (0,0) to (5,3) regardless of the path taken.

**Why This Matters**:
1. **Unification**: Different PCFs may come from different 1D trajectories through the same CMF
2. **Generalization**: Extends Apéry's ζ(3) proof technique to multiple dimensions
3. **Discovery**: Can explore "nearby" formulas by varying the trajectory
4. **Proof**: Path independence provides structural constraints useful for proving properties

**Example - Apéry's ζ(3) Formula**:

Instead of viewing this as a single PCF:
$$\text{PCF}(a_n = 34n^3 + 51n^2 + 27n + 5, \quad b_n = -n^6)$$

We understand it as a trajectory through a 2D CMF with matrices:
$$M_x, M_y \text{ satisfying conservation}$$

**Package Implementation**:
- `ramanujantools.cmf.CMF({x: Mx, y: My, ...})` with automatic conservation validation
- `work(start, end)`: Path-independent calculation between any two points
- `trajectory_matrix()`: Reduces multi-dimensional field to 1D trajectory
- Specialized constructors: `FFbar`, `pFq`, `MeijerG`, `DFinite`

### 3. Linear Recurrences

**Connection to PCFs**: Every PCF generates sequences via a linear recurrence:

$$\sum_{i=0}^{N} a_i(n) \cdot p(n-i) = 0$$

**Package Implementation**:
- `ramanujantools.LinearRecurrence` with inflation/deflation operations
- `unfold()`: Attempt to factor into lower-order recurrences
- `fold()`: Combine recurrences into higher-order ones
- Can be converted to/from companion matrices

### 4. Euler Family PCFs

**Definition**: A PCF is in the Euler family if there exist polynomials $h_1, h_2, f$ such that:

$$b(x) = -h_1(x)h_2(x)$$
$$f(x)a(x) = f(x-1)h_1(x) + f(x+1)h_2(x+1)$$

**Significance**: Euler family PCFs can be **transformed into infinite sum representations**:

$$\mathbb{K}_1^\infty \frac{b(i)}{a(i)} = \frac{f(1)h_2(1)}{f(0)} \left(\left(\sum_{k=0}^\infty \frac{f(0)f(1)}{f(k)f(k+1)} \prod_{i=1}^{k} \frac{h_1(i)}{h_2(i+1)}\right)^{-1} - 1\right)$$

This provides an **alternative form** that may be easier to analyze or prove properties about.

**Package Implementation**:
- `ramanujantools.solvers.EulerSolver.solve_for(a, b)` finds all Euler representations
- Returns `EulerSolution` objects with h₁, h₂, and f polynomials
- Algorithm: Enumerate factorizations of b, solve degree constraints, construct generic f

---

## Key Algorithms and Their Origins

### 1. The Kamidelta Algorithm

**Paper**: "On the Connection Between Irrationality Measures and Polynomial Continued Fractions" (arXiv:2111.04468, Arnold Mathematical Journal 2021)

**Problem**: Given a PCF, can we **predict** its irrationality measure **without knowing** what constant it converges to?

**Irrationality Measure Definition**:
$$\left|\frac{p_n}{q_n} - L\right| = \frac{1}{q_n^{1+\delta}}$$

where δ is the irrationality measure and L is the true limit.

**The Kamidelta Insight**: The value of δ can be predicted from:
1. **Eigenvalue analysis** of the companion matrix (approximation error)
2. **GCD behavior** of the sequence (denominator growth)

**Algorithm** (`Matrix.kamidelta(depth)`):
```python
def kamidelta(self, depth=20):
    # 1. Compute eigenvalues of companion matrix
    eigenvals = self.sorted_eigenvals()  # Poincaré form

    # 2. Predict approximation errors from eigenvalue ratios
    errors = []
    for i in range(1, len(eigenvals)):
        errors.append(log(|λ_0| / |λ_i|))

    # 3. Fit GCD slope: log(q_reduced) ~ slope * n
    q_reduced_values = [evaluate sequence at depths 1..depth]
    slope = linear_fit(q_reduced_values)

    # 4. Predict delta values
    deltas = [-1 + error/slope for error in errors]
    return deltas
```

**Why This Works**:
- **Eigenvalue ratios** determine convergence speed (how fast error decreases)
- **GCD slope** determines denominator growth rate
- **Delta** is the balance between these two rates

**Package Implementation**:
- `Matrix.kamidelta()` for matrices
- `PCF.kamidelta()` for continued fractions
- `LinearRecurrence.kamidelta()` for recurrences

### 2. Coboundary Transformations

**Paper**: "The conservative matrix field" (arXiv:2303.09318)

**Problem**: Given two matrix sequences $M_1(n)$ and $M_2(n)$, are they equivalent?

**Equivalence Definition**: They're **coboundary equivalent** if there exists $U(n)$ such that:
$$M_1(n) \cdot U(n+1) = U(n) \cdot M_2(n)$$

**Geometric Interpretation**: U is a "gauge transformation" that converts between coordinate systems.

**Significance**:
1. Two different-looking PCFs may be coboundary equivalent (same underlying structure)
2. CMF conservation can be verified via coboundary relations
3. Can transform PCFs to canonical forms

**Package Implementation**:
- `Matrix.coboundary(U, symbol)`: Compute coboundary transformation
- `CoboundarySolver.find_coboundary(M1, M2, max_deg)`: Find U connecting M1 and M2
- `CMF.coboundary(U)`: Apply gauge transformation to entire CMF

### 3. FFbar Construction

**Paper**: "The conservative matrix field" (arXiv:2303.09318)

**Method**: Build a 2D CMF from two functions $f(x,y)$ and $\bar{f}(x,y)$ satisfying:

**Linear Condition**:
$$f(x+1, y-1) - \bar{f}(x, y-1) + \bar{f}(x+1, y) - f(x, y) = 0$$

**Quadratic Condition**:
$$f\bar{f}(x, y) - f\bar{f}(x, 0) - f\bar{f}(0, y) + f\bar{f}(0, 0) = 0$$

**Resulting CMF**:
$$M_x = \begin{pmatrix} 0 & b(x) \\ 1 & a(x,y) \end{pmatrix}, \quad M_y = \begin{pmatrix} \bar{f}(x,y) & b(x) \\ 1 & f(x,y) \end{pmatrix}$$

**Package Implementation**:
- `ramanujantools.cmf.FFbar(f, fbar)`: Construct and validate
- `FFbarSolver.from_pcf(pcf)`: Find f, f̄ where M_x matches given PCF
- `FFbarSolver.solve_ffbar(f, fbar)`: Solve conditions for generic f, f̄

### 4. Hypergeometric and D-finite CMFs

**Papers**: "Conservative Matrix Fields: Continuous Asymptotics and Arithmetic" (arXiv:2507.08138)

**Framework**: Many CMFs arise from **D-finite functions** (solutions to linear differential equations).

**Examples**:
- **pFq functions**: Generalized hypergeometric functions
- **Meijer G-functions**: More general special functions
- Connection to **contiguous relations** in special function theory

**Package Implementation**:
- `ramanujantools.cmf.pFq(p, q, z)`: Construct pFq CMF from differential equation
- `ramanujantools.cmf.MeijerG(m, n, p, q, z)`: Meijer G-function CMF
- Abstract `DFinite` base class for custom D-finite CMFs
- `pFq.evaluate()`: Use CMF work calculations for symbolic evaluation

---

## The Kamidelta Algorithm Explained

Since this is a unique contribution of the Ramanujan Machine research, let's dive deeper.

### The Problem Kamidelta Solves

You have a PCF that converges to some unknown constant. You want to know:
1. How "good" is this formula? (How fast does it converge?)
2. Can it be used for an irrationality proof?
3. What's the quality of the approximations it generates?

Normally, you'd need to:
- Know what constant L it converges to
- Evaluate the PCF to high depth
- Measure the error numerically

**Kamidelta bypasses this**: It predicts δ using only the **algebraic structure** of the PCF.

### The Mathematical Theory

**Theorem (Simplified)**: For a linear recurrence with companion matrix M:

The convergence behavior is governed by the **eigenvalues** λ₀, λ₁, ..., λₙ of M:
- **Largest eigenvalue** λ₀ determines the leading term
- **Ratios** λ₀/λᵢ determine how fast correction terms vanish

In Poincaré form (rescaled so eigenvalues approach finite limits as n→∞):

$$\text{Error at depth n} \sim \left|\frac{\lambda_1}{\lambda_0}\right|^n$$

Meanwhile, the denominators grow as:

$$\log(q_n) \sim \text{slope} \cdot n$$

The irrationality measure is:

$$\delta = -1 + \frac{\log|\lambda_0/\lambda_1|}{\text{slope}}$$

### Implementation Details

**Step 1: Poincaré Characteristic Polynomial**

The companion matrix has eigenvalues that may grow with n. We rescale:

```python
def poincare_poly(poly):
    # Find degree of leading coefficient growth
    current_degree = 0
    for i, coeff in enumerate(poly.coeffs):
        degree = sp.degree(coeff.numerator) - sp.degree(coeff.denominator)
        if current_degree * i < degree:
            current_degree = ceil(degree / i)

    # Rescale coefficients
    rescaled = [coeff / n^(current_degree * i) for i, coeff in enumerate(coeffs)]
    return sp.Poly(rescaled).limit(n, oo)
```

**Step 2: GCD Slope Analysis**

Evaluate the PCF at multiple depths and track denominator growth:

```python
def gcd_slope(self, depth=20):
    q_reduced = []
    for d in range(1, depth):
        limit = self.limit(d)
        rational = limit.as_rational()
        q_reduced.append(log(rational.q))

    # Linear fit
    slope = np.polyfit(range(1, depth), q_reduced, deg=1)[0]
    return slope
```

**Step 3: Combine for Delta**

```python
def kamidelta(self, depth=20):
    errors = self.errors()  # log(|λ₀/λᵢ|)
    slope = self.gcd_slope(depth)
    return [-1 + err/slope for err in errors]
```

### Why This is Powerful

1. **Predictive**: Determines δ before knowing L
2. **Algebraic**: Uses structure, not numerical approximation
3. **Fast**: No need to evaluate to extreme depths
4. **Classification**: Can cluster formulas by predicted δ values

**Real-world use**: The "Unsupervised Discovery" paper (NeurIPS 2024) used kamidelta to analyze and cluster **1.7 million formulas** based on convergence patterns rather than numerical values.

---

## From Discovery to Proof

### The Discovery Pipeline

1. **Algorithmic Search** (MITM, Gradient Descent):
   - Search numerical space for patterns
   - Generate PCF candidates: (a_n, b_n) polynomials
   - Verify numerical convergence to target constant

2. **Structural Analysis** (ramanujantools):
   - Represent as matrices, recurrences, or CMFs
   - Check for Euler family membership
   - Compute irrationality measures
   - Find coboundary equivalences

3. **Theoretical Understanding**:
   - Prove why the formula works
   - Connect to known special functions
   - Generalize to higher dimensions (CMF framework)

4. **Applications**:
   - Generate new irrationality proofs
   - Find computational improvements
   - Discover hidden relationships

### Case Study: Apéry's Proof Generalization

**Original (Apéry, 1979)**: One specific PCF proves ζ(3) is irrational

**CMF Framework Discovery**:
- Apéry's PCF is actually a 1D trajectory through a 2D CMF
- The 2D structure provides **many** related formulas
- Conservation property gives additional constraints useful for proof

**Package Implementation**:
```python
from ramanujantools.cmf import zeta3

# Get the Apéry CMF
cmf = zeta3()

# Different trajectories give different formulas
trajectory1 = {x: 1, y: 0}  # Original Apéry direction
trajectory2 = {x: 1, y: 1}  # Alternative formula
trajectory3 = {x: 0, y: 1}  # Another alternative

# All converge to related expressions
```

**Result**: CMF framework **systematically generalizes** Apéry's technique to discover new irrationality proofs.

---

## Real-World Impact

### Scale of Applications

From analyzing the papers, ramanujantools has been used for:

1. **Euler2AI (NeurIPS 2025)**:
   - Processed **455,050 arXiv papers** to harvest formulas
   - Validated **385 formulas** for mathematical constants
   - Proved **360 equivalences** (94% of validated formulas)
   - Connected **166 formulas** to a single mathematical object
   - Used `CoboundarySolver` and `EulerSolver` for proving equivalences

2. **Unsupervised Discovery (NeurIPS 2024)**:
   - Analyzed **1,768,900 PCF formulas**
   - Used kamidelta for convergence-based clustering
   - Discovered new formulas for π, ln(2), Gauss' constant, Lemniscate constant
   - Pattern identification without numerical target values

3. **The Ramanujan Library (ICLR 2025)**:
   - Organized relationships in a hypergraph structure
   - Discovered **75 new formulas** using PSLQ (`Limit.identify()`)
   - Created searchable database of constant relationships

### Scientific Publications

ramanujantools has contributed to research published in:
- **Nature** (2021): Foundational Ramanujan Machine paper
- **PNAS** (2024): Conservative Matrix Fields introduction
- **Arnold Mathematical Journal** (2024): Irrationality measures
- **NeurIPS** (2024, 2025): Multiple applications
- **ICLR** (2025): Formula library

### Mathematical Contributions

**New Irrationality Proofs**: Generalizations of Apéry's technique

**Formula Unification**: Showing that thousands of formulas are related

**Computational Methods**: Efficient algorithms for:
- Symbolic verification at scale
- Pattern detection in formula spaces
- Equivalence checking between representations

**Theoretical Understanding**: CMF framework provides unified theory for:
- Polynomial continued fractions
- D-finite sequences
- Hypergeometric functions
- Contiguous relations

---

## Philosophical Perspective

### Why This Matters for Mathematics

**Traditional Mathematical Discovery**:
1. Genius makes intuitive leap (Ramanujan's notebooks)
2. Years later, mathematicians prove and understand
3. Connections emerge slowly over decades

**Algorithmic Discovery** (Ramanujan Machine approach):
1. Algorithms generate thousands of conjectures
2. ramanujantools provides tools to analyze structures
3. Patterns emerge quickly, guiding theoretical work
4. Theory and discovery inform each other in tight loop

### The Role of Computation

ramanujantools represents a **new paradigm**:
- Not just numerical verification
- Not just symbolic manipulation
- **Structural analysis** at scale

It's a bridge between:
- **Computational exploration** (what formulas exist?)
- **Mathematical understanding** (why do they work?)
- **Rigorous proof** (how can we prove it?)

### Future Directions

The papers suggest ongoing work in:
1. **Higher-dimensional CMFs**: Beyond 2D to arbitrary dimensions
2. **Automated proof generation**: From discovery to rigorous proof
3. **AI-assisted conjecture**: Combining neural networks with symbolic tools
4. **Unification theory**: Understanding the "space" of all formulas

ramanujantools is the **computational laboratory** making this research possible.

---

## Summary

ramanujantools is not just a software library—it's the **computational manifestation** of a new approach to mathematical discovery:

**Core Ideas**:
- PCFs and CMFs provide structure for organizing formulas
- Eigenvalue analysis (kamidelta) predicts properties without numerical evaluation
- Coboundary transformations reveal hidden equivalences
- D-finite framework connects to classical special function theory

**Unique Contributions**:
- First implementation of CMF theory
- Kamidelta algorithm for irrationality measure prediction
- Scalable symbolic verification infrastructure
- Integration of multiple mathematical frameworks (PCFs, recurrences, CMFs, D-finite)

**Research Impact**:
- Enabling discovery at scale (millions of formulas)
- Contributing to peer-reviewed publications in top venues
- Providing tools for rigorous mathematical proof
- Advancing automated conjecture generation

When you use ramanujantools, you're not just doing computation—you're participating in a **new way of doing mathematics** that combines algorithmic discovery, symbolic analysis, and theoretical insight.

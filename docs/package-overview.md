# Ramanujantools Package Overview

A comprehensive guide to the structure, architecture, and usage of the ramanujantools library.

## **Overview**

`ramanujantools` is a sophisticated Python library for discovering mathematical conjectures involving constants like e, π, Catalan's constant, Euler's constant, and Riemann Zeta values. It provides tools for working with three main mathematical structures: Polynomial Continued Fractions (PCFs), Linear Recurrences, and Conservative Matrix Fields (CMFs).

---

## **CORE ARCHITECTURE**

### **1. Foundation Layer - Core Mathematical Objects**

#### **Position** (`position.py`)
- **Purpose**: Represents points in multi-dimensional space, also used as substitution dictionaries
- **Features**:
  - Algebraic operations (addition, subtraction, scalar multiplication)
  - Vector norms (longest/shortest element)
  - Type checking (is_polynomial, is_integer)
  - Sign extraction
  - Denominator LCM calculation
- **Usage Pattern**: `Position({x: 1, y: 2})` supports operations like `2 * position + Position({x: 3})`

#### **Matrix** (`matrix.py`)
- **Purpose**: Extended sympy.Matrix with symbolic manipulation and numeric evaluation
- **Key Features**:
  - **Walk operation**: Multiply matrix sequences along trajectories: `M.walk(trajectory={n: 1}, iterations=[10, 20], start={n: 0})`
  - **Companion form**: Convert matrices to companion representation via `as_companion()`
  - **Coboundary relations**: `M.coboundary(U, symbol)` computes U⁻¹(n)·M(n)·U(n+1)
  - **Eigenvalue analysis**: Poincare characteristic polynomials, sorted eigenvalues
  - **Delta prediction**: `kamidelta()` predicts irrationality measures
  - **Fast evaluation**: Uses FLINT for performance-critical operations
- **Performance Optimization**:
  - `subs()` internally uses `xreplace()` for speed
  - Automatic routing to NumericMatrix for numeric walks
  - Automatic routing to SymbolicMatrix with FLINT for symbolic operations

#### **LinearRecurrence** (`linear_recurrence.py`)
- **Purpose**: Represents linear recurrences Σaᵢ(n)p(n-i) = 0
- **Construction**: From coefficient lists or companion matrices
- **Operations**:
  - **Inflation/Deflation**: Multiply/divide by polynomials while preserving limits
  - **Folding**: Combine recurrences into higher-order ones
  - **Unfolding**: Decompose into lower-order recurrences (with generic polynomial inflation)
  - **Composition**: Compose two recurrences
  - **Normalization**: Remove common factors and set leading coefficient to 1
- **Key Methods**:
  - `limit(iterations, start)`: Calculate convergence limits
  - `evaluate_solution(initial_values, start, end)`: Generate sequence values
  - `unfold(inflation_degree)`: Attempts to factor into lower-order recurrence

#### **Limit** (`limit.py`)
- **Purpose**: Represents convergence results from matrix walks
- **Structure**: Stores `current` and `previous` matrices for precision estimation
- **Extraction**:
  - `as_rational()`: Returns p/q as sympy.Rational
  - `as_float()`: High-precision mpmath float
  - `as_rounded_number()`: Shortest representation within error bounds
  - `precision()`: Digits of accuracy
- **Analysis**:
  - `delta(L)`: Irrationality measure |p/q - L| = 1/q^(1+δ)
  - `identify(L)`: Use PSLQ to find integer relation for constant L
  - `identify_rational()`: Find integer linear combinations

---

### **2. PCF Layer - Polynomial Continued Fractions**

#### **PCF** (`pcf/pcf.py`)
- **Representation**: Continued fractions with polynomial terms a_n and b_n
- **Construction**:
  - From (a_n, b_n) pairs
  - From LinearRecurrence objects
  - From 2×2 matrices
- **Key Operations**:
  - `deflate_all()`: Remove maximum common content from a_n and b_n
  - `limit(iterations, start)`: Evaluate convergence including a₀ term
  - `M()`: Get 2×2 matrix representation
  - `A()`: Get initial a₀ matrix [[1, a₀], [0, 1]]
  - `delta(depth, L)`: Calculate irrationality measure
  - `kamidelta(depth)`: Predict delta from eigenvalue analysis

#### **Hypergeometric** (`pcf/hypergeometric.py`)
- **Classes**: `Hypergeometric1F1Limit`, `Hypergeometric2F1Limit`
- **Purpose**: Convert PCFs with specific degree patterns to hypergeometric form
- **Degrees**: (1,1) → 1F1, (1,2) → 2F1
- **Features**:
  - `limit()`: Symbolic evaluation via sympy hypergeometric functions
  - `as_mathematica_prompt()`: Generate Mathematica code for evaluation
  - Automatic parameter extraction (α, β, z)

---

### **3. CMF Layer - Conservative Matrix Fields**

#### **CMF** (`cmf/cmf.py`)
- **Definition**: Multi-dimensional matrix field satisfying conservation: Mx(x,y)·My(x+1,y) = My(x,y)·Mx(x,y+1)
- **Structure**: Dictionary mapping axes (symbols) to matrices
- **Validation**: Automatic conservation checking unless `validate=False`
- **Key Operations**:
  - `work(start, end)`: Calculate transformation matrix between two points
  - `trajectory_matrix(trajectory, start)`: Matrix for single-variable trajectory
  - `walk(trajectory, iterations, start)`: Multiply trajectory matrices
  - `limit(trajectory, iterations, start)`: Get convergence limits
  - `sub_cmf(basis)`: Extract sub-CMF along trajectory basis
  - `coboundary(U)`: Apply coboundary transformation
  - `dual()`: Get inverse-transpose CMF
- **Algorithm**: Diagonal decomposition with backtracking for singularity avoidance

#### **FFbar** (`cmf/ffbar.py`)
- **Construction**: 2D CMF from functions f(x,y) and f̄(x,y)
- **Conditions**:
  - **Linear**: f(x+1,y-1) - f̄(x,y-1) + f̄(x+1,y) - f(x,y) = 0
  - **Quadratic**: ff̄(x,y) - ff̄(x,0) - ff̄(0,y) + ff̄(0,0) = 0
- **Matrices**:
  - Mx = [[0, b(x)], [1, a(x,y)]] where a = f - f̄(x+1,y)
  - My = [[f̄(x,y), b(x)], [1, f(x,y)]] where b = ff̄(x,0) - ff̄(0,0)

#### **DFinite** (`cmf/d_finite.py`)
- **Purpose**: Abstract base for CMFs derived from D-finite functions
- **Subclasses Must Implement**:
  - `axes_and_signs()`: Axes and increment directions
  - `differential_equation()`: Defining differential equation
  - `construct_matrix()`: Build matrix from θ-matrix
- **Workflow**: Constructs companion matrix from differential equation

#### **pFq** (`cmf/pfq.py`)
- **Represents**: Generalized hypergeometric function ₚFᵧ
- **Axes**: x₀,...,x_{p-1} (numerator params), y₀,...,y_{q-1} (denominator params)
- **Differential Equation**: θ∏(θ+yᵢ-1) = z∏(θ+xᵢ)
- **Special Methods**:
  - `ascend()`: Extend to ₚ₊₁Fᵧ₊₁ preserving delta
  - `state_vector()`: Calculate [ₚFᵧ, θₚFᵧ, θ²ₚFᵧ, ...]
  - `evaluate()`: Symbolic evaluation via contiguous relations
  - `predict_rank()`: Determine matrix dimension

#### **MeijerG** (`cmf/meijer_g.py`)
- **Represents**: Meijer G-function G^{m,n}_{p,q}
- **Parameters**: m, n, p, q with constraints 0≤n≤p, 0≤m≤q
- **Axes**: a₀,...,a_{p-1}, b₀,...,b_{q-1}
- **Construction**: Via differential equation similar to pFq

---

### **4. Solver Layer**

#### **EulerSolver** (`solvers/euler_solver.py`)
- **Problem**: Given a(x), b(x), find h₁(x), h₂(x), f(x) such that:
  - b(x) = -h₁(x)h₂(x)
  - f(x)a(x) = f(x-1)h₁(x) + f(x+1)h₂(x+1)
- **Algorithm**:
  1. Factor b(x) into all possible products
  2. For each factorization, determine leading coefficients
  3. Solve for f(x) degree using characteristic coefficients
  4. Construct generic f and solve linear system
- **Output**: `EulerSolution` dataclass with h₁, h₂, a, b, f
- **Use Case**: Transforms PCF to infinite sum representation

#### **CoboundarySolver** (`solvers/coboundary_solver.py`)
- **Problem**: Find m(x) such that m₁(x)·m(x+1) = m(x)·m₂(x)
- **Method**:
  - Create generic polynomial matrix with free variables
  - Solve polynomial equation system
  - Return matrix and remaining free variables
- **Applications**:
  - Verify CMF conservation
  - Find transformations between matrix representations

#### **FFbarSolver** (`solvers/ffbar_solver.py`)
- **Two Modes**:
  1. `from_pcf(pcf)`: Find f, f̄ where Mx matches PCF matrix
  2. `solve_ffbar(f, fbar)`: Given generic f, f̄, solve conditions
- **Algorithm**:
  - Generate generic polynomials with symbolic coefficients
  - Extract polynomial coefficients from linear/quadratic conditions
  - Solve system symbolically
  - Return all valid FFbar CMFs

---

### **5. Performance Layer - FLINT Integration**

#### **FlintContext** (`flint_core/context.py`)
- **Purpose**: Manage polynomial rings (integer or rational coefficients)
- **Types**:
  - `fmpz_mpoly_ctx`: Integer polynomials
  - `fmpq_mpoly_ctx`: Rational polynomials
- **Creation**: `flint_ctx(symbols, fmpz=True/False)`
- **Ordering**: Lexicographic ordering of variables

#### **FlintRational** (`flint_core/rational.py`)
- **Representation**: Numerator/denominator pairs with automatic GCD reduction
- **Operations**: Addition, multiplication, division with rational optimization
- **Performance**: Special GCD handling for rational coefficients

#### **SymbolicMatrix** (`flint_core/symbolic_matrix.py`)
- **Purpose**: Matrix of FlintRational for symbolic operations
- **Key Operation**: Fast `walk()` implementation for symbolic trajectories
- **Performance**: Polynomial arithmetic via FLINT (10-100x faster than sympy)
- **Conversion**: `from_sympy()` and `factor()` for seamless integration

#### **NumericMatrix** (`flint_core/numeric_matrix.py`)
- **Purpose**: High-precision rational matrix arithmetic
- **Key Feature**: `lambda_from_rt()` creates compiled evaluator
- **Performance**: Pre-compiles symbolic matrix to callable function
- **Walk Implementation**: Fast numeric matrix multiplication with flint.fmpq

---

### **6. Utility Layer**

#### **Batched Decorator** (`utils/batched.py`)
- **Purpose**: Unified API accepting scalar or list inputs
- **Type**: `Batchable[X] = X | list[X]`
- **Behavior**:
  - If scalar provided: wraps in list, calls function, unwraps result
  - If list provided: passes through directly
- **Usage**: `@batched("iterations")` on function parameters

#### **Generic Polynomial** (`generic_polynomial.py`)
- **Functions**:
  - `of_degree(deg, var_name, s)`: Create generic poly with symbolic coefficients
  - `of_combined_degree(deg, var_name, variables)`: Multi-variable generic poly
  - `symmetric_polynomials(*exprs)`: Elementary symmetric polynomials
  - `as_symmetric(poly, symbols)`: Express in symmetric polynomial basis

---

## **TYPICAL USAGE WORKFLOWS**

### **Workflow 1: Analyze a PCF**
```python
from ramanujantools.pcf import PCF
import sympy as sp
from sympy.abc import n

# Define PCF
pcf = PCF(a_n=n, b_n=-(n-1)*n)

# Deflate to simplest form
pcf = pcf.deflate_all()

# Calculate convergence
limits = pcf.limit([10, 20, 30], start=1)
print(limits[0].as_float())  # Approximation at depth 10
print(limits[0].precision())  # Digits of precision

# Predict irrationality measure
delta = pcf.kamidelta(depth=20)
```

### **Workflow 2: Construct and Analyze CMF**
```python
from ramanujantools.cmf import CMF
from ramanujantools import Matrix
from sympy.abc import x, y

# Define CMF matrices
Mx = Matrix([[0, x], [1, x+y]])
My = Matrix([[y, x], [1, x]])
cmf = CMF({x: Mx, y: My})

# Walk trajectory
limits = cmf.limit(
    trajectory={x: 1, y: 1},
    iterations=[10, 20],
    start={x: 0, y: 0}
)

# Calculate delta
delta = cmf.delta(
    trajectory={x: 1, y: 1},
    depth=20,
    start={x: 0, y: 0}
)
```

### **Workflow 3: Solve for Euler Form**
```python
from ramanujantools.solvers import EulerSolver
from sympy.abc import x

a = x**2 + 2*x + 1
b = -x*(x+1)

solutions = EulerSolver.solve_for(a, b)
for sol in solutions:
    print(f"h1={sol.h_1}, h2={sol.h_2}, f={sol.f}")
```

### **Workflow 4: Find FFbar Construction**
```python
from ramanujantools.solvers import FFbarSolver
from ramanujantools.pcf import PCF
from sympy.abc import n

pcf = PCF(a_n=n, b_n=-n*(n-1))
ffbars = FFbarSolver.from_pcf(pcf)

for ffbar in ffbars:
    print(f"f={ffbar.f}, fbar={ffbar.fbar}")
```

### **Workflow 5: Use pFq CMF**
```python
from ramanujantools.cmf import pFq
from ramanujantools import Position
import sympy as sp

# Create 2F1 CMF
cmf = pFq(p=2, q=1, z=sp.Rational(1, 2))

# Define trajectory and starting point
trajectory = Position({'x0': 0, 'x1': 1, 'y0': -1})
start = Position({'x0': 1, 'x1': 2, 'y0': 3})

# Calculate trajectory matrix
M = cmf.trajectory_matrix(trajectory, start)

# Evaluate specific pFq value
value = pFq.evaluate([1, 2], [3], sp.Rational(1, 2))
```

---

## **KEY DESIGN PATTERNS**

1. **Dual Representation**: Objects maintain both symbolic (sympy) and numeric (FLINT/mpmath) representations
2. **Lazy Evaluation**: Operations return new objects, actual computation deferred until needed
3. **Batching**: Most methods accept scalar or list inputs via `@batched` decorator
4. **Context Management**: FLINT contexts track variable sets for polynomial operations
5. **Conservation Validation**: CMFs validate mathematical constraints at construction
6. **Walk Optimization**: Automatic path finding through singularities in CMF work calculations

---

## **PERFORMANCE CHARACTERISTICS**

- **Matrix walks**: O(n) for depth n, with FLINT providing ~10x speedup for symbolic, ~100x for numeric
- **PCF limits**: Batched computation reuses intermediate results
- **CMF trajectory matrices**: Diagonal decomposition reduces complexity
- **Solver algorithms**: Exponential in factorization count (EulerSolver), polynomial in coefficients (FFbarSolver)

---

## **IMPORTANT CONVENTIONS**

1. **Reserved Symbol `n`**: The symbol `n` is reserved for PCF/recurrence operations and should not be used as a CMF axis
2. **Matrix Substitution**: `Matrix.subs()` internally uses `xreplace()` for performance, not the full sympy `subs()`
3. **Test Naming**: Test files follow the pattern `*_test.py`
4. **Validation**: CMFs and matrices validate constraints at construction time by default
5. **Caching**: Heavy use of `@lru_cache` and `@cached_property` for expensive computations
6. **Position as Dict**: Position objects are dicts and can be used directly in substitutions

---

## **MATHEMATICAL BACKGROUND**

### **Polynomial Continued Fractions**
A PCF is a continued fraction where both the numerator and denominator terms are polynomials in n:

$$a_0 + \cfrac{b_1}{a_1 + \cfrac{b_2}{a_2 + \cfrac{b_3}{a_3 + \ddots}}}$$

PCFs can converge to mathematical constants and can be represented as 2×2 matrix products.

### **Conservative Matrix Fields**
A CMF is a multi-dimensional matrix field where matrices commute along different axes:
$$M_x(x,y) \cdot M_y(x+1,y) = M_y(x,y) \cdot M_x(x,y+1)$$

This conservation property allows for path-independent calculations and powerful transformations.

### **Euler Form**
A PCF is in Euler form if there exist polynomials h₁, h₂, f such that:
- $b(x) = -h_1(x)h_2(x)$
- $f(x)a(x) = f(x-1)h_1(x) + f(x+1)h_2(x+1)$

This allows the PCF to be expressed as an infinite sum, enabling closed-form evaluation.

---

## **INTEGRATION WITH EXTERNAL TOOLS**

- **SymPy**: Core symbolic computation engine
- **mpmath**: High-precision arithmetic for limit evaluation
- **FLINT**: Fast polynomial arithmetic (via python-flint)
- **gmpy2**: Multiple precision arithmetic library
- **NumPy**: Array operations for eigenvalue analysis
- **tqdm**: Progress bars for long computations

---

## **ARCHITECTURAL LAYERS SUMMARY**

```
┌─────────────────────────────────────────────────┐
│              Application Layer                  │
│  (User Scripts, Notebooks, Research Code)       │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│               Solver Layer                      │
│  EulerSolver | CoboundarySolver | FFbarSolver   │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│          Domain-Specific Layer                  │
│      PCF | CMF | FFbar | pFq | MeijerG         │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│            Foundation Layer                     │
│  Position | Matrix | LinearRecurrence | Limit   │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│          Performance Layer                      │
│  FlintContext | SymbolicMatrix | NumericMatrix  │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│           External Libraries                    │
│  sympy | mpmath | flint | gmpy2 | numpy         │
└─────────────────────────────────────────────────┘
```

This package represents a sophisticated mathematical research tool with deep integration between symbolic mathematics, numeric computation, and specialized algorithms for discovering mathematical constants and their representations.

# SequentialConvexRelaxation.jl

An introduction to the package can also be found [here](https://rdoelman.bitbucket.io/2019/09/09/SequentialConvexRelaxationjl.html).

## Installation
```julia-repl
(v1.1) pkg> add https://github.com/rdoelman/SequentialConvexRelaxation.jl
```

## What does this package do?
This package gives a convenient approach to attempt to solve nonconvex optimization problems that can be expressed as convex problems with additional bilinear equality constraints.

Bilinear equality constraints are constraints of the form A(x) * P * B(x) == C(x), where A(x) and B(x) are (matrix valued) decision variables (any `AbstractExpr` in the Convex.jl modelling framework), P is a constant matrix and C(x) is either a decision variable or a constant matrix (again, any `AbstractExpr`).
The additional bilinear equality constraints make the overall optimization problem (in general) non-convex and non-linear.
This package uses a convex **heuristic** approach to find good and feasible solutions.

The advantages:
- The ease of use of Convex.jl to model your optimization problem.
- Only an SDP solver needed (LP solver only in some cases).
- No initial feasible guess is necessary for the solver to start.
- Only 1 regularization parameter to tune (if necessary) for every bilinear equality constraint.

The disadvantages:
- No convergence guarantees (the problem is non-convex and non-linear after all). No guarantees of finding global optima, or feasible solutions.
- High computational complexity of iterations.
- Depending on the solver and the size of the problem, numerical issues in the solvers can spoil the fun.

## What kind of problems can I try to solve with this?
The class of problems that can be formulated as "convex + bilinear (or quadratic) equality constraints" is very large. Although some of these have elegant solutions, often one has to look for specific solvers.

Some examples of "convex + bilinear equality constraints" include:
- matrix factorization
- binary variables + LP/QP/SDP
- sudokus (see the examples folder)

More examples are listed [here](https://rdoelman.bitbucket.io/2019/09/09/SequentialConvexRelaxationjl.html).
If you have an interesting example that you want to share, please do not hesitate to get in touch!

## How do I use it?
In this example we try to find the minimizers for the nonconvex [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function).
```julia
using Convex
using SequentialConvexRelaxation
using SCS

# We're minimizing the Rosenbrock function with standard values for a and b.
# f(x,y) = (a-x)^2 + b*(y-x^2)^2
a = 1.
b = 100.

# Another way to write the function is (a-x)^2 + b*z^2 where y-z = x^2.
x = Variable()
y = Variable()
z = Variable()

# This is the convex part
p = minimize(norm([a-x; sqrt(b)*z]))

# Constructing the bilinear equality constraint
# In general this is of the form A*P*B=C.
# Here A=B=x, P=1., C=y-z . X and Y are guesses for -A and -B.
x0 = -1 # some guess for the optimal x
λ = 0.1 # this is a regularization parameter
bc = BilinearConstraint(x,1.,x,y-z,X=-x0,Y=-x0,λ=λ)

# The bilinear problem is the convex problem with the bilinear constraint
bp = BilinearProblem(p,bc)

# Call the solve!() function on the bilinear problem
r = solve!(bp,() -> SCS.Optimizer(verbose=0),iterations=2)

# Use Convex.jl to inspect the result. It turns out to be the global minimizer
xopt = evaluate(x)
yopt = evaluate(y)
```

To solve the problem with `Mosek` instead, replace the `solve!` call above with
```julia
using MosekTools
r = solve!(bp,() -> Mosek.Optimizer(QUIET=true),iterations=2)
```

For more examples, check the files in the 'examples' folder. It contains an interactive demo of exactly what the method does, an interactive demo with the Rosenbrock function and a demo on solving a sudoku puzzle.

The package exposes 4 functions: ``BilinearConstraint``, ``BinaryConstraint``,``BilinearProblem`` and ``solve!``.

A ``BilinearConstraint(A,P,B,C)`` is a constraint of the form A(x) * P * B(x) = C(x), where A(x) is a variable (constructed using Convex.jl), P is a constant (scalar or matrix), B(x) is a variable, C(x) is variable or a constant (scalar or matrix).

A ``BinaryConstraint(A)`` is a constraint of the form  A ∈ {0,1}.

A ``BilinearProblem(p,bc)`` consists of two elements: a convex problem ``p`` (as in ``p = Convex.minimize(x)``), and (a vector of) additional bilinear equality constraints ``bc`` (making the complete problem non-convex in general).

``solve!(bp, solver)`` then attempts to solve the non-linear non-convex problem with a heuristic method (no guarantees here). It returns a structure with the result, that is also stored in a field of the ``BilinearProblem``.

The values of the variables can be inspected using Convex.jl's ``evaluate(x)``.
It is recommended to verify whether the bilinear equality constraints hold!

More description on how the functions work can be found using ``?functionname``.

## How does it mathematically work?
This package attempts to solve problems in the "convex + bilinear equality constraint" class and it constructs a regularized convex problem that is solved with conventional convex optimization solvers.
It does so by reformulating the problem to a rank constraint.
The package constructs a convex heuristic problem to handle this rank constraint, using the current guesses for the variables in the constraints (if no guess is supplied, a randomly generated matrix is used).
The resulting heuristic problem uses semidefinite constraints, so the convex optimization solver needs to be able to handle those. Use for example SCS or Mosek.
The solution provides new guesses for the optimal values, and the process is iterated for a specified number of iterations.

More details can also be found [here](https://rdoelman.bitbucket.io/2019/09/09/SequentialConvexRelaxationjl.html).

For mathematical background, see:
Doelman, Reinier, and Michel Verhaegen. "Sequential convex relaxation for convex optimization with bilinear matrix equalities." 2016 European Control Conference (ECC). IEEE, 2016.

## Citing the package and/or method
The method:
```
@inproceedings{doelman2016sequential,
  title={Sequential convex relaxation for convex optimization with bilinear matrix equalities},
  author={Doelman, Reinier and Verhaegen, Michel},
  booktitle={2016 European Control Conference (ECC)},
  pages={1946--1951},
  year={2016},
  organization={IEEE}
}
```

The software:
```
@misc{SequentialConvexRelaxation,
  author={Reinier Doelman},
  title={SequentialConvexRelaxation.jl},
  howpublished={\url{https://github.com/rdoelman/SequentialConvexRelaxation.jl/}},
  year={2019}
}
```

# Compatibility issues
Currently Convex.jl and many solvers are in a process to switch to different optimization interface packages, which might break this package's functionality.
Currently the tested versions are:
- Convex.jl v0.13.1
- MosekTools.jl v0.9.3
- SCS.jl v0.6.6

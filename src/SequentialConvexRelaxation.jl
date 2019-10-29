module SequentialConvexRelaxation

# Reinier Doelman, 2/9/2019

using Convex
import Convex.solve!
using LinearAlgebra
import MathProgBase

export solve!, BilinearProblem, BilinearConstraint, BinaryConstraint

const InputType = Union{AbstractVecOrMat{T},T,Complex{T}} where T <: Union{AbstractFloat,Integer}

"""
    BilinearConstraint(A::AbstractExpr,
        P  :: Union{Complex{T}, AbstractArray{T,2}, AbstractArray{T,1}, T} where T<:AbstractFloat
        B  :: AbstractExpr,
        C  :: AbstractExpr;
        X  :: AbstractExpr,
        Y  :: AbstractExpr,
        W1 :: AbstractExpr,
        W2 :: AbstractExpr,
        λ  :: AbstractFloat)

A bilinear constraint is a constraint in the form
    A*P*B=C,
where A and B are decision variables in an optimization problem, C is either a decision variable or a constant (or constant matrix), and where P is a constant (or constant matrix).
A and B can be the same variable (e.g. x^2 = 1).
A,B,C and P can be scalars or matrices, as long as the dimensions match.

X and Y are constants, where size(X)=size(A) and size(Y)=size(B).
By default they are initialized with randomly generated matrices, but they can optionally be supplied to the function.
Note that if one has a guess for A and/or B, one is encouraged to set X=-Aguess, Y=-Bguess.

W1 and W2 are square, invertible matrices that function as a sort of 'weighting' of the problem.
They are set to identity matrices by default. Their sizes should correspond to the size of C such that W1*C*W2 exists.

λ is a scalar regularization parameter.
For simplicity's sake, think of it 'as if' we form the sum
    f + ∑ λi || Ci - Ai*Pi*Bi ||,
where f is the objective function of the convex problem.
Tuning guidelines: from experience I'd say, a high λ gives a feasible solution quicker, but with a higher f*, and a low λ gives a feasible solution after more iterations, with lower f*.
A λ too low might give unbounded objective functions or not a feasible solution at all,
Feasible solutions (or globally optimal, for that matter), are not guaranteed.
λ defaults to 1.

See also: [`BinaryConstraint`]

# Example:
```julia-repl
julia> A = Variable()
julia> B = Variable()
julia> BilinearConstraint(A,1.,B,1.) # A*1.*B=1.
```
"""
mutable struct BilinearConstraint
    # Matrices and Convex.jl expressions that characterize the problem APB=C
    A::AbstractExpr
    P::InputType
    B::AbstractExpr
    C::AbstractExpr

    # These are the `bias' matrices in SCR
    X::AbstractExpr
    Y::AbstractExpr

    # Weighting matrices in SCR
    W1::AbstractExpr
    W2::AbstractExpr

    # Regularization parameter
    λ::AbstractFloat

    function BilinearConstraint(A::AbstractExpr,
        P::InputType,
        B::AbstractExpr,
        C::Union{AbstractExpr,InputType};
        X::Union{InputType,Nothing}=nothing,
        Y::Union{InputType,Nothing}=nothing,
        W1::Union{InputType,Nothing}=nothing,
        W2::Union{InputType,Nothing}=nothing,
        λ::AbstractFloat=1.)

        X = _biasvalidation(X,A) # X will be a Variable whose value is fixed.
        Y = _biasvalidation(Y,B) # Y will be a Variable whose value is fixed.

        C = C isa Number ? C = fill(C, (1, 1)) : C
        C = C isa AbstractVector ? C = reshape(C,length(C),1) : C
        C = C isa InputType ? C = Constant(C) : C # C will be an AbstractExpr, either variable or constant

        W1,W2 = _weightvalidation(W1,W2,C) # W1 and W2 will be Variables whose values are fixed.

        P = P isa Number ? P = fill(P, (1, 1)) : P
        P = P isa AbstractVector ? P = reshape(P,length(P),1) : P

        @assert size(A,1) == size(C,1) &&
            size(B,2) == size(C,2) &&
            size(A,2) == size(P,1) &&
            size(P,2) == size(B,1) "A,P,B and C are not of the correct size such that APB = C is defined."

        new(A,P,B,C,X,Y,W1,W2,λ)
    end
end

function _biasvalidation(X::Union{InputType,Nothing},A::AbstractExpr)
    T = sign(A) == ComplexSign() ? Complex{Float64} : Float64

    X isa Nothing ? X = 1E-2*randn(T,size(A)) :
    X isa Number ? X = fill(X, (1, 1)) :
    X isa AbstractVector ? X = reshape(X,length(X),1) :

    size(X) == size(A) || throw(DimensionMismatch("Variable and value sizes do not match!"))

    Xn = sign(A) == ComplexSign() ? ComplexVariable(size(A)) : Variable(size(A))

    size(X) == (1,1) ? fix!(Xn,X[1]) : fix!(Xn,X)

    return Xn
end

function _weightvalidation(W1::Union{InputType,Nothing},W2::Union{InputType,Nothing},C::AbstractExpr)
    T = sign(C) == ComplexSign() ? Complex{Float64} : Float64
    n,m = size(C)

    W1 = W1 isa Nothing ? W1 = Matrix{T}(I,n,n) : W1
    W2 = W2 isa Nothing ? W2 = Matrix{T}(I,m,m) : W2

    W1 = W1 isa Number ? W1 = fill(W1, (1, 1)) : W1
    W2 = W2 isa Number ? W2 = fill(W2, (1, 1)) : W2

    W1 = W1 isa AbstractVector ? W1 = reshape(W1,1,1) : W1
    W2 = W2 isa AbstractVector ? W2 = reshape(W2,1,1) : W2

    size(W1) == (n,n) || throw(DimensionMismatch("Weight matrix W1 must be square of size nxn where C = n x m!"))
    size(W2) == (m,m) || throw(DimensionMismatch("Weight matrix W2 must be square of size mxm where C = n x m!"))

    @assert rank(W1) == n "W1 is not full rank!"
    @assert rank(W2) == m "W2 is not full rank!"

    W1n = sign(C) == ComplexSign() ? ComplexVariable(n,n) : Variable(n,n)
    W2n = sign(C) == ComplexSign() ? ComplexVariable(m,m) : Variable(m,m)

    size(W1) == (1,1) ? fix!(W1n,W1[1]) : fix!(W1n,W1)
    size(W2) == (1,1) ? fix!(W2n,W2[1]) : fix!(W2n,W2)

    return W1n,W2n
end

"""
    BinaryConstraint(A::AbstractExpr;
            X::Union{InputType,Nothing}=nothing,
            W1::Union{InputType,Nothing}=nothing,
            λ::Float64 = 1.)

A variable that should take the value 0 or 1 is a binary value, and this is a special case of a bilinear constraint.
This function is here to produce a constraint A ∈ {0,1}.

See also: [`BilinearConstraint`]

# Example:
```julia-repl
julia> A = Variable()
julia> BinaryConstraint(A)
```
"""
function BinaryConstraint(A::AbstractExpr;
            X::Union{InputType,Nothing}=nothing,
            W1::Union{InputType,Nothing}=nothing,
            λ::Float64 = 1.)

    return BilinearConstraint(2.0*A-1.,1.,2.0*A-1.,1.0,X=2.0*X-1.,Y=2.0*X-1.,W1=W1,W2=W1,λ=λ)
end

"""
    Result(iterations::Int,
        update_weights::Bool,
        objective_values::Vector{AbstractFloat},
        constraint_violations,
        tracked_variables_values)

Records the result of solve!(), including the number of iterations, whether the matrices W1,W2 were update, what the objective values were, what the (bilinear) constraint violations were during the optimization, and what the values were of the variables that were tracked during the optimization.
"""
mutable struct Result
    iterations::Int
    update_weights::Bool
    objective_values::Vector{AbstractFloat}
    constraint_violations
    tracked_variables_values
end

"""
    BilinearProblem(
        convexproblem::Convex.Problem,
        bilinearconstraints::Vector{BilinearConstraint},
        result::Result)

A bilinear problem for (for the purpose of this package) is a convex optimization problem with additional bilinear constraints,
making the total optimization problem non-convex (in general, there are exceptions).
The convex problem is specified using Convex.jl, the bilinear constraints using the BilinearConstraint structs.
After calling solve!() on this problem, the result is stored in the field 'result'.

# Example
```julia-repl
julia> A = Variable()
julia> A0 = 20 # initial guess
julia> bc = BilinearConstraint(A,1.,A,1. X=-A0, Y=-A0)
julia> BilinearProblem(minimize(0.),A) # find A, s.t. A^2=1
```

"""
mutable struct BilinearProblem
    convexproblem::Convex.Problem
    bilinearconstraints::Vector{BilinearConstraint}
    result::Result

    function BilinearProblem(convexproblem::Convex.Problem,
        bilinearconstraints::Vector{BilinearConstraint}=BilinearConstraint[])

        isempty(bilinearconstraints) && error("No bilinear constraints supplied")

        return new(convexproblem,bilinearconstraints)
    end
end

BilinearProblem(convexproblem::Convex.Problem,bilinearconstraints::BilinearConstraint...) =
    BilinearProblem(convexproblem,[bilinearconstraints...])

function _setXY!(b::BilinearConstraint,Xn::Union{Number,AbstractArray},Yn::Union{Number,AbstractArray}; isestimate=true)

    Xn = Xn isa Number ? Xn = fill(Xn, (1, 1)) : Xn
    Xn = Xn isa AbstractVector ? Xn = reshape(Xn,length(Xn),1) : Xn
    Yn = Yn isa Number ? Yn = fill(Yn, (1, 1)) : Yn
    Yn = Yn isa AbstractVector ? Yn = reshape(Yn,length(Yn),1) : Yn

    isestimate ? fix!(b.X,reshape(-Xn,size(b.X))) : fix!(b.X,reshape(Xn,size(b.X)))
    isestimate ? fix!(b.Y,reshape(-Yn,size(b.Y))) : fix!(b.Y,reshape(Yn,size(b.Y)))
end

function _setW1W2!(b::BilinearConstraint, W1::Union{Number,AbstractArray},W2::Union{Number,AbstractArray})
    fix!(b.W1,W1)
    fix!(b.W2,W2)
end

function _regularizationMatrix(b::BilinearConstraint)
    return [ b.W1*(b.C + b.X*b.P*b.Y + b.A*b.P*b.Y + b.X*b.P*b.B)*b.W2  b.W1*(b.A+b.X)*b.P; b.P*(b.B+b.Y)*b.W2 b.P]
end

"""
    solve!(bilinearproblem::BilinearProblem,
        solver::MathProgBase.AbstractMathProgSolver;
        iterations::Int=1,
        trackvariables::Tuple{Vararg{AbstractExpr}}=tuple(),
        update_weights::Bool=false,
        weight_update_tuning_param)

Attempt to solve the bilinear problem (first argument) using the convex optimization solver.
Note that solver must be able to handle semidefinite programming (SDP) problems.
For example, use SCS or Mosek. For larger problems, Mosek seems to have less numerical problems than SCS.

Note that this function modifies the original Convex.Problem.
Calling this function twice on the same problem may give unexpected results.

The solver is iterative, and the number of iterations can be set as an argument. Defaults to 1 iteration.

If a tuple of decision variables is supplied, their values at each iteration are stored
for later analysis in the Result object that the function returns and stores in the bilinearproblem struct.

# Example
```julia-repl
julia> using Convex, SCS
julia> A = Variable()
julia> A0 = 20 # initial guess
julia> bc = BilinearConstraint(A,1.,A,1. X=-A0, Y=-A0)
julia> bp = BilinearProblem(minimize(0.),A) # find A, s.t. A^2=1
julia> solve!(bp,SCSSolver(),iterations=3)
```
For this specific example, A^2 = 1, one can show the solve!() method always (globally) converges to the correct solution given enough iterations.
The number of required iterations for this specific example to converge scales logarithmically with A0.

The option update_weights ∈ {true, false} is a heuristic reweighting method that can improves the convergence speed for matrix-valued constraints.
Think of it as a reweighted regularization ||W1(C-APB)W2||, where W1 and W2 are weighting matrices computed as:
E := C-APB, W1 = inv(E*E' + γ*I), W2 = inv(E'*E + γ*I), γ = weight_update_tuning_param > 0.
"""
function solve!(bilinearproblem::BilinearProblem,
    solver::MathProgBase.AbstractMathProgSolver;
    iterations::Int=1,        # SCR iterations
    trackvariables::Tuple{Vararg{AbstractExpr}}=tuple(),
    update_weights::Bool=false,
    weight_update_tuning_param::AbstractFloat=0.1)

    @assert weight_update_tuning_param > 0

    bp = bilinearproblem
    regularization = sum([nuclearnorm(b.λ*_regularizationMatrix(b)) - b.λ*sum(svdvals(b.P)) for b in bp.bilinearconstraints])

    p = bp.convexproblem
    obj = p.objective
    p.objective = p.head == :minimize ? obj+regularization : obj-regularization

    constraint_violations = Vector{AbstractFloat}[]
    objective_values = AbstractFloat[]
    track = false
    tracked_variables_values = Any[]
    if !isempty(trackvariables)
        track = true
    end

    for i in 1:iterations

        for b in bp.bilinearconstraints

            if update_weights
                A = -evaluate(b.X)
                B = -evaluate(b.Y)
                C = evaluate(b.C)

                # constraint violations
                E = C.- A*b.P*B
                F = inv(E*E' + weight_update_tuning_param * Matrix{Float64}(I,size(C,1),size(C,1)))
                G = inv(E'*E + weight_update_tuning_param * Matrix{Float64}(I,size(C,2),size(C,2)))

                _setW1W2!(b,F,G)
            end
        end

        Convex.solve!(bp.convexproblem, solver)

        push!(objective_values, evaluate(obj))

        cv = AbstractFloat[]
        for b in bp.bilinearconstraints
            A = evaluate(b.A)
            B = evaluate(b.B)
            C = evaluate(b.C)

            # record constraint violations
            E = C.- A*b.P*B
            push!(cv, norm(E))

            _setXY!(b,A,B)
        end
        push!(constraint_violations,cv)

        # record the evolution of the requested variables
        if track
            im = tuple([evaluate(t) for t in trackvariables]...)
            push!(tracked_variables_values,im)
        end


    end

    p.objective = obj # reset the original objective function

    # issue a warning for constraints that have not been satisfied
    for (i,b) in enumerate(bp.bilinearconstraints)
        A = evaluate(b.A)
        B = evaluate(b.B)
        C = evaluate(b.C)

        E = C.- A*b.P*B

        if norm(E) > 1E-5
            @warn "Bilinear constraint $i not satisfied (gap = $(norm(E)))"
        end
    end

    bilinearproblem.result = Result(iterations,
        update_weights,
        objective_values,
        constraint_violations,
        tracked_variables_values)
end
end

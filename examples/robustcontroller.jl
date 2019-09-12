# Example for designing a robust static output feedback controller for a system with a Linear Fractional Representation for the uncertainty

# From: "Sequential Convex Relaxation for Robust Static Output Feedback Structured Control"
# Reinier Doelman, Michel Verhaegen

using Convex
using SequentialConvexRelaxation
using Mosek # v0.9.11
using LinearAlgebra
using Plots
using Printf
using Random
Random.seed!(1)

In(n) = Matrix{Float64}(I,n,n) # Identity matrix

# The LMIs/BMIs are for systems with the following description (including LPV systems):
# [xdot; q; z; y] = [A Bp Bw Bu; Cq Dqp Dqw Dqu; Cz Dzp Dzw Dzu; Cy Dyp Dyw 0][x;p;w;u]
# p = Δ * q, Δ in a polytopic set of uncertainties.
#
# x is a state vector, xdot its time derivative,
# z the performance output,
# y the measurement,
# w the disturbance,
# u the input,
# p&q are the channels for describing the uncertainty with a Linear Fractional Representation.
# The system's H∞ norm (The variable 'γ' below) measures the effect of the disturbance signal on the performance signal.
# This is what we would like to minimize using feedback control of the form u = Ky.
# We try to find the feedback matrix K that minimizes γ for all possible system realizations.
# Optionally we might want K to be of a certain structure, like K[1,2] == 0, if the application demands it.

# The data for the system matrices is taken from:
# "Robust static output feedback H∞ control design for linear systems with polytopic uncertainties,"
# Xiao-Heng Chang, Ju H. Park and Jianping Zhou
# Section 4, example 1.

# System description in Chang et al.
# [xdot; z; y] = [A B E; C1 D F; C2 0 H][x; u; w]
# A = (1-α)A1 + αA2, α ∈ [0,1],
# B = (1-α)B1 + αB2
# etc.
A1 = [-2.98 -0.57 0 -0.034;
    -0.99 -0.21 0.035 -0.0011;
    0 0 0 1;
    0.39 -5.5550 0 -1.89]

A2 = [-2.98 2.43 0 -0.034;
    -0.99 -0.21 0.035 -0.0011;
    0 0 0 1;
    0.39 -5.555 0 -1.89]

B1 = [0.032; 0; 0; 1.6]
B2 = B1

E1 = [0.0; 0.0; 0.0; 1.0]
E2 = E1

C11 = [1.0 0.0 0.0 2.0]
C12 = C11

D1 = 1.0
D2 = D1

F1 = 0.0
F2 = F1

C21 = [0 0 1 0; 0 0 0 1.0]
C22 = C21

H1 = [0.5; -1.0]
H2 = H1

# System description in Doelman & Verhaegen
# [xdot; q; z; y] = [A Bp Bw Bu; Cq Dqp Dqw Dqu; Cz Dzp Dzw Dzu; Cy Dyp Dyw 0][x;p;w;u]
# p = Δ * q

# Signal dimensions
ns = 4 # states
mp = 6 # inputs
mw = 1
mu = 1
rq = 6 # outputs
rz = 1
ry = 2

A = A1
Bu = B1
Bw = E1
Bp = [A2-A1 E2-E1 B2-B1]

# q = [x; w; u]
Cq = [In(ns); zeros(mw+mu,ns)]
Dqu = [zeros(ns+mw,mu); In(mu)]
Dqw = [zeros(ns,mw); In(mw); zeros(mu,mw)]
Dqp = zeros(rq,mp)

Cz = C11
Dzu = D1
Dzw = F1
Dzp = [C12-C11 F2-F1 D2-D1]

Cy = C21
Dyw = H1
Dyp = [C21-C22 H2-H1 zeros(ry,mu)]

Δ = (zeros(mp,mp), In(mp)) # Vertices of the polytopic uncertainty

Σ = [A  Bp  Bw  Bu;
     Cq Dqp Dqw Dqu;
     Cz Dzp Dzw Dzu;
     Cy Dyp Dyw zeros(ry,mu)] # Verify if all the dimensions are correct

# Decision variables
# - Full-block multiplier
Q = -Semidefinite(mp)
S = Variable(mp,rq)
R = Semidefinite(rq)
P = [Q S; S' R]

# - Stability
Y = Semidefinite(ns)

# - Static feedback gain matrix u = Ky
K = Variable(mu,ry)

# - Substitution for products
Ev1 = Variable(mu,ns) # Ev1 == K*Cy*Y
Ev2 = Variable(mu,mp) # Ev2 == K*Dyp*Q
Ev3 = Variable(mu,rq) # Ev3 == K*Dyp*S

# Squared H∞ norm (objective)
γ2 = Variable(1,1)

# Construct the LMIs for robust performance, section 2&4 in Doelman & Verhaegen.
# This is the convex part of the "convex + bilinear equality constraint" problem.
N = [zeros(ns,ns+rq+rz)                 ;
     zeros(rq,ns)       R   zeros(rq,rz);
     zeros(rz,ns+rq)        γ2*In(rz)]
Lbar = [Q zeros(mp,mw); zeros(mw,mp) -In(mw)]
LbarGbar = -[(Bp*Q+Bu*Ev2)   (-Bw-Bu*K*Dyw);
             (Dqp*Q+Dqu*Ev2) (-Dqw-Dqu*K*Dyw);
             (Dzp*Q+Dzu*Ev2) (-Dzw-Dzu*K*Dyw)]'
GtW = -[(A *Y+Bu *Ev1) (Bp *S+Bu *Ev3) zeros(ns,rz);
        (Cq*Y+Dqu*Ev1) (Dqp*S+Dqu*Ev3) zeros(rq,rz);
        (Cz*Y+Dzu*Ev1) (Dzp*S+Dzu*Ev3) zeros(rz,rz)]
T = [-Lbar LbarGbar; LbarGbar' N+GtW+GtW']

constraints = [T in :SDP,
    -Q in :SDP,
    Y in :SDP,
    R in :SDP]

for Δi in Δ
    global constraints
    constraints += - [In(mp); -Δi']' * P * [In(mp); -Δi'] in :SDP
end

# constraints += K[1,2] == 0 # Uncomment for structured control

problem = minimize(γ2, constraints)

# These represent the bilinear equality constraints. The values for λ are tuned.
# The initial guess for K,Y,Q,S are taken randomly, but the seed in line 13 may affect the result.
λ = 0.11 # For structured control set λ = 0.3
XK = randn(size(K))
YY = randn(size(Y))
YQ = randn(size(Q))
YS = randn(size(S))
bc = [BilinearConstraint(K,Cy,Y,Ev1,λ=130*λ,X=XK,Y=YY),
    BilinearConstraint(K,Dyp,Q,Ev2,λ=0.4λ,X=XK,Y=YQ),
    BilinearConstraint(K,Dyp,S,Ev3,λ=40*λ,X=XK,Y=YS)]

# Combine the convex problem part with the bilinear constraint part.
bp = BilinearProblem(problem,bc)

# Call the solver. Number of iterations is tuned by hand.
r = solve!(bp, MosekSolver(MSK_IPAR_LOG=0), iterations=300)

# What is the optimal controller?
println("Controller:")
Kstar = evaluate(K)
Kstar |> display # [-2.09325  -0.869719]
# Kstar = [-2.1609 -0.9597] # Doelman & Verhaegen, γ = 0.64189.
# Kstar = [-2.7375 -0.8618] # Chang et al., γ = 0.6581

println("H∞ norm (closed loop): $(@sprintf("%.5f", sqrt(evaluate(γ2))))") # 0.6436

# Display algorithm behaviour in a plot
p1 = plot(1:r.iterations,hcat(r.constraint_violations...)',yaxis=(:log10),xaxis=("iteration"),lab=["Constraint $i" for i in 1:3])
title!(p1,"Constraint violation")
p2 = plot(1:r.iterations,r.objective_values .|> sqrt, lab="gamma",xaxis=("iteration"))
title!(p2,"H infinity norm: $(@sprintf("%.4f", sqrt(evaluate(γ2))))")
p3 = plot(p1,p2)
gui(p3)

# Verify the result by substituting the computed controller gain as fixed values, and compute the H∞ norm.
Ev1 = Kstar*Cy*Y
Ev2 = Kstar*Dyp*Q
Ev3 = Kstar*Dyp*S
N = [zeros(ns,ns+rq+rz)                 ;
     zeros(rq,ns)       R   zeros(rq,rz);
     zeros(rz,ns+rq)        γ2*In(rz)]
Lbar = [Q zeros(mp,mw); zeros(mw,mp) -In(mw)]
LbarGbar = -[(Bp*Q+Bu*Ev2)   (-Bw-Bu*K*Dyw);
             (Dqp*Q+Dqu*Ev2) (-Dqw-Dqu*K*Dyw);
             (Dzp*Q+Dzu*Ev2) (-Dzw-Dzu*K*Dyw)]'
GtW = -[(A *Y+Bu *Ev1) (Bp *S+Bu *Ev3) zeros(ns,rz);
        (Cq*Y+Dqu*Ev1) (Dqp*S+Dqu*Ev3) zeros(rq,rz);
        (Cz*Y+Dzu*Ev1) (Dzp*S+Dzu*Ev3) zeros(rz,rz)]
T = [-Lbar LbarGbar; LbarGbar' N+GtW+GtW']

constraints = [T in :SDP,
    -Q in :SDP,
    Y in :SDP,
    R in :SDP]

for Δi in Δ
    global constraints
    constraints += - [In(mp); -Δi']' * P * [In(mp); -Δi'] in :SDP
end
problem = minimize(γ2, constraints)
solve!(problem, MosekSolver(MSK_IPAR_LOG=0))
println("H∞ norm (closed loop, verified): $(@sprintf("%.5f", sqrt(evaluate(γ2))))") # 0.64222

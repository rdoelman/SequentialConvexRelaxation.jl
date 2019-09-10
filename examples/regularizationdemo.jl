# This example demonstrates how the convex regularization can have minima that are solutions to bilinear constraints.
using Convex
using SequentialConvexRelaxation
using LinearAlgebra
using Plots
using Interact
using SCS

M(a,b,x,y) = [1. + x*y + a*y + x*b a+x; b+y 1.]
nn(X) = sum(svdvals(X)) # nuclear norm

N = 50
K = 3
X = LinRange(-K*1.1,K*1.1,100)
xpos = LinRange(1E-4,K,50)
@manipulate for aprev=slider(LinRange(-3,3,N),label="a previous",value=0.2),
    bprev=slider(LinRange(-3,3,N),label="b previous",value=0.3)

    plot(-xpos,[-1. / p for p in xpos],color="dodgerblue",lab="")
    plot!(xpos,[1. / p for p in xpos],color="dodgerblue",lab="")
    scatter!([aprev],[bprev],color="red",lab="previous")
    contour!(X,X, (a,b) -> nn(M(a,b,-aprev,-bprev)))

    A = Variable()
    B = Variable()
    bc = BilinearConstraint(A,1.,B,1.,X=-aprev,Y=-bprev)
    bp = BilinearProblem(minimize(0.),bc)
    solve!(bp,SCSSolver(verbose=0))
    scatter!([evaluate(A)],[evaluate(B)],color="yellow",lab="minimum")

    xaxis!((-K,K),"a")
    yaxis!((-K,K),"b")

end

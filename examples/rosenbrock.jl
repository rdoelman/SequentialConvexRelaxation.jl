using Convex
using SequentialConvexRelaxation
using LinearAlgebra
using Plots
using Interact
using SCS

a = 1.
b = 100.

# The Rosenbrock function
f(x,y) = (a-x)^2 + b*(y-x^2)^2

X = LinRange(-2,2,100)
Y = LinRange(-1,3,100)

x = Variable()
y = Variable()
c = Variable()

@manipulate pause=0.3 for x0 = slider(X,label="x0",value=0.2),
    λ=slider(LinRange(0,5,50),label="λ",value=0.2),
    iterations=slider(1:7,label="iterations",value=3)

    # Plot the Rosenbrock with logarithmic contour lines
    contour(X,Y,(x,y) -> f(x,y) |> log10)
    yaxis!((-1,3),"y")
    xaxis!((-2,2),"x")
    title!("Rosenbrock function")

    bc = BilinearConstraint(x,1.,x,y-c,X=-x0,Y=-x0,λ=λ)
    p = minimize(norm([a-x; sqrt(b)*c]))
    bp = BilinearProblem(p,bc)
    r = solve!(bp,SCSSolver(verbose=0),iterations=iterations, trackvariables=(x,y))
    xx = [r.tracked_variables_values[i][1] for i in 1:r.iterations]
    yy = [r.tracked_variables_values[i][2] for i in 1:r.iterations]

    plot!([1],[1],marker=:circle,color=colorant"yellow",label="Global minimum",ms=8)
    plot!(vcat(x0,xx),vcat(1.,yy),marker=:circle,color=colorant"dodgerblue",label="Iterative guess of x*")
end

# This example shows how to train a neural network using convex optimization
# The training problem is formulated as a bilinearly constraint problem, and then optimized with convex optimization.
using Convex
using SequentialConvexRelaxation
using Mosek # use 0.9.11
solver = MosekSolver(MSK_IPAR_LOG=0)
using LinearAlgebra
using Random
Random.seed!(1)
using Plots

ReLU(x) = max.(0.,x)
⊗ = kron

# ==== Neural network with: ====
# - 1 input layer (size Nli)
# - 1 hidden layer (size Nlh)
# - 1 output layer (size Nlo)
# - N input/output data points
Nli = 3
Nlh = 5
Nlo = 3
N = 30 # number of measurements / training samples

W1 = 0.5*randn(Nlh,Nli)
W2 = 0.5*randn(Nlo,Nlh)
b1 = 0.1*randn(Nlh,1)
b2 = 0.1*randn(Nlo,1)
B1 = b1 ⊗ fill(1., (1,N))
B2 = b2 ⊗ fill(1., (1,N))

X0 = randn(Nli,N) # inputs
X1 = ReLU(W1*X0 + B1) # hidden layer
X2 = ReLU(W2*X1 + B2)
Y = X2 # measurements

# Z1 = W1*X0 + B1
# Z2 = W2*X1 + B2
# S1 = X1 - Z1
# S2 = X2 - Z2

# ==== Training the weights with SCR ====

# Decision variables
W1v = Variable(size(W1))
W2v = Variable(size(W2))
b1v = Variable(size(b1))
b2v = Variable(size(b2))
B1v = b1v ⊗ fill(1., (1,N))
B2v = b2v ⊗ fill(1., (1,N))

X1v = Variable(size(X1))

Z1v = W1v*X0 + B1v
Z2v = Variable(size(W2,1),size(X1,2))
S1v = X1v - Z1v
S2v = X2 - Z2v

# - Affine constraints
constraints = [X1v >= 0., S1v >= 0., S2v >= 0., X2 .* S2v == 0.]

# - Bilinear constraints
bc = BConstraint[]
X1v0 = 0.05*randn(size(X1v)) # initial guess
S1v0 = 0.05*randn(size(S1v)) # initial guess
W2v0 = 0.05*randn(size(W2v)) # initial guess
for i in 1:size(X1v,1), j in 1:size(X1v,2)
  xi = X1v[i,j]
  si = S1v[i,j]
  push!(bc, BilinearConstraint(xi,1.,si,0.,X=-X1v0[i,j],Y=-S1v0[i,j],λ = 1.1))
end
push!(bc, BilinearConstraint(W2v,Matrix{Float64}(I,Nlh,Nlh),X1v,Z2v - B2v,X=-W2v0,Y=-X1v0, λ = 10.))

# ==== Construct the bilinear problem and try to solve it ====
problem = minimize(0.,constraints)
bp = BilinearProblem(problem, bc)
solve!(bp, solver, iterations=25, update_weights=true, weight_update_tuning_param = 0.2)

# ==== Plotting the results ====
cl = (-0.1,1)
# - Training set error
X1e = ReLU(evaluate(W1v)*X0 + evaluate(B1v)) # hidden layer
X2e = ReLU(evaluate(W2v)*X1e + evaluate(B2v)) # hidden layer
Ye = X2e # estimated measurements

plot(
  heatmap(Y', clims=cl, xticks=1:Nlo, title="real"),
  heatmap(Ye', clims=cl, xticks=1:Nlo, title="estimated", xlabel="training set"),
  heatmap((Y - Ye)', clims=cl, xticks=1:Nlo, title="difference"),
  layout=(1,3)
  )

# - Test set error
X0t = randn(Nli,N) # inputs
X1t = ReLU(W1*X0t + B1) # hidden layer
X2t = ReLU(W2*X1t + B2) # hidden layer
Yt = X2t # measurements

X1et = ReLU(evaluate(W1v)*X0t + evaluate(B1v)) # hidden layer
X2et = ReLU(evaluate(W2v)*X1et + evaluate(B2v)) # hidden layer
Yet = X2et # estimated measurements
plot(
  heatmap(Yt', clims=cl, xticks=1:Nlo, title="real"),
  heatmap(Yet', clims=cl, xticks=1:Nlo, title="estimated", xlabel="test set"),
  heatmap((Yt - Yet)', clims=cl, xticks=1:Nlo, title="difference"),
  layout=(1,3)
  )

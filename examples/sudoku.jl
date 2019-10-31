using Convex
using SequentialConvexRelaxation
using LinearAlgebra
using SCS
solver = SCSSolver(verbose=1, max_iters=40000)
using Random
Random.seed!(1)


# https://www.websudoku.com/?level=3&set_id=4720027535
# 0 indicates unknown value
S = [
  0 0 0 0 0 0 2 0 6;
  2 0 5 9 1 0 4 0 0;
  0 3 0 0 0 0 0 7 0;
  4 0 0 0 0 6 0 0 0;
  3 0 0 4 9 5 0 0 7;
  0 0 0 2 0 0 0 0 8;
  0 4 0 0 0 0 0 6 0;
  0 0 2 0 7 4 8 0 9;
  9 0 8 0 0 0 0 0 0;
]

# for each square [i,j] we have 9 variables, where x_i,j,k = 1 if the square
# should have k as the number, and zero otherwise.
x = Variable[]
for k = 1:9
  push!(x,Variable(9,9))
end
x

constraint = Constraint[]

# x is larger than 0, smaller than 1
# These constraints are automaticall added when using BinaryConstraint()
# for k in 1:9
#   push!(constraint, x[k] <= 1)
#   push!(constraint, x[k] >= 0)
# end

# each number k appears once in column j
for j = 1:9, k = 1:9
  push!(constraint, sum(x[k][:,j]) == 1)
end

# each k appears once in row i
for i = 1:9, k = 1:9
  push!(constraint, sum(x[k][i,:]) == 1)
end

# each k appears once in a block
for i in (1,4,7), j in (1,4,7), k in 1:9
  push!(constraint,sum(x[k][i:i+2,j:j+2]) == 1)
end

# each cell contains exactly one number
for i in 1:9, j in 1:9
  r = 0
  for k in 1:9
    r = r + x[k][i,j]
  end
  push!(constraint,r == 1)
end

# when the initial assignment has number k in cell i,j
for i in 1:9, j in 1:9, k in 1:9
  if S[i,j] == k
    push!(constraint,x[k][i,j] == 1)
  end
end

bc = BConstraint[]
for i in 1:9, j in 1:9, k = 1:9
  push!(bc,BinaryConstraint(x[k][i,j],X=-1*rand()))
end

p = minimize(0.,constraint)
bp = BilinearProblem(p,bc)
solve!(bp, solver, iterations=2)

X = [evaluate(x[k]) for k in 1:9]
X = [k*round.(Int,X[k]) for k in 1:9]
X = reduce(+,X)
display(X)

# solution = X = [
#   7  1  9  3  4  8  2  5  6
#   2  6  5  9  1  7  4  8  3
#   8  3  4  5  6  2  9  7  1
#   4  2  1  7  8  6  3  9  5
#   3  8  6  4  9  5  1  2  7
#   5  9  7  2  3  1  6  4  8
#   1  4  3  8  5  9  7  6  2
#   6  5  2  1  7  4  8  3  9
#   9  7  8  6  2  3  5  1  4
# ]
# X - solution

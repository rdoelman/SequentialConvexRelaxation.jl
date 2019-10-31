using SequentialConvexRelaxation
using Test
using Convex
using LinearAlgebra
using SCS
using Random
Random.seed!(123)

@testset "SequentialConvexRelaxation.jl" begin
    A = Variable()
    B = Variable()
    C = Variable()
    D = 1.
    P = 1.
    X = 0.2
    Y = 0.3
    W1 = 2.
    W2 = 3.
    λ = 1.

    @testset "Bilinear constraint construction" begin
        @test BilinearConstraint(A,P,B,C) isa BilinearConstraint
        @test BilinearConstraint(A,P,B,D) isa BilinearConstraint
        @test BilinearConstraint(A,[P],B,[D]) isa BilinearConstraint
        @test BilinearConstraint(A,P,B,C,X=X) isa BilinearConstraint
        @test BilinearConstraint(A,P,B,C,X=X, Y=Y) isa BilinearConstraint
        @test BilinearConstraint(A,P,B,C,W2=W2, W1=W1) isa BilinearConstraint
        @test BilinearConstraint(A,P,B,C,λ=λ) isa BilinearConstraint
        @test_throws AssertionError BilinearConstraint(A,P,B,C,W2=0.)
        @test_throws AssertionError BilinearConstraint(A,P,B,C,W1=0.)
        @test_throws AssertionError BilinearConstraint(A,[2., 1.],B,C,W1=0.)
        @test BilinearConstraint(Variable(2,1),P,B,Variable(2,1)) isa BilinearConstraint
        @test BilinearConstraint(Variable(2,1),[P],B,Variable(2,1)) isa BilinearConstraint
        @test_throws DimensionMismatch BilinearConstraint(Variable(2,1),[P],B,Variable(2,1),W1=ones(2,1))
        @test_throws AssertionError BilinearConstraint(Variable(2,1),[P],B,Variable(2,1),W1=[2. 1.; 2. 1.])
        @test BilinearConstraint(ComplexVariable(2,1),P,B,ComplexVariable(2,1)) isa BilinearConstraint
        @test BinaryConstraint(A,X=X) isa BinaryConstraint
    end

    bc1 = BilinearConstraint(A,P,B,D,X=X,Y=Y)
    bc2 = BilinearConstraint(A,P,B,D,X=60*X,Y=60*X)
    problem = minimize(0.)

    @testset "Bilinear problem construction" begin
        @test BilinearProblem(problem,[bc1]) isa BilinearProblem
        @test BilinearProblem(problem,[bc1,bc2]) isa BilinearProblem
        @test BilinearProblem(problem,bc1) isa BilinearProblem
        bp = BilinearProblem(problem,bc1)
        @test_throws UndefRefError bp.result
    end

    @testset "Test solver" begin
        @testset "A*B = 1." begin
            bc1 = BilinearConstraint(A,P,B,D,X=X,Y=Y)
            problem = minimize(0.)
            bp = BilinearProblem(problem,bc1)
            SequentialConvexRelaxation.solve!(bp,SCSSolver(verbose=0),iterations=3)
            @test bp.result.iterations == 3
            @test bp.result.update_weights == false
            @test bp.result.constraint_violations[1][1] <= 1E-5
            @test isempty(bp.result.tracked_variables_values)
            @test ≈(evaluate(A)*evaluate(B),1.,rtol=1E-5)
        end

        @testset "A*B = 1., multiple iterations" begin
            problem = minimize(0.)
            bc2 = BilinearConstraint(A,P,B,D,X=60*X,Y=60*X)
            bp = BilinearProblem(problem,bc2)
            SequentialConvexRelaxation.solve!(bp,SCSSolver(verbose=0),iterations=5)
            @test bp.result.iterations == 5
            @test bp.result.update_weights == false
            @test bp.result.constraint_violations[end][1] <= 1E-3
            @test isempty(bp.result.tracked_variables_values)
            @test ≈(evaluate(A)*evaluate(B),1.,rtol=1E-4)
        end

        @testset "A*B = 1, track variables" begin
            problem = minimize(0.)
            bc2 = BilinearConstraint(A,P,B,D,X=20*X,Y=20*X)
            bp = BilinearProblem(problem,bc2)
            r = SequentialConvexRelaxation.solve!(bp,SCSSolver(verbose=0),trackvariables=(A, B),iterations=4)
            @test !isempty(bp.result.tracked_variables_values)
        end

        @testset "Complex-valued bilinear constraints" begin
            problem = minimize(0.)
            A = ComplexVariable()
            B = ComplexVariable()
            bc2 = BilinearConstraint(A,P,B,D,X=0.1+0.1im,Y=0.3-0.2im)
            bp = BilinearProblem(problem,bc2)
            r = SequentialConvexRelaxation.solve!(bp,SCSSolver(verbose=0),trackvariables=(A, B),iterations=7)
            @test !isempty(bp.result.tracked_variables_values)
            @test norm(evaluate(A)*evaluate(B) - 1.) <= 1E-3
        end

        @testset "Matrix-valued bilinear constraints" begin
            problem = minimize(0.)
            A = Variable(2,1)
            B = Variable(1,2)
            D = rand(2)*rand(2)'
            bc2 = BilinearConstraint(A,1.,B,D)
            bp = BilinearProblem(problem,bc2)
            r = SequentialConvexRelaxation.solve!(bp,SCSSolver(verbose=0),trackvariables=(A, B),iterations=3)
            @test !isempty(bp.result.tracked_variables_values)
            @test norm(evaluate(A)*evaluate(B) - D) <= 1E-4
        end
    end

end

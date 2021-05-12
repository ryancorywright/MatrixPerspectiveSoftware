using JuMP, Mosek, MosekTools, LinearAlgebra, Suppressor, StatsBase, CSV, DataFrames, Compat, Random
include("convexpenalties.jl")

results_template = DataFrame(n=Int[],k=Int[], theSeed=Int[], gap_persp=Real[], LB_persp=Real[], UB_persp=Real[], UB_BM=Real[], t_persp=Real[], t_BM=Real[],
mse_persp=Real[], mse_BM=Real[], acc_persp=Real[], acc_BM=Real[])

results_fact=similar(results_template, 0)

for ARG in ["5"]



array_num = parse(Int, ARG)
seeds=1:20

n=50
k_true=10
ks=2:2:40
gamma=1e6

theSeed=seeds[(array_num-1)÷20+1]
k=ks[(array_num-1)%20+1]
for theSeedI in (theSeed*10):(theSeed*10+9)
    Random.seed!(theSeedI)
    U=rand(n,k_true)
    E=0.05*sqrt(k)*randn(n,n)
    A=U*U'.+E
    @show t_persp=@elapsed X_persp, LB_persp, UB_persp, U_persp=solveNNMFRelax(A, k)
    @show gap_persp=(UB_persp-LB_persp)/UB_persp
    mse_persp=norm(A.-X_persp)/norm(A)
    acc_persp=norm(U*U'.-X_persp)/norm(U*U')

    # Solve using BM with random initialization

    t_BM=@elapsed X_BM, UB_BM=solveBM(A, k, rand(n,k))
    mse_BM=norm(A.-X_BM)/norm(A)
    acc_BM=norm(U*U'.-X_BM)/norm(U*U')

    # # Solve using BM with rounding initialization
    #
    t_BM=@elapsed X_BM, UB_BM=solveBM(A, k, U_persp)
    mse_BM=norm(A.-X_BM)/norm(A)
    acc_BM=norm(U*U'.-X_BM)/norm(U*U')
    # Write results

    push!(results_fact, [n, k, theSeedI, gap_persp, LB_persp, UB_persp, UB_BM, t_persp, t_BM,
    mse_persp, mse_BM, acc_persp, acc_BM])
end

CSV.write("createFig1_times_acc.csv", results_fact, append=true)

end

using JuMP, Mosek, MosekTools, LinearAlgebra, Suppressor, StatsBase, CSV, DataFrames, Compat, Random
include("convexpenalties.jl")

results_template = DataFrame(n=Int[],p=Int[], m=Int[], k=Int[], theSeed=Int[], gamma=Real[], lambda_rnk=Real[], lambda_nn=Real[],
gap_persp=Real[], gap_dcl=Real[], t_persp=Real[], t_dcl=Real[], t_nn=Real[], acc_persp=Real[], acc_dcl=Real[], acc_nn=Real[],
rnk_persp=Int[], rnk_dcl=Int[], rnk_nn=Int[])

results_exact=similar(results_template, 0)

for ARG in ARGS


array_num = parse(Int, ARG)
seeds=1:20

ns=10:10:200
k=10
gamma=1e6

lambda_rnk=1.0
lambda_nn=1.0

# n_obs_oos=1000
theSeed=seeds[(array_num-1)÷20+1]
n=ns[(array_num-1)%20+1]
m=n
p=n
Random.seed!(theSeed)
X=rand(m,p)

U_true=rand(p,k)
V_true=rand(n,k)
beta_true=U_true*V_true'
E=0.05*rand(m,n)

Y=X*beta_true.+E


for t in 1:1

        t_persp=@elapsed beta_est,ofv_dual, ofv_primal,=solvepersprelax(X, Y, m, n, p, 0.279883043, gamma)
        @show t_persp
        @show gap_persp=100*abs.(ofv_primal-ofv_dual)/ofv_primal
        @show rel_error_persp=norm(beta_est.-beta_true)/norm(beta_true)
        @show ofv_primal
        @show 1.0/(2.0*gamma)*norm(beta_est)^2

        u,v,=svd(beta_est)
        @show rnk_persp=sum(v.>1e-4)

        t_dcl=@elapsed beta_est,ofv_dual, ofv_primal,= solvepersprelax_DCL(X, Y, m, n, p, 0.01215675, gamma)
        @show gap_dcl=100*abs.(ofv_primal-ofv_dual)/ofv_primal

        @show rel_error_dcl=norm(beta_est.-beta_true)/norm(beta_true)

        u,v,=svd(beta_est)
        @show rnk_dcl=sum(v.>1e-4)

        t_nn=@elapsed beta_est,=solveNNrelax(X, Y, m, n, p, 0.000573504, gamma)

        @show rel_error_nn=norm(beta_est.-beta_true)/norm(beta_true)
        u,v,=svd(beta_est)
        @show rnk_nn=sum(v.>1e-4)

        push!(results_exact, [n, p, m, k, theSeed, gamma, lambda_rnk, lambda_nn,
        gap_persp, gap_dcl, t_persp, t_dcl, t_nn, rel_error_persp, rel_error_dcl, rel_error_nn,
        rnk_persp, rnk_dcl, rnk_nn])

end
CSV.write("createFig2_times.csv", results_exact, append=true)

end

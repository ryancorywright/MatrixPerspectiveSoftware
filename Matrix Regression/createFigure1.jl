using JuMP, Mosek, MosekTools, LinearAlgebra, Suppressor, StatsBase, CSV, DataFrames, Compat, Random
include("convexpenalties.jl")

results_template = DataFrame(n=Int[],p=Int[], m=Int[], k=Int[], theSeed=Int[], gamma=Real[], lambda_rnk=Real[], lambda_nn=Real[],
gap_persp=Real[], gap_dcl=Real[], t_persp=Real[], t_dcl=Real[], t_nn=Real[], acc_persp=Real[], acc_dcl=Real[], acc_nn=Real[],
mse_persp=Real[], mse_dcl=Real[], mse_nn=Real[], rnk_persp=Int[], rnk_dcl=Int[], rnk_nn=Int[])

results_exact=similar(results_template, 0)

for ARG in ARGS


array_num = parse(Int, ARG)
seeds=1:20

n=50
p=50
ms=5:5:100
k=10
gamma=1e6

lambdas_rnk=10.0.^(range(-4,stop=0,length=20))
lambdas_nn=10.0.^(range(-6,stop=-2,length=20))

n_obs_oos=1000
theSeed=seeds[(array_num-1)÷20+1]
m=ms[(array_num-1)%20+1]
Random.seed!(theSeed)
X=rand(m,p)
X_oos=rand(n_obs_oos, p)

U_true=rand(p,k)
V_true=rand(n,k)
beta_true=U_true*V_true'
E=0.05*rand(m,n)
E_oos=0.05*rand(n_obs_oos,n)
Y=X*beta_true.+E
Y_oos=X_oos*beta_true.+E_oos

for t in 1:20
        lambda_nn=lambdas_nn[t]
        lambda_rank=lambdas_rnk[t]

        t_persp=@elapsed beta_est,ofv_dual, ofv_primal,=solvepersprelax(X, Y, m, n, p, lambda_rank, gamma)
        @show t_persp
        @show gap_persp=100*abs.(ofv_primal-ofv_dual)/ofv_primal
        @show rel_error_persp=norm(beta_est.-beta_true)/norm(beta_true)
        @show ofv_primal
        @show 1.0/(2.0*gamma)*norm(beta_est)^2
        @show mse_persp=norm(X_oos*beta_est.-Y_oos)^2/norm(X_oos*beta_true.-Y_oos)^2

        u,v,=svd(beta_est)
        @show rnk_persp=sum(v.>1e-4)

        t_dcl=@elapsed beta_est,ofv_dual, ofv_primal,= solvepersprelax_DCL(X, Y, m, n, p, lambda_rank, gamma)
        @show gap_dcl=100*abs.(ofv_primal-ofv_dual)/ofv_primal

        @show rel_error_dcl=norm(beta_est.-beta_true)/norm(beta_true)

        u,v,=svd(beta_est)
        @show rnk_dcl=sum(v.>1e-4)
        @show mse_dcl=norm(X_oos*beta_est.-Y_oos)^2/norm(X_oos*beta_true.-Y_oos)^2

        t_nn=@elapsed beta_est,=solveNNrelax(X, Y, m, n, p, lambda_nn, gamma)

        @show rel_error_nn=norm(beta_est.-beta_true)/norm(beta_true)
        u,v,=svd(beta_est)
        @show rnk_nn=sum(v.>1e-4)
        @show mse_nn=norm(X_oos*beta_est.-Y_oos)^2/norm(X_oos*beta_true.-Y_oos)^2

        push!(results_exact, [n, p, m, k, theSeed, gamma, lambda_rank, lambda_nn,
        gap_persp, gap_dcl, t_persp, t_dcl, t_nn, rel_error_persp, rel_error_dcl, rel_error_nn,
        mse_persp, mse_dcl, mse_nn, rnk_persp, rnk_dcl, rnk_nn])

end
CSV.write("createFig1_analyselambdas.csv", results_exact, append=true)

end

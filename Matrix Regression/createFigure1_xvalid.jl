# This script uses the convex penalties with cross-validated lambdas (as obtained in the excel sheet) on a test set
using JuMP, Mosek, MosekTools, LinearAlgebra, Suppressor, StatsBase, CSV, DataFrames, Compat, Random
include("convexpenalties.jl")

results_template = DataFrame(n=Int[],p=Int[], m=Int[], k=Int[], theSeed=Int[], gamma=Real[], lambda_persp=Real[], lambda_dcl=Real[], lambda_nn=Real[],
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

lambdas_persp=[0.0001,	0.0001,	0.00784759970351461,	0.00784759970351461,	0.00784759970351461,	0.615848211066026,	0.615848211066026,	0.233572146909012,	0.0885866790410082,	0.00784759970351461,	0.0335981828628378,	0.143844988828766,	0.233572146909012,	0.379269019073224,	0.379269019073224,	0.379269019073224,	0.615848211066026,	0.615848211066026,	0.615848211066026,	0.615848211066026]
lambdas_dcl=[0.0127427498570313,	0.00112883789168468,	0.00784759970351461,	0.0206913808111479,	0.0206913808111479,	0.00483293023857175,	0.00183298071083243,	0.0206913808111479,	0.00297635144163131,	0.0545559478116852,	0.0206913808111479,	0.0206913808111479,	0.00297635144163131,	0.0127427498570313,	0.00483293023857175,	0.00483293023857175,	0.00784759970351461,	0.00784759970351461,	0.00784759970351461,	0.00483293023857175]
lambdas_nn=[0.000001,	0.000001,	0.000001,	0.000001,	0.000001,	0.000885866790410082,	0.000885866790410082,	0.000545559478116852,	0.000335981828628378,	0.000078475997035146,	0.000335981828628378,	0.000545559478116852,	0.000545559478116852,	0.000885866790410082,	0.000885866790410082,	0.000885866790410082,	0.000885866790410082,	0.000885866790410082,	0.00143844988828766,	0.00143844988828766]

n_obs_oos=1000
lambda_persp=lambdas_persp[(array_num-1)%20+1]
lambda_dcl=lambdas_dcl[(array_num-1)%20+1]
lambda_nn=lambdas_nn[(array_num-1)%20+1]


m=ms[(array_num-1)%20+1]

seed_mult=seeds[(array_num-1)÷20+1]


for theSeed in (100+10*(seed_mult-1)+1):(100+10*seed_mult)
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

        t_persp=@elapsed beta_est,ofv_dual, ofv_primal,=solvepersprelax(X, Y, m, n, p, lambda_persp, gamma)
        @show t_persp
        @show gap_persp=100*abs.(ofv_primal-ofv_dual)/ofv_primal
        @show rel_error_persp=norm(beta_est.-beta_true)/norm(beta_true)
        @show ofv_primal
        @show 1.0/(2.0*gamma)*norm(beta_est)^2
        @show mse_persp=norm(X_oos*beta_est.-Y_oos)^2/norm(X_oos*beta_true.-Y_oos)^2

        u,v,=svd(beta_est)
        @show rnk_persp=sum(v.>1e-4)

        t_dcl=@elapsed beta_est,ofv_dual, ofv_primal,= solvepersprelax_DCL(X, Y, m, n, p, lambda_dcl, gamma)
        @show gap_dcl=100*abs.(ofv_primal-ofv_dual)/ofv_primal
        @show t_dcl
        @show rel_error_dcl=norm(beta_est.-beta_true)/norm(beta_true)

        u,v,=svd(beta_est)
        @show rnk_dcl=sum(v.>1e-4)
        @show mse_dcl=norm(X_oos*beta_est.-Y_oos)^2/norm(X_oos*beta_true.-Y_oos)^2

        t_nn=@elapsed beta_est,=solveNNrelax(X, Y, m, n, p, lambda_nn, gamma)
        @show t_nn
        @show rel_error_nn=norm(beta_est.-beta_true)/norm(beta_true)
        u,v,=svd(beta_est)
        @show rnk_nn=sum(v.>1e-4)
        @show mse_nn=norm(X_oos*beta_est.-Y_oos)^2/norm(X_oos*beta_true.-Y_oos)^2

        push!(results_exact, [n, p, m, k, theSeed, gamma, lambda_persp, lambda_dcl, lambda_nn,
        gap_persp, gap_dcl, t_persp, t_dcl, t_nn, rel_error_persp, rel_error_dcl, rel_error_nn,
        mse_persp, mse_dcl, mse_nn, rnk_persp, rnk_dcl, rnk_nn])

end
CSV.write("createFig1_withfixedlambdas.csv", results_exact, append=true)

end

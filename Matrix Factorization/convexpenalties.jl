using JuMP, Mosek, MosekTools, LinearAlgebra, Suppressor, StatsBase, Ipopt, Gurobi

function solveNNMFRelax(A, theK)
    mod=Model(Mosek.Optimizer)
    n=size(A,1)

    @variable(mod, X[1:n, 1:n], PSD)
    @variable(mod, Y[1:n, 1:n], Symmetric)
    @variable(mod, theta[1:n, 1:n], Symmetric)
    @constraint(mod, X.>=0.0)

    @constraint(mod, Symmetric(Matrix(1.0I, n, n).-Y) in PSDCone());

    @constraint(mod,Symmetric([theta X; X' Y]) in PSDCone());

    @constraint(mod, sum(Y[i,i] for i=1:n)<=theK)

    @objective(mod, Min, sum(theta[i,i] for i=1:n)-2.0*Compat.dot(A, X)+Compat.dot(A,A))

    optimize!(mod)


    @show ofv_dual=objective_value(mod)
    Y_0=value.(Y)
    # solve for U
    mod_U=Model(Mosek.Optimizer)
    @variable(mod_U, U[1:n,1:theK]>=0.0)
    @variable(mod_U, t)
    @constraint(mod_U, [t; 1.0; vec(U-Y_0*U)] in RotatedSecondOrderCone());
    @constraint(mod_U, Y_0*U.>=0.0)
    @suppress optimize!(mod_U)
    U_0=value.(U)

    # Solve for a SDD^+ D, where A~=U_0*D*U_0'. Note that 2x2 non-neg psd matrices are completely positive, so this is a valid inner approx.
    mod2=Model(Mosek.Optimizer)
    @variable(mod2, D[1:theK, 1:theK], Symmetric)
    @variable(mod2, M[1:theK, 1:theK, 1:2, 1:2]>=0.0);
    @constraint(mod2, defineM1[i=1:theK, j=1:theK], M[i,j,1,2]==M[i,j,2,1]);
    @constraint(mod2, defineM2[i=1:theK, j=1:theK], [M[i,j,1,1]; M[i,j,2,2]; M[i,j, 1,2]] in RotatedSecondOrderCone());
    @constraint(mod2, defineD[i=1:theK, j=(i+1):theK], D[i,j]==M[i,j,1,2])
    @constraint(mod2, defineD2[i=1:theK], D[i,i]==sum(M[i,l,2,2] for l=1:(i-1))+sum(M[i,l,1,1] for l=(i+1):theK))

    @variable(mod2, t)
    @constraint(mod2, [t; 1.0; vec(A-U_0*D*U_0')] in RotatedSecondOrderCone());
    @objective(mod2, Min, t)
    optimize!(mod2)
    X_rounded=U_0*value.(D)*U_0'

    return X_rounded, ofv_dual, objective_value(mod2), U_0*sqrt(value.(D))
end

function solveBM(A, k, V_t)
    n=size(A,1)
    ofv_prev=1e10
    ofv_best=1e10
    V_t_best=V_t
    maxEpochs=100
    rho=0.0001
    # V_t=rand(n,k)

    for t=1:maxEpochs

    mod1=Model(Mosek.Optimizer)

    @variable(mod1, U[1:n, 1:k])
    @constraint(mod1, U.>=0.0)

    @objective(mod1, Min, (sum((A[i,j]-U[i,:]'*V_t[j,:])^2 for i=1:n for j=1:n))+rho*sum((U[i,j]-V_t[i,j])^2 for i=1:n for j=1:k))
    @suppress optimize!(mod1)
    # @show objective_value(mod1)

    U_t=value.(U)

    mod2=Model(Mosek.Optimizer)
    @variable(mod2, V[1:n, 1:k])
    @constraint(mod2, V.>=0.0)
    @objective(mod2, Min, (sum((A[i,j]-U_t[i,:]'*V[j,:])^2 for i=1:n for j=1:n))+rho*sum((V[i,j]-U_t[i,j])^2 for i=1:n for j=1:k))
    @suppress optimize!(mod2)
    # @show objective_value(mod2)
    V_t=value.(V)

    @show ofv_current=norm(A.-V_t*V_t')^2 # Sometimes find a better solution near the end of the reg path but not actually at the end
    if ofv_current<ofv_best
        ofv_best=ofv_current
        V_t_best=V_t
    end
    rho=min(rho*2.0, 1e5)

    if abs(ofv_current-ofv_prev)<1e-4
        break
    end
    ofv_prev=ofv_current

    end

    ofv_final=norm(A.-V_t_best*V_t_best')^2

    return V_t_best*V_t_best', ofv_final
end

using JuMP, Mosek, MosekTools, LinearAlgebra, Suppressor, StatsBase#, SCS#, Ipopt


# Solve the perspective relaxation (naive but cheaper formulation)
function solvepersprelax(X, y, m, n, p, lambda, gamma)
    mod=Model(Mosek.Optimizer)

    @variable(mod, beta[1:p, 1:n])
    @variable(mod, W[1:n, 1:n], Symmetric)
    @variable(mod, theta[1:p, 1:p], Symmetric)
    @variable(mod, t) # Epigraph variable for rotated SOC constraint

    @constraint(mod, Symmetric(Matrix(1.0I, n, n)-W) in PSDCone());

    @constraint(mod, Symmetric([theta beta; beta' W]) in PSDCone());

    @constraint(mod, [t; 1.0; vec(y-X*beta)] in RotatedSecondOrderCone());

    @objective(mod, Min, (0.5/m)*t+0.5*1.0/(2.0*gamma)*sum(theta[i,i] for i=1:p)+lambda*sum(W[i,i] for i=1:n))

    @suppress optimize!(mod)

    ofv_dual=objective_value(mod)
    u,v,=svd(value.(W))
    # Use a cutoff of 10^-4 for singular values, to avoid e.g. Julia counting singular values of 10^-12 as part of the rank
    ofv_primal=(0.5/m)*LinearAlgebra.dot(y.-X*value.(beta), y.-X*value.(beta))+1.0/(2.0*gamma)*norm(value.(beta))^2+lambda*sum(abs.(v).>1e-4)
    # Todo: round greedily and resolve second stage problem??

    return value.(beta), ofv_dual, ofv_primal, value.(W)
end

# Solve the perspective relaxation (expensive DCL version)
function solvepersprelax_DCL(X, y, m, n, p, lambda, gamma)
    mod=Model(Mosek.Optimizer)

    @variable(mod, beta[1:p, 1:n])
    @variable(mod, B[1:p, 1:p], Symmetric)
    @variable(mod, W[1:n, 1:n], Symmetric)

    @constraint(mod, Symmetric(Matrix(1.0I, n, n)-W) in PSDCone());

    @constraint(mod, Symmetric([B beta; beta' W]) in PSDCone());

    @objective(mod, Min, (0.5/m)*sum(y[i,j]^2 for i=1:m for j=1:n)-(1.0/m)*LinearAlgebra.dot(y, X*beta)+0.5*LinearAlgebra.dot(B, 1.0/gamma*Matrix(1.0I, p, p)+(1.0/m)X'*X)+lambda*sum(W[i,i] for i=1:n))

    @suppress optimize!(mod)

    @show ofv_dual=objective_value(mod)

    u,v,=svd(value.(W))
    # Use a cutoff of 10^-4 for singular values, to avoid e.g. Julia counting singular values of 10^-12 as part of the rank
    ofv_primal=(0.5/m)*LinearAlgebra.dot(y.-X*value.(beta), y.-X*value.(beta))+1.0/(2.0*gamma)*norm(value.(beta))^2+lambda*sum(abs.(v).>1e-4)




    return value.(beta), ofv_dual, ofv_primal, value.(W)
end

# Solve the nuclear norm "relaxation" (not actually a relaxation without big-M constraints and projection matrices)
function solveNNrelax(X, y, m, n, p, lambda, gamma)
    mod=Model(Mosek.Optimizer)

    @variable(mod, beta[1:p, 1:n])
    @variable(mod, t) # Epigraph variable for rotated SOC constraint
    @variable(mod, U[1:p, 1:p])
    @variable(mod, V[1:n, 1:n])

    @constraint(mod, [t; 1.0; vec(y-X*beta)] in RotatedSecondOrderCone());

    @constraint(mod, Symmetric([U beta; beta' V]) in PSDCone());

    @objective(mod, Min, (0.5/m)*t+1.0/(2.0*gamma)*sum(beta[i,j]^2 for i=1:p for j=1:n)+0.5*lambda*sum(U[i,i] for i=1:p)+0.5*lambda*sum(V[i,i] for i=1:n))

    @suppress optimize!(mod)

    # Todo: round greedily and resolve second stage problem??


    return value.(beta), objective_value(mod)
end

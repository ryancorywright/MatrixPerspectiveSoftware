using JuMP, Mosek, MosekTools, LinearAlgebra, Suppressor, MATLAB # Requires a matlab license and the DDS package to be in the right subfolder to run this code.

function solveBasicRelax(A, theK, eps)
    model=Model(Mosek.Optimizer)
    n=size(A,1)
    m=size(A,2)

    @variable(model, z[1:m]>=0.0)
    @constraint(model, z.<=1.0)
    @constraint(model, sum(z)<=theK)
    @variable(model, t)
    E = A * LinearAlgebra.diagm(z) * A'+eps*Matrix(I,n,n)
    @constraint(model, [t, 1, (E[i, j] for i in 1:n for j in 1:i)...] in MOI.LogDetConeTriangle(n))

    @objective(model, Max, t)

    @suppress optimize!(model)


    @show ofv_dual=objective_value(model)

    z_rounded=zeros(m)
    ind_max=sortperm(value.(z),rev=true)[1:theK]
    z_rounded[ind_max].=1.0
    @show ofv_primal=logdet(sum(z_rounded[i]*A[:,i]*A[:,i]' for i in 1:m)+eps*Matrix(I,n,n))
    @show ofv_dual=logdet(sum(value.(z[i])*A[:,i]*A[:,i]' for i in 1:m)+eps*Matrix(I,n,n))

    return value.(z), ofv_dual, z_rounded, ofv_primal
end

function solveBasicGD(A, k, eps)
    n=size(A,1)
    m=size(A,2)
    z0=zeros(m)
    ofv_primal=0.0
    for t=1:k
        z0, ofv_primal=getkplusone(A, k, m, n, z0, eps)
    end

    return z0, ofv_primal
end



function getkplusone(A, k, m, n, z0, eps) # Remark: introducing a +epsilon I term here so that the logdet term isn't negative
    candidateIndices=findall(z0.==0)
    iBest=-1
    ofvBest=-1e6
    Sigma_Gram_current=sum(z0[i]*A[:,i]*A[:,i]' for i in 1:m)
    for i in candidateIndices
        ofv_test=logdet(Sigma_Gram_current+A[:,i]*A[:,i]'+eps*Matrix(I,n,n)) # Use det rather than logdet here since logdet is undefined in the rank-deficient case... Note that greedy doesn't do anything when k<n anyway.
        if ofv_test>ofvBest
            ofvBest=ofv_test
            iBest=i
        end
    end
    z0[iBest]=1.0
    ofv_best=logdet(sum(z0[i]*A[:,i]*A[:,i]' for i in 1:m)+eps*Matrix(I,n,n))
    return z0, ofv_best
end


function solveQRERelax(Adata, theK, eps)

    n=size(Adata,1)
    m=size(Adata,2)
    # Call Matlab here, run via DDS solver, return results, round greedily and return an upper bound
    # eps_mat=mxarray(eps)
    # n_mat=mxarray(n)
    # m_mat=mxarray(m)
    # k_mat=mxarray(theK)
    # A_mat=mxarray(Adata)
    # Plan: build it in Matlab with random data, debug there, then move it here once it works!
    mat"""
    clear; A_mat=[[0.541299839388014,0.449426591726197,0.564473921782696,0.408709992443545,0.855835900703399,0.876145347529076,0.365236483082946,0.104519529950634,0.232254753050780,0.295421839485955,0.927064140870894,0.416765701751856,0.709240953533567,0.899426437101584,0.0990172329213107,0.241554770085929,0.955437531793689,0.228764772549632,0.517723514394718,0.490445656962020;0.868968871664997,0.659616768761805,0.540981282819544,0.947512642788601,0.724406283688824,0.420428701240150,0.598590500684565,0.00997765604969036,0.739832227063707,0.622028790367341,0.0877717606376327,0.280292866619113,0.641341360210975,0.452946115671495,0.570988144269953,0.912720162652781,0.142650346985794,0.423524871222756,0.0618247144123362,0.599441720175534;0.557047467058915,0.753208345950567,0.0689214476617044,0.919280969984633,0.199110301818659,0.487663607210372,0.668487146264776,0.0591533040753343,0.888992345152850,0.0475341941914539,0.332398561915437,0.598100317141488,0.174060860147872,0.0580261764965854,0.325875094193265,0.825734269699093,0.512563384780399,0.273596220894793,0.231367220579307,0.0902161331201864;0.0213981169649879,0.804739013812769,0.988430867357282,0.121243677375576,0.157285874413643,0.460325494339181,0.894564090918275,0.322652370507717,0.859807625912606,0.994609895249091,0.526180895421029,0.0364748567011317,0.0621519426336853,0.106269376654518,0.450492336647337,0.444545883918906,0.971925137586938,0.444565843866667,0.118485991561029,0.978224034127668;0.482680551891677,0.0291555821799541,0.251095440428348,0.591944834715854,0.370475519779316,0.515677249607505,0.0873355239070054,0.779476506848221,0.597060788610351,0.206764149130446,0.246642124058313,0.0636865030513627,0.406659420297776,0.998425291506397,0.577843795117458,0.982062563214574,0.648320621265659,0.627515046901424,0.0987800358735032,0.653018772578069;0.807985579141521,0.779817267477441,0.315472991527951,0.359652173164073,0.862272004954744,0.271994390193763,0.539010349313424,0.335478586307354,0.654752143480193,0.607363956107084,0.542899456586946,0.322882505170560,0.463062925909165,0.866327403667170,0.0748443075998355,0.578267561462169,0.614671003416509,0.534641253536576,0.890275455546976,0.461132126549800;0.736009366155118,0.567350498584087,0.300656948802795,0.719306932475474,0.684762628110137,0.231579838051303,0.428449936007297,0.619568612502970,0.915013993756726,0.347630325692841,0.780867850604077,0.0983785303848749,0.202674299437306,0.615205796593277,0.0573426693754819,0.234423496393930,0.469650371915959,0.385442159974304,0.0333890773864002,0.863758110483555;0.572304395272699,0.0761166268515728,0.0420390381653749,0.523551091500771,0.634197879289704,0.899534076872039,0.617153494459035,0.992885717464124,0.433183080639157,0.717737067921327,0.521882381409282,0.170044626268754,0.869549890049143,0.0269444844838657,0.300956435005654,0.810590442462569,0.577780697646748,0.873450099080692,0.839034991901461,0.262829072110785;0.00898485336788568,0.251621824309164,0.527901300708439,0.260843351982082,0.141323291186943,0.908697326644413,0.558875876956837,0.648005694952403,0.289760755842240,0.0279932951713195,0.931949933886006,0.371163798766144,0.597940369107138,0.322522635620462,0.521721248782275,0.451274164510932,0.911313227169849,0.300348734897046,0.507266724816296,0.823962942846020;0.718280463661178,0.133454528618960,0.256022333768320,0.493079306041446,0.0793022450665238,0.603642852234244,0.225850688199737,0.539775820318468,0.631882300545926,0.0668417611778476,0.147111980322787,0.0397616226212385,0.0230135370914409,0.463776233943276,0.561880235062444,0.249969169410276,0.376220283460707,0.400029715619638,0.113716441044558,0.328975305064346]];

    n_mat=10;
    m_mat=20;
    k_mat=7;
    eps_mat=1e-6;
    disp(n_mat)


    cvx_begin

        variable Y(n_mat,n_mat) semidefinite
        variable z(m_mat)

        eye(n_mat)-Y==semidefinite(n_mat);
        trace(Y)<=min(k_mat, n_mat);
        sum(z)==k_mat;
        z>=0;
        z<=ones(m_mat,1);

        minimize quantum_rel_entr(Y, A_mat*diag(z)*A_mat'+eps_mat*eye(n_mat))-(n_mat-trace(Y))*log(eps_mat)
    cvx_end

      $ofv_dual=-quantum_rel_entr(Y, A_mat*diag(z)*A_mat'+eps_mat*eye(n_mat))+(n_mat-trace(Y))*log(eps_mat);
      disp(ofv_dual)
      $z_relax=z
      [~,inds]=maxk(z, k_mat);
      $z_rounded=zeros(m_mat,1);
      $z_rounded(inds)=1.0;
      $ofv_primal=log(det(A_mat*diag(z_rounded)*A_mat'+eps_mat*eye(n_mat)));

    """



    @show ofv_primal, ofv_dual, z_relax, z_rounded

    return z_relax, ofv_dual, z_rounded, ofv_primal
end
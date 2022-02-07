clear;
    for seednum=1:20
    n_mat=10;
    m_mat=20;
    rng(seednum,'twister');
    A_mat=1/(n_mat^0.25)*randn(n_mat,m_mat);
    eps=1e-6;
    tStart = tic;

    for k_mat=1:(n_mat-1)


% First: solve the problem using the standard d-opt design approach and rounding
    cvx_begin
        variable z(m_mat)

        sum(z)==k_mat;
        z>=0;
        z<=ones(m_mat,1);

        maximize log_det(A_mat*diag(z)*A_mat'+eps*eye(n_mat))
        t1 = toc(tStart);
    cvx_end
        t2 = toc(tStart);
      t_basic = t2 - t1;

    LB_basic=log_det(A_mat*diag(z)*A_mat'+eps*eye(n_mat));
    [~,inds]=maxk(z, k_mat);
     z_rounded=zeros(m_mat,1);
     z_rounded(inds)=1.0;
     ofv_primal_basic=log(det(A_mat*diag(z_rounded)*A_mat'+eps*eye(n_mat)));
     gap_basic=abs(ofv_primal_basic-LB_basic)/abs(ofv_primal_basic);


% Second: solve the problem using a submodular maximization heuristic
    t1 = toc(tStart);
    ofv_submodular=0.0;
    z_submodular=zeros(m_mat,1);
    for t=1:k_mat
        [z_submodular, ofv_submodular]=getkplusone(A_mat, n_mat, z_submodular, eps);
    end
    t2 = toc(tStart);
    t_submodular=t2-t1;
    gap_submodular=abs(ofv_submodular-LB_basic)/abs(ofv_submodular);

% Last: solve the problem by using the new quantum relaxation and rounding

    cvx_begin

        variable Y(n_mat,n_mat) semidefinite
        variable z(m_mat)

        eye(n_mat)-Y==semidefinite(n_mat);
        trace(Y)<=min(k_mat, n_mat);
        sum(z)==k_mat;
        z>=0;
        z<=ones(m_mat,1);

        minimize quantum_rel_entr(Y, A_mat*diag(z)*A_mat'+eps*Y)-(n_mat-trace(Y))*log(eps)
      t1 = toc(tStart);
      cvx_end

      t2 = toc(tStart);
      t_persp = t2 - t1;

      ofv_dual_quant=-quantum_rel_entr(Y, A_mat*diag(z)*A_mat'+eps*Y)+(n_mat-trace(Y))*log(eps);

      [~,inds]=maxk(z, k_mat);
      z_rounded=zeros(m_mat,1);
      z_rounded(inds)=1.0;
      ofv_primal_quant=log(det(A_mat*diag(z_rounded)*A_mat'+eps*eye(n_mat)));

      gap_persp=abs(ofv_primal_quant-ofv_dual_quant)/abs(ofv_primal_quant);

      % Write all results to csv
      fid = fopen( 'createFig1_new.csv', 'a' );
      fprintf(fid, '\r\n'); % Need on seperate line since Matlab is finicky...
      fprintf( fid, '%d,%d, %d, %d, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f', n_mat, m_mat, k_mat, seednum, eps, gap_basic, gap_submodular, gap_persp, ofv_primal_basic, ofv_submodular, ofv_primal_quant, t_basic, t_submodular, t_persp);
      fclose( fid );
    end
end



function [z0, ofv_best]=getkplusone(A_mat, n_mat, z0, eps)
    candidateIndices=find(~z0);
    iBest=-1.0;
    ofvBest=-1e6;
    Sigma_Gram_current=A_mat*diag(z0)*A_mat';
    for j=1:size(candidateIndices, 1)
        i=candidateIndices(j);
        ofv_test=log(det(Sigma_Gram_current+A_mat(:,i)*A_mat(:,i)'+eps*eye(n_mat)));
        if ofv_test>ofvBest
            ofvBest=ofv_test;
            iBest=i;
        end
    end
    z0(iBest)=1.0;
    ofv_best=log(det(Sigma_Gram_current+A_mat(:,iBest)*A_mat(:,iBest)'+eps*eye(n_mat)));
end

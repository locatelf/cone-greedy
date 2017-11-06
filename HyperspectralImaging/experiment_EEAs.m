% Running the algorithm on data set M 

function [Kall, Hall, resultsErr, Hall2, resultsErr2, nEEAs,resultsErr3,resultsErr4,resultsErr5] = experiment_EEAs( M , r  )

% 2. Running the EEAs on the full and subsampled data set
maxitNNLS = 10; 
nEEAs{1} = 'SPA  '; 
nEEAs{2} = 'VCA  '; 
nEEAs{3} = 'XRAY '; 
nEEAs{4} = 'H2NMF'; 
nEEAs{5} = 'SNPA '; 
for algo = 1 : 5
    
    D = M;   % full data set
    fprintf([nEEAs{algo}, ' on the full data set...']);
    
    %% Normalization to apply EEAs that need it (SPA, VCA, H2NMF, SNPA)
    
    e = cputime;
    K = EEAs(D,r,algo); % extract endmembers from 'dictionary'
    Kall{algo} = K;
    % Error
    [H, err] = plotnnlsHALSupdt(M,D(:,K),[],maxitNNLS,algo); % optimal weights
    [H2, Hslow,HPW,HA] = NNMPmatrix(M,D(:,K),maxitNNLS,algo,err,nEEAs{algo});
    Hall{algo} = H;
    Hall2{algo} = H2;

    resultsErr(algo) = 100*norm(M - D(:,K)*H,'fro')/norm(M,'fro');
    resultsErr2(algo) = 100*norm(M - D(:,K)*H2,'fro')/norm(M,'fro');
    resultsErr3(algo) = 100*norm(M - Hslow,'fro')/norm(M,'fro');
    resultsErr4(algo) = 100*norm(M - D(:,K)*HPW,'fro')/norm(M,'fro');
    resultsErr5(algo) = 100*norm(M - D(:,K)*HA,'fro')/norm(M,'fro');

    fprintf(' done.\n');
end


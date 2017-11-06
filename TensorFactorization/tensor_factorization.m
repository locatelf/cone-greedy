%clear
rng('default')

it = 1;

low_rank = 20;
t_e = zeros(low_rank,it);
for i=1:it
    nil = 0;
    prc_trn = 0.5;
    [X_trn,X_tst,idx_trn,idx_tst] = getdcm(prc_trn);
    
    X_tst = X_trn;
    idx_tst =[];
    
    
    
    X0 = zeros(size(X_trn));
    lmo_it = 100;
    f_reopt = 100;
    
    [X_pred,iter,REC] = ncp(tensor(X_trn),low_rank,'method','anls_bpp');
    norm(tensor(X_trn)-full(X_pred))/norm(X_trn(:))
    [X_pred,iter,REC] = ncp(tensor(X_trn),low_rank,'method','anls_asgroup');
    norm(tensor(X_trn)-full(X_pred))/norm(X_trn(:))
    [X_pred,iter,REC] = ncp(tensor(X_trn),low_rank,'method','hals');
    norm(tensor(X_trn)-full(X_pred))/norm(X_trn(:))
    [X_pred,iter,REC] = ncp(tensor(X_trn),low_rank,'method','mu');
    norm(tensor(X_trn)-full(X_pred))/norm(X_trn(:))
    
    [X_pred, residual2,test_error] = PWNMP(X_trn,X0,low_rank,nil,lmo_it,idx_tst,X_tst);
    norm(X_trn(:)-X_pred(:))/norm(X_trn(:))
    [X_pred, residual3,test_error] = FCNMP(X_trn,X0,low_rank,nil,lmo_it,idx_tst,X_tst);
    norm(X_trn(:)-X_pred(:))/norm(X_trn(:))
    
end


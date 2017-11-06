function [Vout,Vslow,VPW,VA] = NNMPmatrix(M,U,maxit,algo,errOld,tit)
    U = [zeros(size(U,1),1) U];
    V = zeros(size(U,2),size(M,2));
    Vslow = zeros(size(M,1),size(M,2));
    VPW = zeros(size(U,2),size(M,2));

    errors{1} = zeros(maxit,size(M,2));
    errors{2} = zeros(maxit,size(M,2));
    errors{3} = zeros(maxit,size(M,2));
    errors{4} = zeros(maxit,size(M,2));

    dims = size(M,1);
    fun = eye(dims);
    cnt = 0;
    for i=1:size(M,2)

        [Vslow(:,i),errors{1}(:,i)] = NNMPslow(U,M(:,i),fun,maxit,dims);
        [VA(:,i),errors{4}(:,i)] = AMP(U,M(:,i),fun,maxit,dims);        
        [VPW(:,i),errors{2}(:,i),cnt,itdone] = NNMP(U,M(:,i),fun,maxit,dims,cnt);
        [V(:,i),errors{3}(:,i)] = FCNNMP(U,M(:,i),fun,maxit,dims);
    end
    er1 = sqrt(sum(errors{1},2));
    er2 = sqrt(sum(errors{2},2));
    er3 = sqrt(sum(errors{3},2));
    er4 = sqrt(sum(errors{4},2));

    optimum = min([errOld(end),er1(end),er2(end),er3(end),er4(end)]);
    subplot(5,1,algo)
    plot(1:size(errOld,1),errOld-optimum,'LineWidth',5);
    title(tit);
    hold on
    
    subplot(5,1,algo)
    plot (1:size(er1,1),er1-optimum,'LineWidth',5);
    
    
    subplot(5,1,algo);
    plot (1:size(er2,1),er2-optimum,'LineWidth',5)

    subplot(5,1,algo)
    plot (1:size(er3,1),er3-optimum,'LineWidth',5);
    
    subplot(5,1,algo)
    plot (1:size(er4,1),er4-optimum,'LineWidth',5);
    if strcmp(tit,'SPA  ')
        hL = legend('nnlsHALSupdt','NNMP','PWMP','FCMP','AMP');
    end
    xlabel('Iteration')
    ylabel('Suboptimality')
    set(gca,'FontSize',24)
    Vout = V(2:end,:);
    VPW = VPW(2:end,:);
    VA = VA(2:end,:);
%     (size(M,2)*itdone)
%     cnt
%     
end
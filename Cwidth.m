rng(0)
ratios = zeros(20,numel(0.1:0.1:pi/2-0.1));
for outIt = 1:20
    alpha=zeros(2,1);
    alphaStar = rand(2,1);
    y=[-1;1].*alphaStar;
    i = 1;
    for theta = 0.1:0.1:pi/2-0.1
        A = [-1, cos(theta); 0, sin(theta)];
        radius = 1;
        diam = norm(A(:,1)-A(:,2));
        x = A*alpha;
        errorInit = 0.5*norm(y-x)^2;
        grad = -(y-x);
        if dot(grad,A(:,1))<=dot(grad,A(:,2))
            z = A(:,1);
            idxZ = 1;
        else
            z = A(:,2);
            idxZ = 2;
        end
        v = zeros(2,1);
        gamma = -dot(grad,z-v)/radius^2;
        x = x + gamma *(z-v);
        empirical = 0.5*norm(y-x)^2;
        theoretical = (1-sin(theta/2)^2/diam^2)*errorInit;
        ratios(outIt,i) = theoretical/empirical;
        i = i+1;
    end
end

rez_fin=ratios;

res = mean(rez_fin);
error_bar_max = max(rez_fin)-res;
error_bar_min = res-min(rez_fin);

%close all
figure
h=errorbar(0.1:0.1:pi/2-0.1,res,error_bar_min,error_bar_max,'LineWidth',1.3);
set(gca,'xscale','log')
set(gca,'FontSize',20)
axis([-inf,2,0.5,inf])
xlabel('$\theta$ (rad)','Interpreter','LaTex','FontSize', 30)
ylabel('ratio', 'FontSize', 30)





%plot(0.1:0.1:pi/2-0.1,ratios)

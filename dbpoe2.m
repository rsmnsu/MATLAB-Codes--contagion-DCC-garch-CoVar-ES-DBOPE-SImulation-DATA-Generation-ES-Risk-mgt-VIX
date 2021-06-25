clc
clear
%% Simulate data set
mu = [0 0 0 0 0 0];
A = rand(6);
Sigma= A * A'
%% Method 1 checking if the matrix semi definite
try chol(Sigma)
    disp('Matrix is symmetric positive definite.')
catch ME
    disp('Matrix is not symmetric positive definite')
end
%% Method 2 checking if the matrix semi definite
tf = issymmetric(Sigma)
d = eig(Sigma)
isposdef = all(d > 0)
ispossemdef=all(d >= 0)

%% Simulation of variables
rng('default')  % For reproducibility
Data = mvnrnd(mu,Sigma,1000);
r=Data(:,1);
T= length(r);
conditionalvariance=[];

%p= [0.5, 0,1, 0.05, 0.025,0.01, 0.001];
p= [0.025];
VarMdl = garch(1,1)
Mdl = arima('ARLags',1,'Variance',VarMdl);
EstMdl = estimate(Mdl,r);
[res,v,logL] = infer(EstMdl,r);
conditionalvariance=[conditionalvariance,v];
Sigma=conditionalvariance;
ESdynamic=[];
VaRdynamic=[];
bpoe=[];
progressbar

for J= 1:T
[Var_Normal, ES_Normal]=hNormalVaRES(Sigma(J),p);
VaR=Var_Normal;
ES=ES_Normal;

disp(J)
disp('');
ESdynamic=[ESdynamic,ES];
VaRdynamic=[]

progressbar(J/T)
end
ES=movmean(ESdynamic, 250);
plot(ES)
%save ES2.mat ES Var_Normal EST5 EST10 Var_T10 Var_T5


function [VaR,ES] = hNormalVaRES(Sigma,p)
    % Compute VaR and ES for normal distribution
    % See [4] for technical details
    
    VaR = -norminv(p);
    ES = -Sigma*quad(@(q)q.*normpdf(q),-6,-VaR)/p;

end
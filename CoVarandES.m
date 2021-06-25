clc
clear
% Simulate data set
mu = [0 0 0 0 0 ];
A = rand(5);
Sigma= A * A';

% Simulation of variables
rng('default')  % For reproducibility
Data = mvnrnd(mu,Sigma,1000);
r=Data(:,1);
T= length(r);
conditionalvariance=[];
p = [0.5, 0.1, 0.05, 0.025, 0.01, 0.001];
VarMdl = garch(1,1);
Mdl = arima('ARLags',1,'Variance',VarMdl);
EstMdl = estimate(Mdl,r);
[res,v,logL] = infer(EstMdl,r);
conditionalvariance=[conditionalvariance,v];
Sigma=conditionalvariance;
ESdynamic=[];
VaRdynamic=[];
bpoe=[];

for P_Index = 1: +1: length(p) 
P_Value = p(P_Index);
for J= 1:T
[Var_Normal, ES_Normal]=hNormalVaRES(Sigma(J),P_Value);
VaR = Var_Normal;
ES = ES_Normal;
disp(J)
disp('');
ESdynamic = [ESdynamic,ES];
VaRdynamic = []
end
ES_Matrix(:,P_Index) = ESdynamic';
plot(ES_Matrix(:,P_Index));
hold on

hold off

function [VaR,ES] = hNormalVaRES(Sigma,p)
    % Compute VaR and ES for normal distribution
    % See [4] for technical details
    
    VaR = -norminv(p);
    ES = -Sigma*quad(@(q)q.*normpdf(q),-6,-VaR)/p;

end
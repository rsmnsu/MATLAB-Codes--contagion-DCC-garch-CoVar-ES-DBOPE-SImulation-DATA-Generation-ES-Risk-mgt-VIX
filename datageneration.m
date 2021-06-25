clc
clear
%% Simulate data set
mu = [0 0 0 0 0 0];
A = rand(6)
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
Data = mvnrnd(mu,Sigma,100000);
r=Data(:,1);
T= length(r);
% plot(movmean(Data(:,1), 2500));
% hold on
% yyaxis right
% plot(movmean(Data(:,2), 2500));
% hold off

%% 
% Estimate conditional variance
VarMdl = garch(1,1);
Mdl = arima('ARLags',1,'Variance',VarMdl);
EstMdl = estimate(Mdl,r);
[res,v,logL] = infer(EstMdl,r);
conditionalvariance=v;
%% Estimate ES
VarLevel= 0.975;

Mu=0;

SigmaNormal=conditionalvariance;

SigmaT10= v*sqrt((10-2)/10);

SigmaT5=v*sqrt((5-2)/5);

[Var_Normal, ES_Normal]=hNormalVaRES(Mu,SigmaNormal,VarLevel);

[Var_T10,ES_T10] = hTVaRES(10,Mu,SigmaT10,VarLevel);

[Var_T5,ES_T5] = hTVaRES(5,Mu,SigmaT5,VarLevel);

ES=ES_Normal;

plot(movmean(-Var_Normal,250));
hold on
plot(movmean(-ES_Normal,250));
%yyaxis right
plot(movmean(Data(:,1), 250));
% hold off



%% Define VaR and ES local functions
function [VaR,ES] = hNormalVaRES(Mu,Sigma,VaRLevel)

    % Compute VaR and ES for normal distribution

   

    VaR = -1*(Mu-Sigma*norminv(VaRLevel));

    ES = -1*(Mu-Sigma*normpdf(norminv(VaRLevel))./(1-VaRLevel));

 

end

function [VaR,ES] = hTVaRES(DoF,Mu,Sigma,VaRLevel)

    % Compute VaR and ES for t location-scale distribution

    % See [4] for technical details

 

    VaR = -1*(Mu-Sigma*tinv(VaRLevel,DoF));

    ES_StandardT = (tpdf(tinv(VaRLevel,DoF),DoF).*(DoF+tinv(VaRLevel,DoF).^2)./((1-VaRLevel).*(DoF-1)));

    ES = -1*(Mu-Sigma*ES_StandardT);

 

end

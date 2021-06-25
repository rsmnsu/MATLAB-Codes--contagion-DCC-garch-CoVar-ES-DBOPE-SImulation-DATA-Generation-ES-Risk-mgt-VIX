clc
clear
load Data_EquityIdx
nasdaq = DataTable.NASDAQ;
r = 100*price2ret(nasdaq);
r = r(2000:end, :);
T = length(r);
conditionalvariance=[];

%p= [0.5, 0,1, 0.05, 0.025,0.01, 0.001];
p= [0.025];
VarMdl = garch(1,1)
Mdl = arima('ARLags',1,'Variance',VarMdl)
EstMdl = estimate(Mdl,r);
[res,v,logL] = infer(EstMdl,r);
conditionalvariance=[conditionalvariance,v];
Sigma=conditionalvariance;
ESdynamic=[];
bpoe=[];
progressbar

for J= 1:T
[Var_Normal, ES_Normal]=hNormalVaRES(Sigma(J),p);
ES=ES_Normal;

disp(J)
disp('');
ESdynamic=[ESdynamic,ES];

progressbar(J/T)
end
ES=ESdynamic;

%save ES2.mat ES Var_Normal EST5 EST10 Var_T10 Var_T5


function [VaR,ES] = hNormalVaRES(Sigma,p)
    % Compute VaR and ES for normal distribution
    % See [4] for technical details
    
    VaR = -norminv(p);
    ES = -Sigma*quad(@(q)q.*normpdf(q),-6,-VaR)/p;

end
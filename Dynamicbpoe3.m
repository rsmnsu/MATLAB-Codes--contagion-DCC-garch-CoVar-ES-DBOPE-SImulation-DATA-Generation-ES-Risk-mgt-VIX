clc
clear
%% Load index and convert to returns 
load Data_EquityIdx
nasdaq = DataTable.NASDAQ;
r = 100*price2ret(nasdaq);
T = length(r);
t = 1000;
%% Declare storage
conditionalvariance=[];

%p= [0.5, 0,1, 0.05, 0.025,0.01, 0.001];
%% Estimate Conditional variances
q = 0.05;
bPoE= 0.1;
VarMdl = garch(1,1)
Mdl = arima('ARLags',1,'Variance',VarMdl)
EstMdl = estimate(Mdl,r);
[res,v,logL] = infer(EstMdl,r);
conditionalvariance=[conditionalvariance,v];
Sigma=conditionalvariance;

%% Estimate dynamic ES and dynamic BpoE
ESdynamic=[];
bPoEdynamic =[];
progressbar

for J= t:T;
    r1 = r(J-t+1:J);
    ES=hNormalVaRES(Sigma(J),q,r1)
    ESdynamic=[ESdynamic,ES];
    disp(J);
    disp('');
    bPoE = samplebPoE(r1, ES, bPoE);
    bPoEdynamic = [bPoEdynamic, bPoE];
    progressbar(J/T)
end
%save ES2.mat ES Var_Normal EST5 EST10 Var_T10 Var_T5
bPoEdynamic=bPoEdynamic';
ESdynamic=ESdynamic';
%% Plot BpoE and ES
plot(bPoEdynamic(bPoEdynamic~=0))
hold on
yyaxis right
plot(ESdynamic(ESdynamic~=0))
hold off
%plot(bPoEdynamic)
%plot(ESdynamic)
%% local function for Normally distributed ES 
function ES = hNormalVaRES(Sigma,p,x)
    % Compute VaR and ES for normal distribution
    % See [4] for technical details
    mu = mean(x);   
    ES = mu + sqrt(Sigma)*normpdf(norminv(p))./p;
end
%% the function S = sampleES estimates bpoe. 
% I am proposing a mean reverting / smoothening model for the ES or the bPoE itself, 
% that would avoid large changes in the ES which makes it approach infinity
% The code calculate bPoE for rolling period, but without updating the bPoE
function S = samplebPoE(x,v,p)
    amin = fminbnd(@(a)sum(max(a.*(x-v)+1,0)), 0, 1e8);
    x1 = max(amin.*(x - v) + 1,0);
    S = mean(x1);

end
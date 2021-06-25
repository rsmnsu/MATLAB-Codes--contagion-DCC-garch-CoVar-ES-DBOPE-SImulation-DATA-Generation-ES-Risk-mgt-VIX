clc
clear
load Data_EquityIdx
nasdaq = DataTable.NASDAQ;
r = 100*price2ret(nasdaq);
%r = r(2000:end, :);
T = length(r);
t = 100;
conditionalvariance=[];
MAX = 1e8;

%p= [0.5, 0,1, 0.05, 0.025,0.01, 0.001];
alpha= 0.95;
p= 1-alpha;
VarMdl = garch(1,1)
Mdl = arima('ARLags',1,'Variance',VarMdl)
EstMdl = estimate(Mdl,r);
[res,v,logL] = infer(EstMdl,r);
conditionalvariance=[conditionalvariance,v];
Sigma=conditionalvariance;
ESdynamic=[];
bPoEdynamic =[];
progressbar

for J= t:T
    r1 = r(J-t+1:J);
    ES=hNormalVaRES(Sigma(J),p,r1);
    ESdynamic=[ESdynamic,ES];
    disp(J);
    disp('');
    amin = fminbnd(@(a)sampleES(a, r1, ES),0, MAX);
    bPoE = sampleES(amin, r1, ES);
    bPoEdynamic = [bPoEdynamic, bPoE];
    p = bPoE;
    progressbar(J/T)
end
% ES=ESdynamic;
% save ES2.mat ES Var_Normal EST5 EST10 Var_T10 Var_T5
plot(bPoEdynamic)
hold on
plot(ESdynamic)
hold off

function ES = hNormalVaRES(Sigma,p,x)
    % Compute VaR and ES for normal distribution
    % See [4] for technical details
    mu = mean(x)   
    ES =  mu + sqrt(Sigma)*normpdf(norminv(p))./p;

end
function S = sampleES(a,x,v)
    x1 = max(a.*(x - v) + 1,0);
    S = sum(x1)./length(x1);
    if v >= max(x)
        S = 0;
    end

end

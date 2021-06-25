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

VaRLevel = 0.975;
EstimationWindowSize= 250;
% Estimate volatility over the test window
Volatility = zeros(length(T),1);

for i = T+1
   
   EstimationWindow = T-EstimationWindowSize:T-1;
   
   Volatility(i) = std(r(EstimationWindow));
   
end

% Mu=0 in this example
Mu = 0;

% Sigma (standard deviation parameter) for normal distribution = Volatility
SigmaNormal = Volatility;
% Sigma (scale parameter) for t distribution = Volatility * sqrt((DoF-2)/DoF)
SigmaT10 = Volatility*sqrt((10-2)/10);
SigmaT5 = Volatility*sqrt((5-2)/5);

% Estimate VaR and ES, normal
[VaR_Normal,ES_Normal] = hNormalVaRES(Mu,SigmaNormal,VaRLevel);
% Estimate VaR and ES, t with 10 and 5 degrees of freedom
[VaR_T10,ES_T10] = hTVaRES(10,Mu,SigmaT10,VaRLevel);
[VaR_T5,ES_T5] = hTVaRES(5,Mu,SigmaT5,VaRLevel);

disp('VaR_Normal=')
disp(VaR_Normal)
disp('ES_Normal=')
disp(ES_Normal)
%%
%The following plot shows the daily returns, and the VaR and ES estimated with the normal method.
% figure;
% plot(r,-VaR_Normal,-ES_Normal)
% title('Historical VaR and ES')
% grid on


%% Local function
function [VaR,ES] = hHistoricalVaRES(Sample,VaRLevel)
    % Compute historical VaR and ES
    % See [7] for technical details

    % Convert to losses
    Sample = -Sample;
    
    N = length(Sample);
    k = ceil(N*VaRLevel);
    
    z = sort(Sample);
    
    VaR = z(k);
    
    if k < N
       ES = ((k - N*VaRLevel)*z(k) + sum(z(k+1:N)))/(N*(1 - VaRLevel));
    else
       ES = z(k);
    end
end

function [VaR,ES] = hNormalVaRES(Mu,Sigma,VaRLevel)
    % Compute VaR and ES for normal distribution
    % See [6] for technical details
    
    VaR = -1*(Mu-Sigma*norminv(VaRLevel));
    ES = -1*(Mu-Sigma*normpdf(norminv(VaRLevel))./(1-VaRLevel));

end

function [VaR,ES] = hTVaRES(DoF,Mu,Sigma,VaRLevel)
    % Compute VaR and ES for t location-scale distribution
    % See [6] for technical details

    VaR = -1*(Mu-Sigma*tinv(VaRLevel,DoF));
    ES_StandardT = (tpdf(tinv(VaRLevel,DoF),DoF).*(DoF+tinv(VaRLevel,DoF).^2)./((1-VaRLevel).*(DoF-1)));
    ES = -1*(Mu-Sigma*ES_StandardT);

end
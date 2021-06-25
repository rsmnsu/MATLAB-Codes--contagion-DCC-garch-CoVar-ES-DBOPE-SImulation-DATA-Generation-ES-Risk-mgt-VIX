clc
clear

%% Simulation input
randn('state',0);    % set seed
S=1e7;               % number of simulations
s2= 0.01^2;          % daily variance
p=0.01;              % probability
r= 0.05;             % risk free
P= 100;              % Today's price

ysim = randn(S,1)* sqrt(s2)+r/365-0.5*s2;   % simulate return
Psim = P * exp(ysim);                        % future prices
q= sort(Psim - P);                         % sort sim P/L
VaR1 = -q(S * p);
disp('one asset simulated VaR')
disp(VaR1)
%% Simulate two asset returns

mu = [r/365 r/365]';                  % return mean
Sigma = [0.01 0.0005; 0.0005 0.02];    % covariance matrix
randn('state',12);                   % set seed
y= mvnrnd(mu,Sigma,S);              % simulated returns
K = 2;                              % 2 assets
P = [100 50]';                       % prices
x = [1 1]';                          % number of assets
Port = P'* x;                        % portfolio at t 
Psim = repmat(P,1,S)' .* exp(y);     % sim prices 
PortSim = Psim * x;                  % sim portfolio
q= sort(PortSim - Port);            % simulated P/L
VaR4 = -q(S * p);
disp('multi asset simulated VaR')
disp(VaR4)
%% ES 
value=1000;
VAR = -norminv(p);
ES = -Sigma * quad(@(q)q.* normpdf(q),-6,-VAR)/p * value;
disp('Multivariate ES simulated=')
disp(ES);




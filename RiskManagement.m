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
Data = mvnrnd(mu,Sigma,100000);
r=Data(:,1);
T= length(r);

%% Adjustable data
ys= sort(Data);
p=0.51;
value=100;
op= T*p;
w= [0.17;0.18;0.19;0.16;0.15;0.15];
y=Data;
%% Multivariate Historic Simulation
yp=y*w;
yps=sort(yp);
format compact
VaR= -yps(op)*value;
disp('Multivariate Historic Simulation')
sprintf('%0.2f',VaR)
%% Multivariate VaR for Normal Distribution
sigma= sqrt(w'*cov(y)*w);
VaR2=-sigma*norminv(p)*value;
disp('Multivariate VaR')
disp(VaR)
%% ES 
VAR = -norminv(p);
ES = -sigma * quad(@(q) q.* normpdf(q),-6,-VAR)/p * value;
disp('Multivariate ES')
disp(ES);





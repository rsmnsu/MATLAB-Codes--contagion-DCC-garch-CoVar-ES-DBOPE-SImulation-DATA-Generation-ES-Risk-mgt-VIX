

%% Load Data
load('C:\Users\islamr\OneDrive - University of Tasmania\PAPER3COVIDADDITION\Newcode\finalreturns.mat');

load('C:\Users\islamr\OneDrive - University of Tasmania\Mardi Meetings\Meeting 22\Testing\MyTesting.mat');
CP1=MyTesting;
CP1(:,14)=[];
CP1(:,32:36)=[];

%% Pre-Processing Data
CPNew = log(CP1); 
CPNew= 200*(trimr(CPNew,1,0)-trimr(CPNew,0,1));
window_Size = 2;
delta = (1/window_Size)*ones(1,window_Size);
gama=1;
filt=filter(delta,gama,CPNew);
y=filt;
y1=[y;finalreturns];
%%
iniGuess = 10;
volatility = fsolve(@(x) myfunc(x,261,0.05,1033.56, 775, 1/52),iniGuess);




%% Black Scholes Functions
function C = bs ( Interest, Volatility, Stock, StrikePrice, TimeToMaturity )
d1 = (log(Stock ./ StrikePrice) + (Interest + (Volatility .^ 2) ./ 2) .* TimeToMaturity) ./ (Volatility .* sqrt(TimeToMaturity));
d2 = (log(Stock ./ StrikePrice) + (Interest - (Volatility .^ 2) ./ 2) .* TimeToMaturity) ./ (Volatility .* sqrt(TimeToMaturity));
C = normcdf(d1) .* Stock - normcdf(d2) .* StrikePrice .* exp(-Interest .* TimeToMaturity);
end
%% Solver function
function F = myfunc(vol,C,Interest, Stock, StrikePrice, TimeToMaturity)
F = C - bs(Interest,vol,Stock,StrikePrice,TimeToMaturity);
end
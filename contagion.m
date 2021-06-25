%=========================================================================
%
%   Compute GMM estimates of a contagion model
%
%=========================================================================
    clc
    clear
    format short
    RandStream.setGlobalStream( RandStream('mt19937ar','seed',123) );
    % Simulate data set
    mu = [0 0 0 0 0 0 0];
    A = rand(7);
    Sigma= A * A';

    % Simulation of variables
    %rng('default')  % For reproducibility

    et = mvnrnd(mu,Sigma,1000);
    t  = length(et);
    
    % Generate squares and cross products of columns of et
    [rows,cols] = size(et);
    e2          = zeros(rows,cols*cols);
    k=1;
    for j = 1:cols   
        for i = 1:cols               
            e2(:,k) = et(:,j).*et(:,i);
            k=k+1;	
       end
    end
    
    % Drop repeated columns
    e2 = e2(:,vech( reshapeg(1:1:cols*cols,cols,cols) ) );

    % Estimate the uncontrained model
    ops    = optimset('LargeScale','off','Display','off');
    theta0 = rand(20,1) ;

    [theta,qu,~,~,H] = fminunc(@(b) gmmcrit(b,e2),theta0,ops);

    
    disp(' ');
    disp(['Value of objective function = ', num2str(qu) ]);
	disp(' ');
    disp(['J statistic                = ', num2str(2*t*qu) ]);
    nu = size(e2,2)-length(theta);
    if  nu > 0.0
        disp(['p-value                 = ', num2str(1-chi2cdf(2*t*qu,nu)) ]);
    end

    disp(' ')
    total = theta(1:7).^2 + theta(8:14).^2 + [theta(15:20).^2 ; 0];
    disp('   Common   Idiosync.   Contagion ') 
    disp( 100*( [ (theta(1:7).^2)./total (theta(8:14).^2)./total   [theta(15:20).^2 ; 0]./total ] ) );

    % Estimate the contrained model
    ops    = optimset('LargeScale','off','Display','off');
    theta0 = rand(14,1) ;

    [theta,qc,~,~,~,H] = fminunc(@(b) gmmcritc(b,e2),theta0,ops);

    stat = 2*t*(qc - qu);
    disp(' ');
    disp(['Value of objective function = ', num2str(qc) ]);
    disp(['Test of contagion = ', num2str(stat) ]);
    disp(['P-value           = ', num2str(1-chi2cdf(stat,6)) ]);
    

%
%--------------------------- Functions -----------------------------------
% 
%-------------------------------------------------------------------------
% GMM unconstrained objective function  
%-------------------------------------------------------------------------
function q = gmmcrit(b,e2)
	
    lam  = b(1:7);
    phi  = b(8:14);
    gam  = b(15:20);
   
    a        = [ lam diag(phi)] ;
    a(1:6,8) = gam;
    
    m = bsxfun( @minus,e2,vech(a*a')' );
    w = m'*m/length(m);
    q = 0.5*mean(m)*inv(w)*mean(m)';
end

%-------------------------------------------------------------------------
% GMM constrained objective function  
%-------------------------------------------------------------------------
function q = gmmcritc(b,e2)
	
    lam  = b(1:7);
    phi  = b(8:14);
    gam  = zeros(6,1);
   
    a        = [ lam diag(phi)] ;
    a(1:6,8) = gam;
    
    m = bsxfun( @minus,e2,vech(a*a')' );
    w = m'*m/length(m);
    q = 0.5*mean(m)*inv(w)*mean(m)';


end



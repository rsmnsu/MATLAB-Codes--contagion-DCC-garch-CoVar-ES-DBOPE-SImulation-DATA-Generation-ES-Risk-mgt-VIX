%=========================================================================
%
%   Estimate the Dynamic Conditional Correlation (DCC) model of 
%
%=========================================================================
    clc
    clear
    % Simulate data set
    mu = [0 0 0 0 0 ];
    A = rand(5);
    Sigma= A * A';

    % Simulation of variables
    rng('default')  % For reproducibility
    data = mvnrnd(mu,Sigma,1000);

   % data = yields(:,[1 5 10 15 20]);
    %data = yields(:,[1 5]);
    y    = 100*(trimr(data,1,0) - trimr(data,0,1));                                   
    y    = bsxfun(@minus,y,mean(y));
 
    [ t,n ] = size( y ); 
        
    % Estimate univariate GARCH models for each variable   
    ops  = optimset( 'LargeScale','off','Display','off' );
    b1   = zeros( 3,n );
    st1  = zeros(3,n);
    h1   = zeros( t,n );

    for k = 1:n 
        
        bstart      = [0.1  -2  2 ];
        tmp         = fminunc( @(b) negloglgarch(b,y(:,k)),bstart,ops);  
        st1(:,k)    = tmp;
        b1(:,k)     = [ tmp(1) normcdf(tmp(2)) normcdf(tmp(3)) ];
        [~,h1(:,k)] = negloglgarch( tmp,y(:,k) );

    end
    b1([2 3],:) = normcdf( b1([2 3],:) );
    disp( 'Garch parameters' );
    disp( b1 );   
    
    clear start
    % Estimate the correlation component
    start = [-2, 2];
    bc    = fminunc(@(b) neglogcor(b,y,h1),start,ops);
    st2   = bc;
    bc    = normcdf( bc );
    disp( 'Correlation components' );
    disp( bc );

    clear start
    
    % Estimate the correlation component
    ops   = optimset( 'LargeScale','off','Display','iter','MaxFunEvals',5000 );
    start = [ st1(:); st2(:) ];
    bf    = fminunc(@(b) neglogf(b,y),start,ops);

    ind = [ 2 3 5 6 8 9 11 12 14 15 16 17 ];
    bf(ind) = normcdf(bf(ind));
    disp([[b1(:); bc(:)] bf]);
    

%--------------------------- Functions  ----------------------------------
% 
%-------------------------------------------------------------------------
% Likelihood function for a GARCH(1,1) model
%-------------------------------------------------------------------------
function [lf,h] = negloglgarch( b,y ) 
    
    u = y ;    
    h = recserar(b(1) + normcdf(b(2))*trimr([0.0;u.^2],0,1),std(u)^2,normcdf(b(3)));
    z = u./sqrt(h);
    f = - 0.5*log( 2*pi ) - 0.5*log( h ) - 0.5*z.^2;
    
    lf = -mean( f );
end
%-------------------------------------------------------------------------
% Correlation component of DCC
%-------------------------------------------------------------------------
function lf = neglogcor(b,y,h1)

    b    = normcdf( b ); 
    [t,~]=size(y);

    f = zeros(t,1);
    u = y;
    z  = u./sqrt(h1);
    
    qbar = z'*z/t;                                          
    q    = qbar;

    for i = 1:t 

        % Diagonal matrix of conditional standard deviations   
        s  = diag(sqrt(h1(i,:)));   
        
        % Conditional correlation matrix  
        tmp = inv(diag(sqrt(diag(q))));
        r   = tmp*q*tmp;
                  
        f(i) = -0.5*log(det(r))-0.5*z(i,:)*inv(r)*z(i,:)'+0.5*z(i,:)*z(i,:)';              

        % Update q
        q  = abs(1-b(1)-b(2))*qbar + b(1)*z(i,:)'*z(i,:) + b(2)*q;
    end
    
    lf = -mean( f );

end
%-------------------------------------------------------------------------
% Full log-likelihood for DCC model
%-------------------------------------------------------------------------
function lf = neglogf(b,y)
    
    ind = [ 2 3 5 6 8 9 11 12 14 15 16 17 ];
    b(ind) = normcdf(b(ind));
   
    [t,n]=size(y);
    f = zeros(t,1);
    u = y;                                       
    hv = zeros(t,n);

    % Construct conditional variances
    hv(:,1) = recserar(b(1)+b(2)*trimr([0.0;u(:,1).^2],0,1),std(u(:,1))^2,b(3));
    hv(:,2) = recserar(b(4)+b(5)*trimr([0.0;u(:,2).^2],0,1),std(u(:,2))^2,b(6));      
    hv(:,3) = recserar(b(7)+b(8)*trimr([0.0;u(:,3).^2],0,1),std(u(:,3))^2,b(9));
    hv(:,4) = recserar(b(10)+b(11)*trimr([0.0;u(:,4).^2],0,1),std(u(:,4))^2,b(12));     
    hv(:,5) = recserar(b(13)+b(14)*trimr([0.0;u(:,5).^2],0,1),std(u(:,5))^2,b(15));    
    
    z    = u./sqrt(hv);                                                                                           
    qbar = z'*z/t;                                          
    q    = qbar;
    for i = 1:t 

        % Diagonal matrix of conditional standard deviations   
        s  = diag(sqrt(hv(i,:)));   
        
        % Conditional correlation matrix  
        tmp = inv(diag(sqrt(diag(q))));
        r   = tmp*q*tmp;
        
        % Update conditional variance-covariance matrix   
        h  = s*r*s;                                                                             
           
        f(i) = -0.5*n*log(2*pi)-0.5*log(det(h))-0.5*u(i,:)*inv(h)*u(i,:)';              

        % Update q
        q  = abs(1-b(16)-b(17))*qbar + b(16)*z(i,:)'*z(i,:) + b(17)*q;
    end
    
    lf = -mean( f );

end
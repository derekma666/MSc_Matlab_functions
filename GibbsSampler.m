function [dec runtime] = GibbsSampler(y,X,gstart,tau,penalty,shrink,N)

%%%%%%%%%%%%%%%Jason Bentley (2008) University of Canterbury%%%%%%%%%%%%%%%
%Gibbs sampler for Bayesian variable selection in linear regression using %
%Zellner's prior or Jeffreys prior with the binomial prior for gamma.     %
%INPUT: y is the n.1 response vector                                      %
%       X is n.(k+1) design matrix                                        %
%       gstart is the specified starting value for the Gibbs sampler, it  %
%           must contain a 1 in the first position and be of length k+1   %
%       tau is the choice of hyper-parameter in the constant Bernoulli    %
%           prior (using tau = 0.5 corresponds to a uniform prior)        %
%       penalty and shrink specify whether Jeffreys prior or Zellner's    %
%           prior is used. For Jeffreys penalty corresponds to p = 2*pi*  %
%           penalty with shrink = 1. For Zellner's prior penalty = (c+1)  %
%           and shrink should be set to c/(c+1).                          %
%       N is the number of samples to be generated                        %
%OUTPUT:dec is an N.1 vector of samples represented in decimal form       %
%       runtime is the required cputime to generate the N samples, memory %
%           for output storage                                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t = cputime; %start recording the time for the Gibbs sampler 
[n k] = size(X); %k includes the intercept
samp = zeros(N,k); %storage
samp(1,:) = gstart; %first row of sample is starting gamma vector
vv = X'*y; V = X'*X; A = [y'*y n/2 sqrt(penalty)*((1-tau)/tau) shrink];
%constants
i = 1; %track sample size 
while i < N %running time
    i = i + 1;
    g = samp(i-1,:); %take previous value
    for j = 2:k %Gibbs sampler
        [P1] = Update(V,g,j,A,vv); %calculate the P(gi=1)
        g(j) = rand <= P1;
    end % for j = 2:k+1
    samp(i,:) = g; %record updated vector as the next g vector
end
runtime = (cputime - t); %record cputime not including conversion of g to dec
dec = bin2dec(num2str(samp(:,2:k),'%1.f')); 
%turn vectors into dec values can do a max of 52 values!

function [P1] = Update(V,g,j,A,vv)

%%%%%%%%%%%%%%%Jason Bentley (2008) University of Canterbury%%%%%%%%%%%%%%%
%Update function for the conditional distribution of the posterior for    %
%model probabilities                                                      %
%INPUT: g is current binary vector                                        %
%       j is the current component being updated                          %
%       hats is the of individual Sums of Squares for each predictor      % 
%       A is various constants as above n the binomial                    %
%       vv is the covariance matrix for the full model                    %
%OUTPUT:P1 is the probability the component is = 1.                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

g1 = g; g1(j) = 1; g0 = g; g0(j) = 0; 
Vg1 =inv(V(g1==1,g1==1)); Vg0 = inv(V(g0==1,g0==1));
P1 = 1/(1+exp(log(A(3)) + A(2)*(log(A(1)-A(4)*vv(g1==1)'*Vg1*vv(g1==1))-...
    log(A(1)-A(4)*vv(g0==1)'*Vg0*vv(g0==1)))));

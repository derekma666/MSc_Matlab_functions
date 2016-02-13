function [dec runtime] = GibbsOSampler(y,X,gstart,tau,penalty,shrink,N)

%%%%%%%%%%%%%%%Jason Bentley (2008) University of Canterbury%%%%%%%%%%%%%%%
%Perfect Gibbs sampler for Bayesian variable selection with an orthogonal %
%design matrix W in linear regression using Zellner's prior or Jeffreys   %
%prior with the binomial prior for gamma                                  %
%INPUT: y is the n.1 response vector                                      %
%       X is n.(k+1) design matrix                                        %
%       gstart is the specified starting value for the Gibbs sampler, it  %
%           must contain a 1 in the first position and be of length k+1,  % 
%           it may be obtained using the perfect sampler                  %
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

t = cputime;
[n k] = size(X); i = 1; samp = zeros(N,k); 
for l = 1:k; hats(l) = y'*X(:,l)*inv(X(:,l)'*X(:,l))*X(:,l)'*y; end
A = [y'*y sqrt(penalty)*((1-tau)/tau) n/2 shrink];
samp(1,:) = ExactStart(hats,A,k);
while i < N    
    i = i + 1;    
    g = samp(i-1,:); %take previous value
    for j = 2:k %Gibbs sampler        
        [P1] = UpdateOrth(g,j,hats,A); %calculate the P(gi=1)
        g(j) = rand <= P1;
    end % for j = 2:k+1    
    samp(i,:) = g; %record updated vector as the next g vector
end
runtime = (cputime - t);
dec = bin2dec(num2str(samp(:,2:k),'%1.f')); %turn vectors into dec values, can do a max of 52 values!

function P1 = UpdateOrth(g,j,hats,A)

%%%%%%%%%%%%%%%Jason Bentley (2008) University of Canterbury%%%%%%%%%%%%%%%
%Update function for the conditional distribution of the posterior for    %
%model probabilities                                                      %
%INPUT: g is current binary vector                                        %
%       j is the current component being updated                          %
%       hats is the of individual Sums of Squares for each predictor      % 
%       A is various constants as above n the binomial                    %
%OUTPUT:P1 is the probability the component is = 1.                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

g0 = g; g0(j) = 0; g0hat = sum(hats(g0==1));
P1 = 1/(1+exp(log(A(2)) + A(3)*(log(1-((A(4)*hats(j))/(A(1)-(A(4)*g0hat)))))));


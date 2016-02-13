function [dec bct runtime] = GibbsPerfect(y,X,tau,penalty,shrink,N)

%%%%%%%%%%%%%%%Jason Bentley (2008) University of Canterbury%%%%%%%%%%%%%%%
%Perfect Gibbs sampler for Bayesian variable selection with an orthogonal %
%design matrix W in linear regression using Zellner's prior or Jeffreys   %
%prior with the binomial prior for gamma                                  %
%INPUT: y is the n.1 response vector                                      %
%       X is n.(k+1) design matrix                                        %
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

t = cputime; [n k] = size(X); v = 0; samp = zeros(N,k); bct = zeros(N,1);
for l = 1:k; hats(l) = y'*X(:,l)*inv(X(:,l)'*X(:,l))*X(:,l)'*y; end
A = [y'*y sqrt(penalty)*((1-tau)/tau) n/2 shrink];
while v <= N     
v = v + 1; 
m = 1; 
k1 = k - 1; 
g1 = ones(1,k); 
g0 = [1 zeros(1,k1)]; 
u = rand(k1,m); 
while ~isequal(g0,g1)    
    u = [rand(k1,m) u]; m = 2*m; g1 = ones(1,k); g0 = [1 zeros(1,k1)];    
    for i = 1:m % iterate from the past
       for j = 2:k % Gibbs sampler
            P10 = UpdateOrth(g0,j,hats,A);             
            g0(j) = u(j-1,i) < P10;            
            if g0(j) == 1;                 
                g1(j) = 1; % by monotonicity: if g0(j) is 1, so must g1(j)                
            else P11 = UpdateOrth(g1,j,hats,A);                 
                g1(j) = u(j-1,i) < P11;            
            end
       end % for j = 2:k
    end % for i = 1:m
end % while ~isequal(g0,g1)
samp(v,:) = g0; bct(v) = m;
end %runtime loop
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

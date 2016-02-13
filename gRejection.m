function [samp count] = gRejection(y,X,penalty,shrink,tau,M)

%%%%%%%%%%%%%%%Jason Bentley (2008) University of Canterbury%%%%%%%%%%%%%%%
%Rejection sampler for the conditional distribution of c using Zellner’s  %
%prior. The prior for c is the Hyper-G-n.                                 %
%INPUT: y is the response vector                                          %
%       X is the predictor matrix                                         %
%       tau is the choice of hyper-parameter in the constant Bernoulli    %
%           prior (using tau = 0.5 corresponds to a uniform prior)        %
%       penalty and shrink specify whether Jeffreys prior or Zellner's    %
%           prior is used. For Jeffreys penalty corresponds to p = 2*pi*  %
%           penalty with shrink = 1. For Zellner's prior penalty = (c+1)  %
%           and shrink should be set to c/(c+1).                          %
%       M is the number of samples to be generated                        %
%OUTPUT:samp is N i.i.d samples from the required conditional posterior   %
%       for c.                                                            %
%       count is the N waiting times to generate each sample point        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

format long g; 
[n k1] = size(X); k = k1-1; d = 2^k; A = y'*y; XX = X'*X; BX = (X'*y); N = n/2;
%constants
for i = 1:k1    
Q(i)=(nchoosek(k,(i-1))*((penalty)^-(i/2)))*(tau^(i-1))*((1-tau)^(k-(i-1)));    
end; Q = Q./sum(Q); %calculate proposal density
B = ((y'*y-shrink*y'*X*inv(X'*X)*X'*y)^-N); %compute bound
cn = 0; v = 0;
while cn < M    
    g = zeros(1,k); q = randsample(0:k,1,'true',Q);    
    if q > 0; g(randsample(1:k,q))=1; g = [1 g]; else; g = [1 zeros(1,k)]; end
    %generate a proposal gamma
    RSS = (BX(g==1)'*inv(XX(g==1,g==1))*BX(g==1)); %hat matrix    
    P = ((y'*y-shrink*RSS)^-N)/B;    
    if rand <= P        
        cn = cn + 1       
        samp(cn) = bin2dec(num2str(g(2:k1))); %store decimal        
        %samp(cn,:) = g; %can choose to store g requires more memory        
        count(cn) = v;        
        v = 0;        
    end    
    v = v + 1;    
end

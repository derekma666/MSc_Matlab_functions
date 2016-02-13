function [P Pqg Eqg Map Med Marg yBMA DIC] = Jeffrey(y,X,penalty,tau)

%%%%%%%%%%%%%%%Jason Bentley (2008) University of Canterbury%%%%%%%%%%%%%%%
%Posterior for gamma for Jeffreys prior with the binomial prior for qg.   %
%INPUT: y is the n.1 response vector                                      %
%       X is n.(k+1) design matrix                                        %
%       tau is the choice of hyper-parameter in the constant Bernoulli    %
%           prior (using tau = 0.5 corresponds to a uniform prior)        %
%       penalty should be set p = 2*pi*(c+1) to mimic the penalty of      %
%           Zellner's prior.                                              %
%           and shrink should be set to c/(c+1).                          %
%OUTPUT:P is the normalized posterior probability for eacg gamma          %
%       Pqg is the posterior probability of the model sizes 0:k           %
%       Eqg is the expected model size                                    %
%       Map is maximum aposteriori estimate model                         %
%       Med is the median probability model which incldues all predictors %
%           with MIP > 0.5                                                %
%       Marg are the MIP                                                  %
%       yBMA is the model averaged fitted response                        %
%       DIC is the model averaged DIC                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

format long g; [n k1] = size(X); %constants
k = k1-1; d = 2^k; C2 = log(penalty/(2*pi)); AA = y'*y; %constants
P = zeros(1,d); S = P; Q = P; D = P; AIC = P; VV = P; %storage
for i = 1:d %loop through models    
    g = [1 str2num(dec2bin(i-1,k)')']; %generate a single gamma vector to be evaluated    
    Xg = X(:,g==1);    
    S(i) = sum(g)-1; %the sum of gamma    
    bhat = inv(Xg'*Xg)*Xg'*y;
    Q(i) = log(AA-y'*Xg*bhat); %the log of the quadratic term    
    sighat = exp(Q(i))/(n-2); %the posterior expectation    
    DIC(i) = -2*(sum(log(normpdf(y,Xg*bhat,sqrt(sighat))))) + 2*S(i) + 4; %This is also DIC
end %end loop which has generated two vectors of values one for each quadratic term in
%the posterior and the second for the sum of the gamma vector.
for i = 1:d; B = ((((S(i)-S(j~=i))/2)*C2)+N*(Q(i)-Q(j~=i))); 
    P(i) = ((1 + sum(exp(B)))^-1)*((tau^S(i))*((1-tau)^(k-S(i)))); 
end; clear Q;
P = P./sum(P); %renormalize after adding the proportional prior
for i = 1:d; yhat(:,i) = P(i).*yhat(:,i); end %multiply each models predicted values 
%by the posterior probability
dec = find(P==max(P)); Map = [1 str2num(dec2bin(dec-1,k)')']; %find the MAP
Marg = Margprob(P,k); %Marginal inclusion probabilities this includes 1 for the intercept
Med = [1 Marg>=0.5]; %calculate the median model
for i = 1:k+1; Pqg(i) = sum(P(S==(i-1))); end %Posterior for model size
yBMA = sum(yhat,2); rBMA = y - yBMA; %the sum across rows and model averaged residuals
Eqg = sum(P.*S); clear S
for i = 1:k+1; Pqg(i) = sum(P(S==(i-1))); end
DIC = sum(P.*DIC);

function M = Margprob(P,k)

Mat = zeros(2^k,k); for i = 1:2^k; 
Mat(i,:) = P(i)*[str2num(dec2bin(i-1,k)')']; end; M = sum(Mat);

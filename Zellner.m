function [P Pqg Eqg Map Med Marg yBMA DIC] = Zellner(y,X,c,tau,dic,tol,v)

%%%%%%%%%%%%%%%Jason Bentley (2008) University of Canterbury%%%%%%%%%%%%%%%
%Posterior for gamma for Zellner's prior with the binomial prior for qg.  %
%INPUT: y is the n.1 response vector                                      %
%       X is n.(k+1) design matrix                                        %
%       tau is the choice of hyper-parameter in the constant Bernoulli    %
%           prior (using tau = 0.5 corresponds to a uniform prior)        %
%       c choice of c in Zellner's prior                                  %
%       dic is a ~=1, 1 option 1 = compute dic, ~=1 = do not compute DIC  %
%       tol minimum error (standard deviation) for the simulation estimate%
%           of deviance and DIC                                           %
%       v is the numer of samples to generate in simulation of deviance   %
%           between each check of the simulation error                    %
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

format long g; [n k1] = size(X); k = k1-1; d = 2^k; C1 = c/(c+1); %constants
C2 = log(c+1); N = n/2; I = eye(n); j = 1:d; %constants
P = zeros(1,d); S = P; Q = P; DIC = P; BIC = P; yhat = zeros(n,d); %storage
for i = 1:d %loop through models    
    g = [1 str2num(dec2bin(i-1,k)')']; %generate a single gamma vector
    Xg = X(:,g==1); %predictor matrix adjusted by gamma
    Hg = Xg*inv(Xg'*Xg)*Xg'; %hat matrix 
    bhat = C1*inv(Xg'*Xg)*Xg'*y;
    yhat(:,i) = Xg*bhat;
    S(i) = sum(g)-1; %the sum of gamma
    Q(i) = log(y'*(I-(C1*Hg))*y); %the log of the quadratic term  
    sighat = exp(Q(i))/(n-2); %the posterior expectation    
    VV(i) = -2*(sum(log(normpdf(y,Xg*bhat,sqrt(sighat))))); 
end %end loop which has generated two vectors of values one for each quadratic term in
%the posterior and the second for the sum of the gamma vector.
for i = 1:d; B = ((((S(i)-S(j~=i))/2)*C2)+N*(Q(i)-Q(j~=i))); 
    P(i) = ((1 + sum(exp(B)))^-1)*((tau^S(i))*((1-tau)^(k-S(i)))); 
end; clear Q;
P = P./sum(P); %renormalize after adding the proportional prior
for i = 1:d; yhat(:,i) = P(i).*yhat(:,i); end %multiply each models predicted values by the posterior probability
dec = find(P==max(P)); Map = [1 str2num(dec2bin(dec-1,k)')']; %find the MAP
Marg = Margprob(P,k); %Marginal inclusion probabilities this includes 1 for the intercept
Med = [1 Marg>=0.5]; %calculate the median model
for i = 1:k+1; Pqg(i) = sum(P(S==(i-1))); end %Posterior for model size
yBMA = sum(yhat,2); %the sum across rows and model averaged residuals
Eqg = sum(S.*P); clear S
if dic == 1;[Dev] = PostExpDevZell(y,X,P,v,c,tol); else; dic = 'NA'; end %calculate Deviance
pd = dev-sum(VV.*P); DIC = dev + pd;

function M = Margprob(P,k) 

Mat = zeros(2^k,k); for i = 1:2^k; 
Mat(i,:) = P(i)*[str2num(dec2bin(i-1,k)')']; end; M = sum(Mat);

function [Dev] = PostExpDevZell(y,X,P,n,c,tol)

sd = 10;
[D] = DevSimZell(y,X,P,c);
while sd > tol 
for i = 1:n
[Dev(i)] = DevSimZell(y,X,P,c);
end
D = [D Dev]; m = length(D); 
sd = sqrt((1/m)*((1/m)*sum(D.^2-mean(D)^2)));
end
Deviance = mean(D);

function [Dev] = DevSimZell(y,X,P,c)

[n k1] = size(X); k = k1 - 1; CDF = cumsum(P);
g = PostGamSimX(CDF,k);
Xg = X(:,g==1); bhat = inv(Xg'*Xg)*Xg'*y;  
B = ((y'*y)/2)-((c/(2*(c+1)))*y'*Xg*inv(Xg'*Xg)*Xg'*y); %beta parameter 
sig2sim = 1/gamrnd(n/2,B^-1,1,1); %simulate sigma2 
bhatsim = csmvrnd((c/(c+1))*bhat,((sig2sim*c)/(c+1))*inv(Xg'*Xg),1); 
Dev = -2*(sum(log(normpdf(y,Xg*bhatsim',sqrt(sig2sim)))));

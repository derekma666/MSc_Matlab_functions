function [tails PC tailstats] = ModelCheck(y,X,P,shrink,N,a)

%%%%%%%%%%%%%%%Jason Bentley (2008) University of Canterbury%%%%%%%%%%%%%%%
%Function for checking model adequacy in linear regression using tail     %
%probabilities for the observations and statistics of y (min, max, median,%
%std. dev.) and the predictive coverage.                                  %
%INPUT: y is the n.1 response vector                                      %
%       X is n.(k+1) design matrix                                        %
%       P is the (2^k).1 vector of posterior probabilities.               %
%       shrink specify whether Jeffreys prior or Zellner's prior is used. %
%           For Jeffreys shrink = 1. For Zellner's prior shrink = c/(c+1).%                          
%       N is the number of samples to be generated for each model from the%                       %           PPD to estimate tail prob for statistics of y.                %
%       a is the tail probability for the (a/2), 1-(a/2) interval for     %
%           assessing predictive coverage.
%OUTPUT:tails is an N.1 vector of tail probabilities model averaged for   %
%           each observation.                                             %
%       PC is the modelaaveraged predictive coverage under the PPD.       %
%       tailtstats are the tail probabilities for the min, max, median and%
%           std. dev. of y.                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[n k1] = size(X); k = k1 -1;
for i = 1:length(P)    
    g = [1 str2num(dec2bin(i-1,k)')'];    
    [tprob PCi(i) pval] = PPD(y,X,g,N,shrink,a);    
    tprob1(:,i) = P(i).*tprob;    
    pval1(i,:) = P(i).*pval;    
end
tails = sum(tprob1,2);
tailstats = sum(pval1,1);
PC = sum(PCi.*P)*100;

function [tprob PCi pval] = PPD(y,X,g,N,shrink,a)

format long g; n = length(y); Ig = eye(n); Xg = X(:,g==1);
Hg = shrink*Xg*inv(Xg'*Xg)*Xg'; sig = ((y'*(Ig-Hg)*y)/n)*(Ig + Hg); mu = Hg*y;    
ytrans = (y - mu)./(sqrt(diag(sig))); %transform to a standard t r.v.
PCi = sum(abs(ytrans) <= tinv(1-(a/2),n))/n;
tprob = tcdf(ytrans,n); %from -inf to x so the left hand tail
tprob(find(tprob>=0.5)) = 1-tprob(find(tprob>=0.5)); 
Y = MVTpRnd(n,mu,sig,N); %simulate from the PPD
sumin = min(Y); A = min(y); pval(1) = min(sum(sumin<=A)/N,sum(sumin>=A)/N); 
sumax = max(Y); B = max(y); pval(2) = min(sum(sumax<=B)/N,sum(sumax>=B)/N); 
sumed = median(Y); C = median(y); pval(3) = min(sum(sumed<=C)/N,sum(sumed>=C)/N);   
sumstd = std(Y); D = std(y); pval(4) = min(sum(sumstd<=D)/N,sum(sumstd>=D)/N); 

function Y = MVTpRnd(v,mu,sig,N)

p = length(mu); Y = zeros(p,N); sig1 = zeros(p,p); sig1 = diag(diag(sig));
X = csmvrnd(zeros(p,1),sig1,N); s = sqrt(chi2rnd(v,N)./v);
for i = 1:N; Y(:,i) = (X(i,:)./s(i))'+mu; end

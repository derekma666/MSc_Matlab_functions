function [GPC Low GS GS1] = OrthDesign(Xo,y)

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

[n k] = size(Xo); X = Xo(:,2:k); Int = ones(n,1)./sqrt(n);
for i = 1:k-1; X(:,i) = X(:,i)-mean(X(:,i)); end %centre every predictor

%%%%%Principal Components%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

D = diag(diag(X'*X).^0.5); A = D*(X'*X)*D; [U V] = eig(A); PC1 = X*(D*U); 
%U is eigen vectors, V is eigen values.
for i = 1:(k-1); PC2(:,i)=PC1(:,i)./norm(PC1(:,i)); end; GPC = [Int PC2];

%%%%%SVD to Lowdin%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[M N K] = svd(X,0); SV = [Int M]; %economy size decomposition
ASV = (N*K'); Low = M*K'; Low = [Int Low];

%%%%%Gram-Schmidt%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[GS1] = cgrscho1(y,X); GS1 = [Int GS1]; %method 1
[GS2] = cgrscho2(y,X); GS2 = [Int GS2]; %method 2

function [A] = cgrscho1(y,X)

% Created by A. Trujillo-Ortiz, R. Hernandez-Walls, A. Castro-Perez
%            and K. Barba-Rojo
%            Facultad de Ciencias Marinas
%            Universidad Autonoma de Baja California
%            Apdo. Postal 453
%            Ensenada, Baja California
%            Mexico.
%            atrujo@uabc.mx
% Copyright. September 28, 2006.

[mag pos] = sort(abs(corr(y,X)),'descend'); 
X = X(:,pos); %re-ordering of X based on correlations
A = X; [m n]=size(A);
for j= 1:n
    R(1:j-1,j)=A(:,1:j-1)'*A(:,j);
    A(:,j)=A(:,j)-A(:,1:j-1)*R(1:j-1,j);
    R(j,j)=norm(A(:,j));
    A(:,j)=A(:,j)/R(j,j);
end
return,

function [A] = cgrscho2(y,X)

% Created by A. Trujillo-Ortiz, R. Hernandez-Walls, A. Castro-Perez
%            and K. Barba-Rojo
%            Facultad de Ciencias Marinas
%            Universidad Autonoma de Baja California
%            Apdo. Postal 453
%            Ensenada, Baja California
%            Mexico.
%            atrujo@uabc.mx
% Copyright. September 28, 2006.
% This copyright does not include the sub-routine orderyX
[X order] = orderyX(y,X); X = X(:,order); %re-order X based on correlation %with y and X
A = X; [m n]=size(A);
for j= 1:n
    R(1:j-1,j)=A(:,1:j-1)'*A(:,j);
    A(:,j)=A(:,j)-A(:,1:j-1)*R(1:j-1,j);
    R(j,j)=norm(A(:,j));
    A(:,j)=A(:,j)/R(j,j);
end
return,

function [Xnew order] = orderyX(y,X)

[n k] = size(X); [mag pos] = sort(abs(corr(y,X)),'descend');
Xnew(:,1) = X(:,pos(1)); %the most correlated variable with y
order(1) = pos(1); mag = mag(2:k); pos = pos(2:k); %remove the first predictor chosen
for i = 2:(k-1)
    stage2 = abs(corr(Xnew(:,i-1),X(:,pos))); %the correlation between the most correlated variable with y and repeat this in a loop for k.
    d = sqrt((stage2.^2)+((1-mag).^2)); %vector operations for distances 
    ind = find(d==min(d));
    Xnew(:,i) = X(:,pos(ind)); %make the next predictor that which minimizes the distance to 0,1 or min corr with previous Xnew and max corr with y   
    order(i) = pos(ind);
    pos = setdiff(pos,pos(ind)); %need to update pos vector by removing the most recent added variable   
    mag = setdiff(mag,mag(ind)); 
end
Xnew(:,k) = X(:,pos);
order(:,k) = pos;

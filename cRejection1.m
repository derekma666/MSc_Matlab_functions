function [sampc m par] = cRejection1(y,X,g,a,N)

%%%%%%%%%%%%%%%Jason Bentley (2008) University of Canterbury%%%%%%%%%%%%%%%
%Rejection sampler for the conditional distribution of c using Zellner’s  %
%prior. The prior for c is the Hyper-G-n.                                 %
%INPUT: y is the response vector                                          %
%       X is the predictor matrix                                         %
%       g is the chosen model                                             %
%       a is the hyper-hyper-parameter for the hyper-G-n prior            %
%       N is the number of samples to be generated                        %
%OUTPUT:samp is N i.i.d samples from the required conditional posterior   %
%       for c.                                                            %
%       m is the N waiting times to generate each sample point            %
%       par is the a and b parameters of an I.G. distribution that        %
%         approximates the conditional posterior of c                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n =length(y); v = 2/(a-2); count = 1; C1 = -(sum(g)/2); %constants
C2 = y'*y; Xg = X(:,g==1); C3 = y'*Xg*inv(Xg'*Xg)*Xg'*y; C4 = -n/2; %constants
cm = -(sum(g)*C2-n*C3)/(sum(g)*(C2-C3)); %maximum
bound = ((cm+1)^C1)*((C2-(cm/(cm+1))*C3)^C4); %optimal bound
while count <= N
    steps = 1; %relative to the previous accepted point
    check = 0;
    while check == 0;
        vv = rand;
        prp = -n*(((vv-1)^v)-1)/((vv-1)^v); %propose a new value
        prmove = (C1*log(prp+1)+C4*log(C2-(prp/(prp+1))*C3))-log(bound); %acceptance probability
        if rand <= exp(prmove)
            sampc(count) = prp; %store accepted value
            m(count) = steps; %store length of run for accepted value
            check = 1; %sample point obtained
        end
        steps = steps + 1;
    end
    count = count + 1; %update sample count
end
par = gamfit(1./sampc); %calculate Inverse Gamma Approximation
par = [par(1) 1/par(2)];

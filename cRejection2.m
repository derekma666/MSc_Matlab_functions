function [sampc m] = cRejection2(y,X,g,a,r,s,N)

%%%%%%%%%%%%%%%Jason Bentley (2008) University of Canterbury%%%%%%%%%%%%%%%
%Rejection sampler for the conditional distribution of c using Zellner’s  %
%prior. The prior for c is the Hyper-G-n.                                 %
%INPUT: y is the response vector                                          %
%       X is the predictor matrix                                         %
%       g is the chosen model                                             %
%       a is the hyper-hyper-parameter for the hyper-G-n prior            %
%       r and s are the required parameters for an I.G. approximation to  %
%       the conditional posterior of c                                    %
%       N is the number of samples to be generated                        %
%OUTPUT:samp is N i.i.d samples from the required conditional posterior   %
%       for c.                                                            %
%       m is the N waiting times to generate each sample point            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n =length(y); C1 = -(sum(g)/2); C2 = y'*y; Xg = X(:,g==1); count = 1;
C3 = y'*Xg*inv(Xg'*Xg)*Xg'*y; C4 = -n/2; v = 2/(a-2);
[cmv bb] = solvepolycm(C2,Xg,C3,n,g,a,r,s);
cm = cmv(bb==max(bb))
bound = max(bb) %optimal bound
while count <= N
    steps = 1; %relative to the previous accepted point    
    check = 0;    
    while check == 0;        
        vv = rand;    
        prn = 1/gamrnd(r,1/s); %propose a new value        
        prp = (((prn+1)^(C1))*((C2-(prn/(prn+1))*C3)^(C4))*...
              ((1+(prn/n))^(-a/2)))/((prn^-(r+1))*exp(-s/prn));    
        if rand <= (prp/bound)        
            sampc(count) = prn; %store accepted value            
            m(count) = steps; %store length of run for accepted value            
            check = 1; %sample point obtained        
        end        
        steps = steps + 1;
    end    
    count = count + 1; %update sample count
end

function [cmv bb] = solvepolycm(A,Xg,B,n,g,a,v,w)
%the w must be as the parameter for the IG
q = sum(g)-1;
%these are the polynomial coefficients
r1=-A+B+q*A-q*B+a*A-a*B-2*v*A+2*v*B;
r2=q*A*n+q*A-A*n+2*a*A-a*B-4*v*A+2*v*B+2*w*A-2*w*B+2*B-3*A-q*B*n-...
    2*v*A*n+2*v*B*n;
r3=q*A*n-n*n*B-3*A*n+a*A-2*v*A+4*w*A-2*w*B+2*n*B-2*A-4*v*A*n+2*v*B*...
    n+2*w*A*n-2*w*n*B;
r4=-2*A*n-2*v*A*n+2*w*A-2*w*n*B+4*w*A*n;
r5=2*w*A*n;
cof = [r1 r2 r3 r4 r5]; cm = roots(cof);
for j = 1:length(cm)
    cmr(j) = isreal(cm(j));    
end
cmv1 = cm(cmr==1); cmv = cmv1(cmv1>0);
for i = 1:length(cmv);     
    bb(i) = (((cmv(i)+1)^(-(q+1)/2))*((A-(cmv(i)/(cmv(i)+1))*B)^(-...
n/2))*((1+(cmv(i)/n))^(-a/2)))/((cmv(i)^-(v+1))*exp(-w/cmv(i)));
end
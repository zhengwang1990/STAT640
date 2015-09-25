function [pFinal] = weightedSum(pMean, pKNNUser, KNNUserVar, ...
    pKNNProfile, KNNProfileVar, pSVD, pMFac)
global rmat;
global predInd;
global Npred;

Nprofile = size(rmat,2);

Pnum = sum(rmat~=0,1);
Psum = sum(rmat,1);
Pmeans = full(Psum./Pnum);
[IA, JA, AA]=sparse_to_csr(rmat');
varVec = zeros(Nprofile,1);
for ip = 1:Nprofile    
    x = AA(IA(ip):IA(ip+1)-1);
    x = x - Pmeans(ip);
    varVec(ip) = (x'*x)/Pnum(ip);
end

pFinal = zeros(Npred,1);

for ip = 1:Npred
    % if knn variance is exact 0    
    if ((KNNUserVar(ip) < 1e-5)||(KNNProfileVar(ip) < 1e-5))
        if (KNNUserVar(ip) <= KNNProfileVar(ip))
            pFinal(ip) = pKNNUser(ip);
        else
            pFinal(ip) = pKNNProfile(ip);
        end        
        continue;
    end
    % take weighted sum
    wUser = 1;
    wPro  = 1;
    wSVD  = 0.4;
    wFac = 0;
    switch nargin
        case 5
            wSum = wUser + wPro;
        case 6
            wSum = wUser + wPro + wSVD;
        case 7       
            wSum = wUser + wPro + wSVD + wFac;
    end
    wUser = wUser/wSum;
    wPro  = wPro/wSum;
    wSVD  = wSVD/wSum;
    wFac  = wFac/wSum;
    variance = varVec(predInd(ip,2));
    alpha = variance/25;
    switch nargin
        case 5
            pWeighted = wUser*pKNNUser(ip) + wPro*pKNNProfile(ip);
        case 6
            pWeighted = wUser*pKNNUser(ip) + wPro*pKNNProfile(ip) + wSVD*pSVD(ip);
        case 7
            pWeighted = wUser*pKNNUser(ip) + wPro*pKNNProfile(ip) + wSVD*pSVD(ip) + wFac*pMFac(ip);
    end
    pWeighted = (0-alpha)*pMean(ip) + (1+alpha)*pWeighted;
    pFinal(ip) = min(max(pWeighted,1),10);
end

end
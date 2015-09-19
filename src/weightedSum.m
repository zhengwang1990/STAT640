function pFinal = weightedSum(pMean, pKNNUser, KNNUserVar, pKNNProfile, KNNProfileVar)
global rmat;
global predInd;
global Npred;

Nprofile = size(rmat,2);

Pnum = sum(rmat~=0,1);
Psum = sum(rmat,1);
Pmeans = full(Psum./Pnum);
[IA, JA, AA]=sparse_to_csr(rmat');
var = zeros(Nprofile,1);
for ip = 1:Nprofile    
    x = AA(IA(ip):IA(ip+1)-1);
    x = x - Pmeans(ip);
    var(ip) = (x'*x)/Pnum(ip);
end

pFinal = zeros(Npred,1);
varTol = 1.0;
for ip = 1:Npred
    % if knn variance is small    
    if ((KNNUserVar(ip) < varTol)||(KNNProfileVar(ip) < varTol))
        if (KNNUserVar(ip) <= KNNProfileVar(ip))
            pFinal(ip) = pKNNUser(ip);            
        else
            pFinal(ip) = pKNNProfile(ip);
        end
        continue;
    end
    % take weighted sum
    wKNNUser = exp(-KNNUserVar(ip)^2/200);
    wKNNProfile = exp(-KNNProfileVar(ip)^2/200);
    wSum = wKNNUser + wKNNProfile;
    wKNNUser = wKNNUser/wSum;
    wKNNProfile = wKNNProfile/wSum;
    %[wKNNUser wKNNProfile KNNUserVar(ip) KNNProfileVar(ip)]
    variance = var(predInd(ip,2));
    alpha = variance/30;
    pFinal(ip) = min(max((0-alpha)*pMean(ip) + (1+alpha)*(wKNNUser*pKNNUser(ip) + wKNNProfile*pKNNProfile(ip)),1),10);
end

end
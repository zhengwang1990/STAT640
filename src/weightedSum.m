function pFinal = weightedSum(pMean, pKNN)
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
for ip = 1:Npred
    variance = var(predInd(ip,2));
    alpha = variance/20;
    pFinal(ip) = min(max((0-alpha)*pMean(ip) + (1+alpha)*pKNN(ip),1),10);
end

end
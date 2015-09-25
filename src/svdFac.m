function [u] = svdFac(svdInfo,k)

global predInd;
global Npred;
global U;
global S;
global V;

if (exist(svdInfo, 'file'))
    if (isempty(U)||isempty(V)||isempty(S))
        load(svdInfo);
    end
else
    msg = 'run svdStartup first!';
    error(msg);
end

UTruc = U(:,1:k);
VTruc = V(:,1:k);
STruc = S(1:k,1:k);
UPre = UTruc*sqrt(STruc);
VPre = VTruc*sqrt(STruc);

u = zeros(Npred,1);
for ip = 1:Npred
    user = predInd(ip,1);
    profile = predInd(ip,2);
    u(ip) = UPre(user,:)*VPre(profile,:)';
    if (u(ip) > 10)
        u(ip) = 10;
    end
    if (u(ip) < 1)
        u(ip) = 1;
    end
end

end
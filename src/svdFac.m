function [u] = svdFac(svdInfo,k)

global predInd;
global Npred;
global rmat;
global U;
global S;
global V;

if (exist(svdInfo, 'file'))
    if (isempty(U)||isempty(V)||isempty(S))
        load(svdInfo);
    end
else
    N = 10000; % Nuser & Nprofile
        
    %% avergae for initial guess
    Pnum = sum(rmat~=0,1);
    Psum = sum(rmat,1);
    Pmeans = full(Psum./Pnum)';
    
    %% construct full matrix
    A = zeros(N,N);
    for i = 1:N
        for j = 1:N
            if (rmat(i,j)~=0)
                A(i,j) = rmat(i,j);
            else
                A(i,j) = Pmeans(j);
            end
        end
    end
    [U,S,V] = svd(A);
    % only store first 5000 info
    truc = 5000;
    U = U(:,1:truc);
    V = V(:,1:truc);
    S = S(1:truc,1:truc);
    if (~strcmp(svdInfo,''))
       savd(svdInfo, 'U','S','V');
    end
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
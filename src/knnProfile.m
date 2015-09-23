function [u, varFlag] = knnProfile(K)

global rmat;
global predInd;
global Npred;

[IA, JA, AA]       = sparse_to_csr(rmat'); % CSR storage for fast computation
[IAtr, JAtr, AAtr] = sparse_to_csr(rmat); 
maxN = 10000;
list = zeros(maxN, 2);
u = zeros(Npred,1);
varFlag = zeros(Npred,1);
sk = 1;
for i = 1:Npred    
    % target user and profile
    user = predInd(i,1);
    profile = predInd(i,2);
    list(1:sk-1,:) = 0;
    sk = 1;
    % loop over all users that might have similar taste with the given user
    for j = IAtr(user):IAtr(user+1)-1
        profilej = JAtr(j);
        N = 0; % number of same rate
        dist = 0; % distance
        ip  = IA(profile);
        ipj = IA(profilej);
        % check for mutual rating
        while ((ip < IA(profile+1))&&(ipj < IA(profilej+1)))
            if (JA(ip)==JA(ipj))
                N = N + 1;
                diff = abs(AA(ip) - AA(ipj));
                dist = dist + diff;
                ip = ip + 1;
                ipj = ipj + 1;
            elseif (JA(ip)<JA(ipj))
                ip = ip + 1;
            else
                ipj = ipj + 1;
            end
        end
        if (N==0)
            continue; % no mutual rating
        else
            dist = dist/N;
            score = dist + 2*exp(-N^2/30); % average distance + penallty of small N
            list(sk,:) = [score profilej];
            sk = sk + 1;
        end
    end    
    [X,I] = sort(list(1:sk-1,1),'ascend');
    k = min(K,size(I,1));
    s = full(rmat(user,list(I(1:k),2)))';
    w = 1./(X(1:k).^2+0.5); % weight
    varFlag(i) = var(s);
    wsum = sum(w);
    w = w/wsum;
    u(i) = s'*w;
end
end
function [u, varFlag] = knnUser(K)

global rmat;
global predInd;
global Npred;

[IA, JA, AA]       = sparse_to_csr(rmat); % CSR storage for fast computation
[IAtr, JAtr, AAtr] = sparse_to_csr(rmat'); 
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
    for j = IAtr(profile):IAtr(profile+1)-1
        userj = JAtr(j);
        N = 0; % number of same rate
        dist = 0; % distance
        iu  = IA(user);
        iuj = IA(userj);
        % check for mutual rating
        while ((iu < IA(user+1))&&(iuj < IA(userj+1)))
            if (JA(iu)==JA(iuj))
                N = N + 1;
                diff = abs(AA(iu) - AA(iuj));
                dist = dist + diff;
                iu = iu + 1;
                iuj = iuj + 1;
            elseif (JA(iu)<JA(iuj))
                iu = iu + 1;
            else
                iuj = iuj + 1;
            end
        end
        if (N==0)
            continue; % no mutual rating
        else
            dist = dist/N;
            score = dist + 2*exp(-N^2/30); % average distance + penallty of small N
            list(sk,:) = [score userj];
            sk = sk + 1;
        end
    end    
    [X,I] = sort(list(1:sk-1,1),'ascend');
    k = min(K,size(I,1));
    s = full(rmat(list(I(1:k),2),profile));
    w = 1./(X(1:k).^2+1); % weight
    varFlag(i) = var(s);
    wsum = sum(w);
    w = w/wsum;
    u(i) = s'*w;
end

end
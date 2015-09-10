function u = knn(K)

global rmat;
global predInd;
global Npred;

Nuser = size(rmat, 1);
[IA, JA, AA]=sparse_to_csr(rmat); % CSR storage for fast computation

for i = 1:Npred
    list = [];
    % target user and profile
    user = predInd(i,1);
    profile = predInd(i,2);
    % loop over all users that might have similar taste with the given user
    for userj = 1:Nuser        
        if (rmat(userj,profile)~=0)
            N = 0; % number of same rate
            dist = 0; % distance            
            iu  = IA(user);
            iuj = IA(userj);
            while ((iu < IA(user+1))&&(iuj < IA(userj+1)))
                if (JA(iu)==JA(iuj))
                    N = N + 1;
                    dist = dist + abs(AA(iu) - AA(iuj));
                    iu = iu + 1;
                    iuj = iuj + 1;
                elseif (JA(iu)<JA(iuj))
                        iu = iu + 1;
                else
                    iuj = iuj + 1;
                end                                        
            end               
            if (N==0)
                continue;
            else
                dist = dist/N;
                score = dist + 2*exp(-N^2/30); %average distance + penallty of small
                list = [list; [score userj]];
            end
        end
    end
    [X,I] = sort(list(:,1),'ascend');
    k = min(K,size(list,1));
    s = full(rmat(list(I(1:k),2),profile));
    w = 1./(X(1:k).^2+0.5);
    wsum = sum(w);
    w = w/wsum;
    u(i) = s'*w;
end
u = u';
end
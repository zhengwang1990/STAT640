function [u, trnErr] = matrixFac(userData, profileData)

global rmat;
global predInd;
global Npred;

useMean = true;

if (exist(userData, 'file')&&exist(profileData, 'file'))
    load(userData);
    load(profileData);
else
    msg = 'Error, Data files not found! Run MatrixFacStartup to generate data!\n';
    error(msg);
end

u = zeros(Npred,1);
for ip = 1:Npred
    user = predInd(ip,1);
    profile = predInd(ip,2);
    u(ip) = P(:,user)'*Q(:,profile);
end

if (useMean)
    Pnum = sum(rmat~=0,1);
    Psum = sum(rmat,1);
    PMean = full(Psum./Pnum)';
    u = u + PMean(predInd(:,2));
end

for ip = 1:Npred
   if (u(ip) > 10)
       u(ip) = 10;
   end
   if (u(ip) < 1)
       u(ip) = 1;
   end
end

[uid, pid, r] = find(rmat);
[Nuser, Nprofile] = size(rmat);
Nrate = nnz(rmat);
uError = zeros(Nuser,1);
Nu = zeros(Nuser,1);
pError = zeros(Nprofile,1);
Np = zeros(Nprofile,1);
for ir = 1:Nrate
    user = uid(ir);
    profile = pid(ir);
    rate = r(ir);
    p = P(:,user);
    q = Q(:,profile);
    eval_rate = p'*q;
    if (useMean)
        eval_rate = eval_rate + PMean(profile);
    end
    err = abs(rate-eval_rate);
    uError(user) = uError(user) + err;
    pError(profile) = pError(profile) + err;
    Nu(user) = Nu(user) + 1;
    Np(profile) = Np(profile) + 1;
end
uCond = uError./Nu;
pCond = pError./Np;

trnErr = zeros(Npred,1);
for ip = 1:Npred
    trnErr(ip) = uCond(predInd(ip,1))*pCond(predInd(ip,2));
end

end

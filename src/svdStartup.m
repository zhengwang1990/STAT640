function svdStartup(svdInfo)
global predInd;
global Npred;
global trainInd;
global output;

N = 10000; % Nuser & Nprofile

%% read data
global ratings;
if (isempty(trainInd))
    trainInd = 1:2:size(ratings,1);
end
rmat = sparse(ratings(trainInd,1), ratings(trainInd,2), ratings(trainInd,3), 10000, 10000);

if (output)
    %knnUserInfo = '../data/knnUserInfo.mat';
    %knnProfileInfo = '../data/knnProfileInfo.mat';
    knnUserInfo = '../data/knnUserInfoTrain.mat';
    knnProfileInfo = '../data/knnProfileInfoTrain.mat';
else
    knnUserInfo = '../data/temp/knnUserInfo.mat';
    knnProfileInfo = '../data/temp/knnProfileInfo.mat';
end
correction = false;
if (exist(knnUserInfo, 'file') && exist(knnProfileInfo, 'file'))
    load(knnUserInfo);
    load(knnProfileInfo);
    pMean = benchmark();
    pCorrect = weightedSum(pMean, pKNNUser, KNNUserVar, ...
        pKNNProfile, KNNProfileVar);
    correction = true;
else
    fprintf('Warning: running svdStartup without correction!\n');
end

%% avergae for initial guess
Pnum = sum(rmat~=0,1);
Psum = sum(rmat,1);
Pmeans = full(Psum./Pnum)';

%% construct full matrix
A = zeros(N,N);
fprintf('Generate full matrix!\n');
Nrate = nnz(rmat);
[user, pro, r] = find(rmat);
for j = 1:N
    A(:,j) = Pmeans(j);
end
for ir = 1:Nrate
   A(user(ir),pro(ir)) = r(ir);
end

if (correction)
    fprintf('Add correction!\n');
    for ip = 1:Npred
        i = predInd(ip,1);
        j = predInd(ip,2);
        A(i,j) = pCorrect(ip);
    end
end

fprintf('Start SVD decomposition!\n');
[U,S,V] = svd(A);

%% save
save(svdInfo, 'U', 'S', 'V');
end
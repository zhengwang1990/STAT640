N = 10000; % Nuser & Nprofile

%% read data
global ratings;
if (isempty(ratings))
    ratings = csvread('../data/ratings.csv',1,0);
end
trainInd = 1:2:size(ratings,1);
rmat = sparse(ratings(trainInd,1), ratings(trainInd,2), ratings(trainInd,3), 10000, 10000);

%% avergae for initial guess
Pnum = sum(rmat~=0,1);
Psum = sum(rmat,1);
Pmeans = full(Psum./Pnum)';

%% construct full matrix
A = zeros(N,N);
fprintf('Generate full matrix!\n');
for i = 1:N
    if (mod(i,100)==0)
        fprintf('i = %d\n',i);
    end
    for j = 1:N
        if (rmat(i,j)~=0)
            A(i,j) = rmat(i,j);
        else
            A(i,j) = Pmeans(j);
        end
    end
end

fprintf('Start SVD decomposition!\n');
[U,S,V] = svd(A);

%% save
filename = '../data/svdInfo.mat';
save(filename, 'U', 'S', 'V');

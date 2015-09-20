% minimize (r_{ui}-p_u*q_i)^2 + lambda (p_u^2+q_i^2)
%
k = 10; % number of features
%% read data
global ratings;
if (isempty(ratings))
    ratings = csvread('../data/ratings.csv',1,0);
end
trainInd = 1:size(ratings,1);
rmat = sparse(ratings(trainInd,1), ratings(trainInd,2), ratings(trainInd,3), 10000, 10000);
[Nuser, Nprofile] = size(rmat);

%% initialize P and Q
% use random numbers as initial guesses for now. should be able to improve.
rng('default');
P = randn(k,Nuser);
Q = randn(k,Nprofile);
Nrate = nnz(rmat);
% substract average. Be careful to add it back when predict!
Pnum = sum(rmat~=0,1);
Psum = sum(rmat,1);
Pmeans = full(Psum./Pnum)';
[u, i, r] = find(rmat);
r(:) = r(:) - Pmeans(i(:));
gradP = zeros(k,Nuser);
gradQ = zeros(k,Nprofile);


%% regularization parameter
lambda = 0.05;

%% verbose
fprintf('Running Matrix Decompostion.\n');
fprintf('Number of Features = %d\n', k);
fprintf('Number of Training Data = %d\n', Nrate);

%% initial loss function value
object = 0;
for ir = 1:Nrate
    user = u(ir);
    profile = i(ir);
    rate = r(ir);
    object = object + (-rate+P(:,user)'*Q(:,profile))^2;
end
fprintf('iter = 0, loss function = %f\n', object);

%% Steepest Decent Iteration
MaxIter = 10000;
gradTol = 1.0;
for iter = 1:MaxIter
    gradP(:,:) = 0;
    gradQ(:,:) = 0;
    % gradient from loss function
    for ir = 1:Nrate
        user = u(ir);
        profile = i(ir);
        rate = r(ir);
        gradP(:,user) = gradP(:,user) + (-rate+P(:,user)'*Q(:,profile))*Q(:,profile);
        gradQ(:,profile) = gradQ(:,profile) + (-rate+P(:,user)'*Q(:,profile))*P(:,user);
    end    
    % gradient from regulization
    gradP(:,:) = gradP(:,:) + lambda*P(:,:);
    gradQ(:,:) = gradQ(:,:) + lambda*Q(:,:);
    
    if (iter > 1)
        tau = (sum(sum((gradP-gradP_old).*dP)) + sum(sum((gradQ-gradQ_old).*dQ)))/...
            (sum(sum((gradP-gradP_old).^2)) + sum(sum((gradQ-gradQ_old).^2)));
    else
        tau = 0.0001;
    end
    gradP_old = gradP;
    gradQ_old = gradQ;
    dP = -tau*gradP;
    dQ = -tau*gradQ;
    P = P - tau*gradP;
    Q = Q - tau*gradQ;
    % check loss function
    object = 0;
    for ir = 1:Nrate
        user = u(ir);
        profile = i(ir);
        rate = r(ir);
        object = object + (-rate+P(:,user)'*Q(:,profile))^2;
    end
    gradNorm = norm(gradP)+norm(gradQ);
    fprintf('iter = %d, loss function = %f, gradient norm = %f\n',iter, object, gradNorm);
    if (gradNorm < gradTol)
        break;
    end
end
fprintf('Finish at iteration %d. Gradient = %f\n', iter, gradNorm);

%% save data
filename_u = '../data/userVec.mat';
save(filename_u,'P');
fprintf('User   vectors saved in %s\n', filename_u);
filename_p = '../data/profileVec.mat';
save(filename_p,'Q');
fprintf('Profile vectors saved in %s\n', filename_p);




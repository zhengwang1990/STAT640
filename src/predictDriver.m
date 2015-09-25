
global ratings;
global rmat;
global idmap;
global genders;
global Npred;
global predInd;
global trainInd;
global output;

%% output
% output = true for submission
% output = false for training and test
output = true;

%% read data
if (isempty(ratings))
    ratings = csvread('../data/ratings.csv',1,0);
end
if (isempty(idmap))
    idmap   = csvread('../data/IDMap.csv',1,0);
end
if (isempty(genders))
    genders = csvread('../data/gender.csv',1,0);
end

%% divide into training and test
if (output)
    trainInd = 1:100:size(ratings,1);
else
    trainInd = 1:3:size(ratings,1);
    testInd = 112:1000:size(ratings,1);    
end
rmat = sparse(ratings(trainInd,1), ratings(trainInd,2), ratings(trainInd,3), 10000, 10000);

%% output array
if (output)
    outSz = size(idmap,1);
    segment = 1:outSz;
    %predInd = idmap(segment,:);
    predInd = [ratings(trainInd,1), ratings(trainInd,2), trainInd'];
    pExac = ratings(trainInd,3);
else
    predInd = [ratings(testInd,1) ratings(testInd,2)];
    pExac = ratings(testInd,3);
end
Npred = size(predInd,1);


%% benchmark -- take mean
disp('TFBoys: Prepare BenchMark Solution');
tic();
pMean = benchmark();
toc();

%% knn
disp('TFBoys: Prepare KNN Solution');
disp('1) KNN User');
tic();
if (output)
    %knnUserInfo = '../data/knnUserInfo.mat';
    knnUserInfo = '../data/knnUserInfoTrain.mat';    
else
    knnUserInfo = '../data/temp/knnUserInfo.mat';
end
if (exist(knnUserInfo, 'file'))
    load(knnUserInfo);
else
    [pKNNUser, KNNUserVar] = knnUser(25);
    save(knnUserInfo, 'pKNNUser', 'KNNUserVar');
end
toc();
disp('2) KNN Profile');
tic();
if (output)
    %knnProfileInfo = '../data/knnProfileInfo.mat';
    knnProfileInfo = '../data/knnProfileInfoTrain.mat';
else
    knnProfileInfo = '../data/temp/knnProfileInfo.mat';
end
if (exist(knnProfileInfo, 'file'))
    load(knnProfileInfo);
else
    [pKNNProfile, KNNProfileVar] = knnProfile(15);
    save(knnProfileInfo, 'pKNNProfile', 'KNNProfileVar');
end
toc();

%% svd
% disp('TFBoys: Prepare SVD Solution');
% tic();
% if (output)
%     %svdInfo = '../data/svdInfo.mat';
%     svdInfo = '../data/svdInfoTrain.mat';
% else
%     svdInfo = '../data/temp/svdInfo.mat';
% end
% if (~exist(svdInfo,'file'))
%     svdStartup(svdInfo);
% end
% [pSVD] = svdFac(svdInfo, 150);
% toc();


%% mFac
%userData = '../data/userVecTest.mat';
%profileData = '../data/profileVecTest.mat';
%[pMFac, facErr] = matrixFac(userData, profileData);

%% weighted sum
[pFinal] = weightedSum(pMean, pKNNUser, KNNUserVar, ...
    pKNNProfile, KNNProfileVar);

%% output
if (output)    
    filename = '../outputs/trainning.csv';
    fileId = fopen(filename,'w');
    fprintf(fileId,'ID,Prediction\n');
    for i = 1:Npred
        fprintf(fileId,'%d,%.15f\n',predInd(i,3),pFinal(i));
    end
    fclose(fileId);
    fprintf('Output written in %s\n',filename);
else
    errFin = norm(pFinal-pExac)^2;
    errBmk = norm(pMean-pExac)^2;
    errUsr = norm(pKNNUser-pExac)^2;
    errPro = norm(pKNNProfile-pExac)^2;
    errSVD = norm(pSVD-pExac)^2;     
    %errFac = norm(pMFac-pExac)^2;      
    fprintf('Num of Tests  = %d\n', Npred);    
    fprintf('knnU  Err Total = %9.5f   RMSE = %f\n', errUsr, sqrt(errUsr/Npred));
    fprintf('knnP  Err Total = %9.5f   RMSE = %f\n', errPro, sqrt(errPro/Npred));
    fprintf('SVD   Err Total = %9.5f   RMSE = %f\n', errSVD, sqrt(errSVD/Npred));
    %fprintf('MFac  Err Total = %9.5f   RMSE = %f\n', errFac, errFac/Npred);
    fprintf('Final Err Total = %9.5f   RMSE = %f\n', errFin, sqrt(errFin/Npred));
    fprintf('Bmk   Err Total = %9.5f   RMSE = %f\n', errBmk, sqrt(errBmk/Npred));
end
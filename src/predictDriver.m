
global ratings;
global rmat;
global idmap;
global genders;
global Npred;
global predInd;
global trainInd;

%% output
% output = true for submission
% output = false for training and test
output = false;

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
    trainInd = 1:size(ratings,1);
else
    trainInd = 1:2:size(ratings,1);
    testInd = 14250:50000:size(ratings,1);    
end
rmat = sparse(ratings(trainInd,1), ratings(trainInd,2), ratings(trainInd,3), 10000, 10000);

%% output array
if (output)
    outSz = size(idmap,1);
    segment = 1:100;%outSz;
    predInd = idmap(segment,:);
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
    knnUserInfo = '../data/knnUserInfo.mat';
    if (exist(knnUserInfo, 'file'))
        load(knnUserInfo);
    else
        [pKNNUser, KNNUserVar] = knnUser(25);
        save(knnUserInfo, 'pKNNUser', 'KNNUserVar');
    end
else
    [pKNNUser, KNNUserVar] = knnUser(25);
end
toc();
disp('2) KNN Profile');
tic();
if (output)
    knnProfileInfo = '../data/knnProfileInfo.mat';
    if (exist(knnProfileInfo, 'file'))
        load(knnProfileInfo)
    else
        [pKNNProfile, KNNProfileVar] = knnProfile(15);
        save(knnProfileInfo, 'pKNNProfile', 'KNNProfileVar');
    end
else
    [pKNNProfile, KNNProfileVar] = knnProfile(15);
end
toc();

%% svd
if (output)
    svdInfo = '/Users/zw14/Documents/svdInfo.mat';
else
    svdInfo = '/Users/zw14/Documents/svdInfoTest.mat';
end
[pSVD] = svdFac(svdInfo, 100);


%% weighted sum
[pFinal] = weightedSum(pMean, pKNNUser, KNNUserVar, ...
    pKNNProfile, KNNProfileVar, pSVD);

%% output
if (output)    
    filename = '../outputs/TFBoys1.0.csv';
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
    errKnn = norm(pKNN-pExac)^2;
    errSVD = norm(pSVD-pExac)^2;      
    fprintf('Num of Tests  = %d\n', Npred);    
    fprintf('knnU  Err Total = %9.5f   Error Per Entry = %f\n', errUsr, errUsr/Npred);
    fprintf('knnP  Err Total = %9.5f   Error Per Entry = %f\n', errPro, errPro/Npred);
    fprintf('SVD   Err Total = %9.5f   Error Per Entry = %f\n', errSVD, errSVD/Npred);
    fprintf('Final Err Total = %9.5f   Error Per Entry = %f\n', errFin, errFin/Npred);
    fprintf('Bmk   Err Total = %9.5f   Error Per Entry = %f\n', errBmk, errBmk/Npred);
end
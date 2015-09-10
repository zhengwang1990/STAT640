
global ratings;
global rmat;
global idmap;
global genders;
global Npred;
global predInd;

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
    trainInd = 1:size(ratings,1);
else
    trainInd = 1:2:size(ratings,1);
    testInd = 10:100000:size(ratings,1);
end
rmat = sparse(ratings(trainInd,1), ratings(trainInd,2), ratings(trainInd,3), 10000, 10000);

%% output array
if (output)
    predInd = idmap(:,1:2);
else
    predInd = [ratings(testInd,1) ratings(testInd,2)];
    pExac = ratings(testInd,3);
end
Npred = size(predInd,1);


%% benchmark -- take mean
pMean = benchmark();
wMean = 0.3;

%% knn
pKNN = knn(40);
wKNN = 0.7;

%% weighted sum
pFinal = wMean*pMean + wKNN*pKNN;

%% output
if (output)    
    filename = '../outputs/TFBoys1.0.csv';
    fileId = fopen(filename,'w');
    fprintf(fileId,'ID,Prediction\n');
    for i = 1:Npred
        fprintf(fileId,'%d,%.15f\n',i,pFinal(i));
    end
    fclose(fileId);
else
    errTot = norm(pFinal-pExac);
    errBmk = norm(pMean-pExac);
    fprintf('Error Total   = %9.5f   Error Per Entry = %f\n', errTot, errTot/Npred);
    fprintf('Bmk Err Total = %9.5f   Error Per Entry = %f\n', errBmk, errBmk/Npred);
end
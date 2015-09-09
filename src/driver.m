%% read data
global ratings;
global rmat;
global idmap;
global genders;
if (isempty(ratings))
    ratings = csvread('../data/ratings.csv',1,0);
end
if (isempty(idmap))
    idmap   = csvread('../data/IDMap.csv',1,0);
end
if (isempty(genders))
    genders = csvread('../data/gender.csv',1,0);
end
rmat = sparse(ratings(:,1), ratings(:,2), ratings(:,3));

%% output array
Npred = size(idmap,1);
pFinal = zeros(Npred,2);
pFinal(:,1) = idmap(:,3);

%% benchmark -- take mean
pMean = means();
wMean = 1.0;

%% weighted sum
pFinal(:,2) = wMean*pMean;

%% output
output = true;
if (output)    
    filename = '../outputs/TFBoys1.0.csv';
    fileId = fopen(filename,'w');
    fprintf(fileId,'ID,Prediction\n');
    for i = 1:Npred
        fprintf(fileId,'%d,%.15f\n',pFinal(i,1),pFinal(i,2));
    end
    fclose(fileId);
end
function [X,X_tst,idx_trn,idx_tst] = getdcm(prc_trn)
 path='../../data/Knee (R)/AX.  FSE PD - 5/';

extension = strcat(path,'*.dcm');
files = dir(extension);
X=[];

minPixels = repmat(0, [1, size(files,1)]);
maxPixels = repmat(0, [1, size(files,1)]);
p=0;
for file = files'
    p=p+1;
    info = dicominfo(strcat(path,file.name));
    minPixels(p) = info.SmallestImagePixelValue;
    maxPixels(p) = info.LargestImagePixelValue;
    X = cat(2,X, reshape(dicomread(strcat(path,file.name)),[],1));
end

X=double(X);
X = X./mean(mean(mean(X)));


idx_trn=[];
idx_tst=[];
X_tst = zeros(size(X));

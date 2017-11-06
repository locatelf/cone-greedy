function [X,X_tst,idx_trn,idx_tst] = getdcm(prc_trn)
 path='data/KNIX/Knee (R)/AX.  FSE PD - 5/';
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
    X = cat(3,X, dicomread(strcat(path,file.name)));
end
b = min(minPixels);
X=double(X);
if size(X)== 0
    fprintf('wrong path in getdcm.m')
    return
end
X = X./mean(mean(mean(X)));



idx_trn=[];
idx_tst=[];
X_tst = zeros(size(X));

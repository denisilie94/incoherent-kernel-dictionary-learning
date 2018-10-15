function [ random_features ] = extract_features( img, IMG_H, IMG_W, dims)
%EXTRACTFEATURES Summary of this function goes here
%   Detailed explanation goes here

% generate the random matrix
randmatrix = randn(dims,IMG_H*IMG_W);
l2norms = sqrt(sum(randmatrix.*randmatrix,2)+eps);
randmatrix = randmatrix./repmat(l2norms,1,size(randmatrix,2));


feature = double(img(:));
random_features = randmatrix*feature;

end


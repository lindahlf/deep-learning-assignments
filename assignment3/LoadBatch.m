function [X, Y, y] = LoadBatch(filename)
% Returns X,Y and y
% X is the image pixel data with size dxn, where d is the dimension of each image (e.g.32x32x3) 
% n = number of images
% Y = Kxn where K is the one-hot representation of the label of each image
% y = nx1 vector of the label of each image

A = load(filename);
X = double(A.data)';
y = 1 + double(A.labels); 
K = max(y);
Y = full(ind2vec(double(y'),K)); % One-hot representation of labels

end


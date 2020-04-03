function [P,H] = EvalClassifier(X, W, b)
% Each column of X corresponds to an image and has size dxn where d is the
% dimension of the image and n is the number of images
% W1, W2, b1 and b2 are the parameters of the network stored in cells.
% Returns P which contains the probability for each label for the image in
% the corresponding column of X. Has size Kxn

W1 = W{1}; W2 = W{2};
b1 = b{1}; b2 = b{2};

s1 = W1*X + b1; 
H = max(0,s1);
s = W2*H + b2;
exponent = exp(s);
P = exponent./sum(exponent,1);
end


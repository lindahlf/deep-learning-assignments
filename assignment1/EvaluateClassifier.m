function P = EvaluateClassifier(X, W, b)
% Each column of X corresponds to an image and has size dxn where d is the
% dimension of the image and n is the number of images
% W and b are the parameters of the network
% Returns P which contains the probability for each label for the image in
% the corresponding column of X. Has size Kxn

s = W*X + b;
exponent = exp(s);
P = exponent./sum(exponent,1);

end


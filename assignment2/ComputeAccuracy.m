function acc = ComputeAccuracy(X, y, W, b)
% Each column of X corresponds to an image, X is dxn
% y is the vector of ground truth values of length n
% Returns the scalar value acc containing the accuracy

P = EvalClassifier(X,W,b);
% Recall that y contains the true label of the data

[~,I] = max(P,[],1);
match = I == y';

acc = sum(match)/length(y);

end


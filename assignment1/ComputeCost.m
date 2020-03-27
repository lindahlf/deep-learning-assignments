function J = ComputeCost(X, Y, W, b, lambda)
% Size dxn where each column of X corresponds to an image
% Each column of Y is either the Kxn one-hot ground truth label for
% corresponding column of X or is simply the 1xn vector of ground truth
% labels.
% Returns a scalar J corresponding the sum of the loss of the network's
% predictions for the images in X relative to the ground truth labels and
% the regularization term on W 


P = EvaluateClassifier(X,W,b);

[~,n] = size(X);

if ismember(0,Y) == 0 % 1xn 
    idx = sub2ind(size(P), Y, (1:1:n)');
    J = 1/n*sum(-log(P(idx))) + lambda*norm(W,'fro')^2;
else % One-hot ground truth label
    J = 1/n*sum(-log(sum(Y.*P))) + lambda*norm(W,'fro')^2;  
end


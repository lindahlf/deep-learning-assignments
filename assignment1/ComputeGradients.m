function [grad_b,grad_W] = ComputeGradients(X, Y, P, W, lambda)
% Each column of X corresponds to an image and it has size dxn
% Each column of Y (K×n) is the one-hot ground truth label for the 
% corresponding column of X
% Each column of P contains the probability for each label for the image in
% the corresponding column of X. P has size Kxn
% Returns grad_W and grad_b which are the gradient matrix of the cost J
% relative to W and b respectively. They have size Kxd and Kx1
% respectively.


[~,n] = size(X);
G = -(Y-P);

dLW = (1/n)*(G*X');
dLb = (1/n)*(G*ones(n,1));

grad_W = dLW + 2*lambda*W;
grad_b = dLb;
end


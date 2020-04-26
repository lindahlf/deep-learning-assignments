function [J,loss] = ComputeCost(X, Y, Params, lambda)
% Computes cost and loss function given data 
% Inputs: 
% X = images 
% Y = labels of images X 
% Params = struct of data with 
%   W = cell of weight matrices 
%   b = cell of bias vectors 
%   gammas = cell of stretch vectors
%   beta = cell of shift vectors
%   use_bn = logical indicating whether to use batch norm. or not
% lambda = regularization
% 
% Returns
% J = value of cost function 
% loss = value of loss function 

W = Params.W; b = Params.b;

if Params.use_bn
    [P,~] = EvalClassifier(X,W,b,Params.gammas,Params.betas);
else
    [P,~] = EvalClassifier(X,W,b);  
end

[~,n] = size(X);

wSum = 0;
for l = 1:length(W)
    wSum = wSum + norm(W{l},'fro')^2;
end

if ismember(0,Y) == 0 % 1xn 
    idx = sub2ind(size(P), Y, (1:1:n)');
    loss = 1/n*sum(-log(P(idx)));
    J = loss + lambda*wSum;
else % One-hot ground truth label
    loss = 1/n*sum(-log(sum(Y.*P)));
    J = loss + lambda*wSum;
end

end
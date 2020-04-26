function acc = ComputeAccuracy(X, y, W, b, varargin)
% Inputs:
% X = matrix of images
% y = matrix of ground truth values of labels 
% W = cell of weight matrices for each layer
% b = cell of bias vectors for each layer
% (optional) gamma = cell of scale vectors for each layer
% (optional) beta = cell of shift vectors for each layer
%
% Returns:
% acc = scalar value of accuracy of data X using given parameters

if ~isempty(varargin)
    gamma = varargin{1};
    beta = varargin{2}; 
    [P,~] = EvalClassifier(X,W,b,gamma,beta);
else
    [P,~] = EvalClassifier(X,W,b);
end

[~,I] = max(P,[],1); % Finding all matching labels
match = I == y';
acc = sum(match)/length(y);

end


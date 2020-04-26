function [P,H,varargout] = EvalClassifier(X,W,b,varargin)
% Computes probability matrix of given classifier
% Inputs: 
% X = images 
% W = cell of weight matrices for each layer
% b = cell of bias vectors for each layer
% (optional) gamma = cell of stretch vectors for each layer
% (optional) beta = cell of shift vectors for each layer
% 
% Returns: 
% P = probability matrix
% H = cell of X_batch at each layer
% (optional) mu = cell of means for each layer of unnormalized scores
% (optional) v = cell of variances for each layer of unnormalized scores
% (optional) s = cell of scores for each layer
% (optional) shat = cell of normalized scores for each layer

k = length(W);
H = cell([1,k]);

if ~isempty(varargin)
    gamma = varargin{1};
    beta = varargin{2};
    
    mu = cell([1,k]);
    v = cell([1,k]);
    s = cell([1,k]);
    shat = cell([1,k]);
    
    bn = true;
else
    bn = false;
end


for l = 1:k    
    if bn

        s{l} = W{l}*X + b{l};
        mu{l} = mean(s{l},2);
        v{l} = mean((s{l}-mu{l}).^2 ,2);
        shat{l} = diag(1./(sqrt(v{l}+eps)))*(s{l}-mu{l});
        stilde = gamma{l}.*shat{l} + beta{l};
        
        X = max(0,stilde);
        
        H{l} = X;
        P = softmax(s{l});
    else
        s = W{l}*X + b{l};
        X = max(0,s);
        H{l} = X;
        P = softmax(s);

    end
end

if bn
    varargout{1} = mu;
    varargout{2} = v;
    varargout{3} = s;
    varargout{4} = shat; 
end


end


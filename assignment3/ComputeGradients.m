function [gradb, gradW, varargout] = ComputeGradients(X,Y,W,b,lambda,varargin)
% Computes gradients for the weights, bias, stretch and shift using forward and
% backward propagation
% Inputs:
% X = data 
% Y = labels of data
% W = cell of weight matrices for each layer
% b = cell of bias vectors for each layer
% lambda = regularization
% (optional) gamma = cell of stretch vectors for each layer
% (optional) beta = cell of shift vectors for each layer
%
% Returns:
% gradb = cell of gradients for bias vectors for each layer
% gradW = cell of gradients for weight matrices for each layer
% (optional) gradgamma = cell of gradients for stretch vectors for each layer
% (optional) gradbeta = cell of gradients for shift vectors for each layer

if ~isempty(varargin)
    bn = true;
    
    gamma = varargin{1};
    beta = varargin{2};
    

    gradgamma = cell([1,length(gamma)-1]);
    gradbeta = cell([1,length(beta)-1]);
    
    
    [P,H,mu,v,s,shat] = EvalClassifier(X, W, b, gamma, beta);

else
    bn = false;
    [P,H] = EvalClassifier(X, W, b);

end


gradb = cell([1,length(b)]);
gradW = cell([1,length(W)]);

[~,n] = size(Y);

k = length(W);

G = -(Y-P);

if bn
    gradW{k} = (1/n)*G*H{k-1}' + 2*lambda*W{k};
    gradb{k} = (1/n)*G*ones(n,1);
    
    G = W{k}'*G;
    Ind = H{k-1} > 0;
    G = G.*Ind;
    
    for l = k-1:-1:1
        gradgamma{l} = (1/n)*(G.*shat{l})*ones(n,1);
        gradbeta{l} = (1/n)*G*ones(n,1);
        
        G = G.*(gamma{l}*ones(1,n));
        
        G = BatchNormBackPass(G,s{l},mu{l},v{l},n);
        
        if l > 1
            gradW{l} = (1/n)*G*H{l-1}' + 2*lambda*W{l};
            gradb{l} = (1/n)*G*ones(n,1);
            
            G = W{l}'*G;
            Ind = H{l-1} > 0;
            G = G.*Ind;
        else
            gradW{1} = (1/n)*G*X' + 2*lambda*W{1};
            gradb{1} = (1/n)*G*ones(n,1);
 
        end
    end
else
   
    for l = k:-1:2
        gradW{l} = (1/n)*G*H{l-1}' + 2*lambda*W{l};
        gradb{l} = (1/n)*G*ones(n,1);

        G = W{l}'*G;
        Ind = H{l-1} > 0;
        G = G.*Ind;
    end
    
    gradW{1} = (1/n)*G*X' + 2*lambda*W{1};
    gradb{1} = (1/n)*G*ones(n,1);
    
end

if bn
    varargout{1} = gradgamma;
    varargout{2} = gradbeta;
end

end


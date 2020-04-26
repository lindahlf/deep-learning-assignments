function [W,b,gamma,beta] = initParams(k, m, n, d, bias, varargin)
% Initializing the parameters for the network
% Parameters:
% k = number of layers
% m = vector with number of nodes in each layer 
% n = number of images
% d = dimension of the images
% bias = 1 or 0. 0 = no bias, 1 = bias.
% (optional) sig = standard deviation of gaussian initialization at every
% level

% Returns:
% Cells W, b, gamma, beta

W = cell([1,k]); % Initialize cell for weight matrices
b = cell([1,k]); % Initialize cell for bias vectors
gamma = cell([1,k]); % Initialize cell for scale matrices
beta = cell([1,k]); % Initialize cell for shift vectors

gamma{1} = ones(m(1),1);
beta{1} = zeros(m(1),1);


if ~isempty(varargin) % same sigma at each layer
    sig = varargin{1};
    W{1} = sig.*randn(m(1),d);
    if bias == 1
        b{1} = sig.*randn(m(1),1);
    else
        b{1} = zeros(m(1),1);
    end
    
    for i = 2:k
        W{i} = sig.*randn(m(i),m(i-1));
        gamma{i} = ones(m(i),1);
        beta{i} = zeros(m(i),1);
        if bias == 1
            b{i} = sig.*randn(m(i),1);
        else
            b{i} = zeros(m(i),1);
        end     
    end
    
else % He initialization
    
    W{1} = sqrt(2/d).*randn(m(1),d);

    if bias == 1
        b{1} = sqrt(2/d).*randn(m(1),1);
    else
        b{1} = zeros(m(1),1);
    end

    for i = 2:k
        W{i} = sqrt(2/m(i-1)).*randn(m(i),m(i-1));
        gamma{i} = ones(m(i),1);
        beta{i} = zeros(m(i),1);
        if bias == 1
            b{i} = sqrt(2/m(i-1)).*randn(m(i),1);
        else
            b{i} = zeros(m(i),1);
        end     
    end
end
    
end 




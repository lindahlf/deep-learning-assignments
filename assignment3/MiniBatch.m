function [Wstar,bstar, loss_train, loss_valid, varargout] = MiniBatch(trainX, trainY, validX,...
    validY, W, b, parameters, varargin)
% Performs cyclical mini-batch algorithm
% Inputs:
% trainX = training data 
% trainY = labels of training data
% validX = validation data
% validY = labels of validation data
% W = cell of initial weights for each layer
% b = cell initial biases for each layer
% parameters = struct of algorithm parameters
% (optional) gamma = cell of initial stretch vectors for each layer
% (optional) beta = cell of initial shift vectors for each leyer
%
% Returns:
% Wstar = cell of final weight matrices
% bstar = cell of final bias vectos
% loss_train = values of loss function for training data at different
% intervals of the iteration
% loss_valid = values of loss function for validation data at different
% intervals of the iteration
% (optional) gammastar = cell of final stretch vectors for each layer
% (optional) betastar = cell of final shift vectors for each layer



W_curr = W; b_curr = b;
k = length(W); % number of layers
[~,n] = size(trainX);

if ~isempty(varargin)
    gamma_curr = varargin{1};
    beta_curr = varargin{2};
    bn = true;
else
    bn = false;
end


% Importing parameter values;
eta_min = parameters.eta_min; 
eta_max = parameters.eta_max;
ns = parameters.ns;
nbatch = parameters.nbatch;
lambda = parameters.lambda; 
cycles = parameters.cycles; 

% Generating values for eta
etaup = eta_min + linspace(0, eta_max - eta_min, ns); % increasing part
etadown = eta_max - linspace(0, eta_max - eta_min, ns); % decreasing part
eta = [etaup, etadown];
eta = repmat(eta,1,cycles);

% Arrays to store values of loss functions
loss_train = zeros(11,1);
loss_valid = zeros(11,1);

plotidx = 1;

for t = 0:2*ns*cycles-1
    
     % Shuffle the dataset
    if mod(t,n/nbatch) + 1 == 1
        perm = randperm(n);
        trainX = trainX(:,perm);
        trainY = trainY(:,perm);
    end

    % Generating batch
    j = mod(t,n/nbatch) + 1; 
    j_start = (j-1)*nbatch + 1;
    j_end = j*nbatch;
    inds = j_start:j_end;
    Xbatch = trainX(:, inds); 
    Ybatch = trainY(:, inds);
    
    if bn
        % Computing gradients
        [gradb, gradW, gradgamma, gradbeta] = ComputeGradients(Xbatch, ...
            Ybatch,W_curr,b_curr,lambda, gamma_curr, beta_curr);
        
        % Updating b,W,gamma and beta
        for l = 1:k
            b_curr{l} = b_curr{l} - eta(t+1)*gradb{l};
            W_curr{l} = W_curr{l} - eta(t+1)*gradW{l};
            if l < k
                gamma_curr{l} = gamma_curr{l} - eta(t+1)*gradgamma{l};
                beta_curr{l} = beta_curr{l} - eta(t+1)*gradbeta{l};
            end
        end
            
    else
        % Computing gradients
        [gradb, gradW] = ComputeGradients(Xbatch,Ybatch,W_curr,b_curr,lambda);

        % Updating b and W 
        for l = 1:k
            b_curr{l} = b_curr{l} - eta(t+1)*gradb{l};
            W_curr{l} = W_curr{l} - eta(t+1)*gradW{l};
        end
    end

    % Recording values for loss function
    % TODO: make sure 
    if mod(t,2*ns*cycles/10) == 0
        disp([num2str(round(t/(2*ns*cycles-1)*100,2)), '% done'])
        
        if bn
            Params.W = W_curr; Params.b = b_curr; Params.use_bn = true;
            Params.gammas = gamma_curr; Params.betas = beta_curr;

            [~, losstrain] = ComputeCost(trainX,trainY, Params, lambda); 
            [~, lossvalid] = ComputeCost(validX,validY, Params, lambda);

            loss_train(plotidx) = losstrain;
            loss_valid(plotidx) = lossvalid; 
        else
            Params.W = W_curr; Params.b = b_curr; Params.use_bn = false;
            

            [~, losstrain] = ComputeCost(trainX,trainY, Params, lambda); 
            [~, lossvalid] = ComputeCost(validX,validY, Params, lambda);

            loss_train(plotidx) = losstrain;
            loss_valid(plotidx) = lossvalid;
        end

        plotidx = plotidx + 1;
    end
end

disp([num2str(round(t/(2*ns*cycles-1)*100,1)), '% done']) 

if bn
    Params.W = W_curr; Params.b = b_curr; Params.use_bn = true;
    Params.gammas = gamma_curr; Params.betas = beta_curr;
    [~, losstrain] = ComputeCost(trainX,trainY, Params, lambda); 
    [~, lossvalid] = ComputeCost(validX,validY, Params, lambda);

    loss_train(plotidx) = losstrain;
    loss_valid(plotidx) = lossvalid;
    
    varargout{1} = gamma_curr;
    varargout{2} = beta_curr;
else
    Params.W = W_curr; Params.b = b_curr; Params.use_bn = false;
    [~, losstrain] = ComputeCost(trainX,trainY, Params, lambda); 
    [~, lossvalid] = ComputeCost(validX,validY, Params, lambda);

    loss_train(plotidx) = losstrain;
    loss_valid(plotidx) = lossvalid;
end

Wstar = W_curr; 
bstar = b_curr;


end


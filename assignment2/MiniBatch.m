function [Wstar,bstar, loss_train, loss_valid] = MiniBatch(trainX, trainY, validX, validY, W, b, parameters)
% Performs cyclical mini-batch algorithm
% W and b are initial weights and bias. trainX and trainY are images and
% corresponding labels for training data. validX and validY for validation
% data.
% Parameters is a struct of all algorithm parametes
% Returns final solutions Wstar and bstar together with arrays of loss 
% functions for training and validation data at different intervals for 
% plotting results.

W_curr = W; b_curr = b;
[~,n] = size(trainX);

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
cycles = 3;
eta = [etaup, etadown];
eta = repmat(eta,1,cycles);

% Arrays to store 
loss_train = zeros(11,1);
loss_valid = zeros(11,1);

plotidx = 1;

for t = 0:2*ns*cycles-1
    % Generating batch
    j = mod(t,n/nbatch) + 1; 
    j_start = (j-1)*nbatch + 1;
    j_end = j*nbatch;
    inds = j_start:j_end;
    Xbatch = trainX(:, inds); 
    Ybatch = trainY(:, inds);
    
    % Computing gradients
    [gradb, gradW] = ComputeGradients(Xbatch,Ybatch,W_curr,b_curr,lambda);
    
    % Updating b and W 
    b_curr{1} = b_curr{1} - eta(t+1)*gradb{1};
    b_curr{2} = b_curr{2} - eta(t+1)*gradb{2};    
    W_curr{1} = W_curr{1} - eta(t+1)*gradW{1};
    W_curr{2} = W_curr{2} - eta(t+1)*gradW{2};
    
    % Recording values for loss function
    if mod(t,2*ns*cycles/10) == 0
    disp([num2str(round(t/(2*ns*cycles-1)*100,2)), '% done'])    
        
    [~, losstrain] = ComputeCost(trainX,trainY, W_curr, b_curr, lambda); 
    [~, lossvalid] = ComputeCost(validX,validY, W_curr, b_curr, lambda);

    loss_train(plotidx) = losstrain;
    loss_valid(plotidx) = lossvalid;
    
    plotidx = plotidx + 1;
    end
end

disp([num2str(round(t/(2*ns*cycles-1)*100,1)), '% done']) 
[~, losstrain] = ComputeCost(trainX,trainY, W_curr, b_curr, lambda); 
[~, lossvalid] = ComputeCost(validX,validY, W_curr, b_curr, lambda);

loss_train(plotidx) = losstrain;
loss_valid(plotidx) = lossvalid;

Wstar = W_curr; 
bstar = b_curr;

end


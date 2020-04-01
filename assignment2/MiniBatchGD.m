 function [Wstar, bstar, losstrain, lossval] = MiniBatchGD(X, Y, Xval, Yval, GDparams, W, b, lambda)
% X contains all the n training images, with dimension d, dxn. 
% Y contains all the labels for the n training images, Kxn,
% W and b are initial parameters, they have sizes Kxd and Kx1
% respectively.
% GDparams is an object containing the parameter values n_batch,
% eta and n_epochs, denoting the size of the mini-batches, the learning
% rate and the number of runs through the whole training set, respectively.
% Lambda is the regularization parameter
% Returns Wstar and bstar. 

n_batch = GDparams.nbatch;
eta = GDparams.eta;
n_epochs = GDparams.nepochs;

[~, n] = size(X);
W_curr = W; b_curr = b;

losstrain = zeros(1,n_epochs);
lossval = zeros(1,n_epochs);

for i = 1:n_epochs
    % Generate set of mini-batches
    disp(['Current epoch: ', num2str(i)])
    perm = randperm(n);
    % Shuffle the dataset
    X = X(:,perm);
    Y = Y(:,perm);
    for j = 1:n/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        Xbatch = X(:, inds); 
        Ybatch = Y(:, inds);
        Pbatch = EvaluateClassifier(Xbatch, W_curr, b_curr);
        
        [grad_b, grad_W] = ComputeGradients(Xbatch, Ybatch, Pbatch, W_curr, lambda);
        
        b_curr = b_curr - eta*grad_b;
        W_curr = W_curr - eta*grad_W;
    end
    
    losstrain(i) = ComputeCost(X, Y, W_curr, b_curr, lambda);
    lossval(i) = ComputeCost(Xval, Yval, W_curr, b_curr, lambda);
    eta = decay*eta; % Add decay
end

Wstar = W_curr;
bstar = b_curr;
end
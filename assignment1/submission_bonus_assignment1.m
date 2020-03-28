%% Functions

function P = EvaluateClassifier(X, W, b)
% Each column of X corresponds to an image and has size dxn where d is the
% dimension of the image and n is the number of images
% W and b are the parameters of the network
% Returns P which contains the probability for each label for the image in
% the corresponding column of X. Has size Kxn

s = W*X + b;
exponent = exp(s);
P = exponent./sum(exponent,1);

end

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

end

function acc = ComputeAccuracy(X, y, W, b)
% Each column of X corresponds to an image, X is dxn
% y is the vector of ground truth values of length n
% Returns the scalar value acc containing the accuracy

P = EvaluateClassifier(X,W,b);
% Recall that y contains the true label of the data

[~,I] = max(P,[],1);
match = I == y';

acc = sum(match)/length(y);

end

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
decay = GDparams.decay;

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

%% Load data 
clear all; close all; clc

% Using all avaliable data
trainX = zeros(3072,10000*5);
trainY = zeros(10, 10000*5);
trainy = zeros(10000*5,1);

for i = 1:5
    filename = sprintf('data_batch_%d.mat', i);
    [fooX, fooY, fooy] = LoadBatch(filename);
    trainX(:,(1+(i-1)*10000:i*10000)) = fooX;
    trainY(:,(1+(i-1)*10000:i*10000)) = fooY;
    trainy((1+(i-1)*10000:i*10000),:) = fooy;
end

validX = trainX(:,49001:end); 
validY = trainY(:,49001:end); 
validy = trainy(49001:end,:);

trainX = trainX(:,1:49000);
trainY = trainY(:,1:49000);
trainy = trainy(1:49000,:);

[testX, testY, testy] = LoadBatch('test_batch.mat');

labels = load('batches.meta.mat');
label_names = labels.label_names;

%% Prepare data and initialize constants

% Compute mean of training data 
mean_X = mean(trainX, 2); 
std_X = std(trainX, 0, 2);

% Normalize data
trainX = trainX - repmat(mean_X, [1, size(trainX, 2)]);
trainX = trainX ./ repmat(std_X, [1, size(trainX, 2)]);


validX = validX - repmat(mean_X, [1, size(validX, 2)]);
validX = validX ./ repmat(std_X, [1, size(validX, 2)]);

testX = testX - repmat(mean_X, [1, size(testX, 2)]);
testX = testX ./ repmat(std_X, [1, size(testX, 2)]);

% Initialize W and b 
[K, ~] = size(trainY);
[d,n] = size(trainX);

rng(400);
W = 0.01.*randn(K, d);
b = 0.01.*randn(K,1);

%% Performing mini-batch step

% Setting minibatch parameters
lambda = 0.1;
n_epochs = 40;
n_batch = 100;
eta = 0.001;
decay = 0.95;

GDparams.nbatch = n_batch;
GDparams.eta = eta;
GDparams.nepochs = n_epochs;
GDparams.decay = decay;

% Mini-batch step
[Wstar, bstar, trainloss, valloss] = MiniBatchGD(trainX, trainY, validX, validY, ...
    GDparams, W, b, lambda);

% Computing accuracies
acc_train = ComputeAccuracy(trainX,trainy, Wstar,bstar);
acc_val = ComputeAccuracy(validX,validy, Wstar, bstar);
acc_test = ComputeAccuracy(testX,testy, Wstar, bstar);


%% Plotting cost function
epoch = (1:1:n_epochs);
plot(epoch, trainloss, epoch, valloss, 'LineWidth', 1.5)
title({'Training and validation loss for each epoch',...
    ['lambda = ' num2str(lambda)],...
    ['nbatch = ' num2str(n_batch)], ['eta = ' num2str(eta)],...
    ['nepochs = ' num2str(n_epochs)], ...
    ['decay = ' num2str(decay)]})
xlabel('Epochs')
ylabel('Loss')
legend('Training loss', 'Validation loss', 'FontSize', 20)
set(gca,'FontSize',20)
set(gcf, 'Position',  [100, 100, 1000, 1000]);
filename = sprintf('bonus_lambda%0.5gnepochs%0.5gnbatch%0.5geta%0.5g.png', lambda, n_epochs, n_batch,eta);
saveas(gcf,filename)
%% Displaying the learnt weight matrix

% Visualize templates
for i = 1:10
    im = reshape(Wstar(i,:), 32, 32, 3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
    
end

% Assembling images
for i = 1:10
    subplot(2,5,i)
    imagesc(s_im{i})
    set(gca,'XTick',[], 'YTick', [])
    title(label_names{i})
end

set(gcf, 'Position',  [100, 100, 2000, 500]);
filename = sprintf('bonus_weight_lambda%0.5gnepochs%0.5gnbatch%0.5geta%0.5g.png', lambda, n_epochs, n_batch,eta);
sgtitle({'Learnt weight matrix for each class',...
    ['lambda = ' num2str(lambda)],...
    ['nbatch = ' num2str(n_batch)], ['eta = ' num2str(eta)],...
    ['nepochs = ' num2str(n_epochs)], ...
    ['decay = ' num2str(decay)]})
saveas(gcf,filename)


    







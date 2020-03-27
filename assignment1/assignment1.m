% clear all; close all; clc

% Load data 
[trainX,trainY,trainy] = LoadBatch('data_batch_1.mat');
[validX,validY,validy] = LoadBatch('data_batch_2.mat');
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

%% Test of functions

% Testing cost function
J1 = ComputeCost(trainX,trainy,W,b,5);
J2 = ComputeCost(trainX,trainY,W,b,5);

% Test accuracy function 
acc = ComputeAccuracy(trainX,trainy, W,b);

% Getting an understanding for the gradients
[gradbslow, gradwslow] = ComputeGradsNumSlow(trainX(1:20,1), trainY(:,1), W(:, 1:20), b, 0, 1e-06);
[gradb, gradw] = ComputeGradsNum(trainX(1:20,1), trainY(:,1), W(:, 1:20), b, 0, 1e-06);

P = EvaluateClassifier(trainX(1:20,1), W(:, 1:20), b);
[own_gradb,own_gradw] = ComputeGradients(trainX(1:20,1), trainY(:,1), P, W(:, 1:20),0);

% Check error of gradient
eps = 1e-10;

errorb = norm(gradbslow - own_gradb)/max(eps,norm(own_gradb)+norm(gradbslow));
errorw = norm(gradwslow - own_gradw)/max(eps,norm(own_gradw)+norm(gradwslow));

%% Performing mini-batch step

% Setting minibatch parameters
lambda = 0.1;
n_epochs = 40;
n_batch = 100;
eta = 0.001;

GDparams.nbatch = n_batch;
GDparams.eta = eta;
GDparams.nepochs = n_epochs;

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
    ['nepochs = ' num2str(n_epochs)]})
xlabel('Epochs')
ylabel('Loss')
legend('Training loss', 'Validation loss', 'FontSize', 20)
set(gca,'FontSize',20)
set(gcf, 'Position',  [100, 100, 1000, 1000]);
filename = sprintf('lambda%0.5gnepochs%0.5gnbatch%0.5geta%0.5g.png', lambda, n_epochs, n_batch,eta);
% saveas(gcf,filename)
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
filename = sprintf('weight_lambda%0.5gnepochs%0.5gnbatch%0.5geta%0.5g.png', lambda, n_epochs, n_batch,eta);
sgtitle({'Learnt weight matrix for each class',...
    ['lambda = ' num2str(lambda)],...
    ['nbatch = ' num2str(n_batch)], ['eta = ' num2str(eta)],...
    ['nepochs = ' num2str(n_epochs)]})
% saveas(gcf,filename)


    



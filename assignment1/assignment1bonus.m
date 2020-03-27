%% Load data 
clear all; close all; clc
% Use all avaliable data

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
gi
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



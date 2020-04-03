%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Functions are found at the end of this script %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc
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

% Initialize parameters  
[K, ~] = size(trainY);
[d,n] = size(trainX);
nodes = 50; 
bias = 0;

rng(400)
[W,b] = initParams(nodes, d, K, bias);
[P,H] = EvalClassifier(trainX, W, b);

%% Testing the gradients

rng(400)
[W,b] = initParams(nodes, d, K, bias);
W1 = W{1};
W{1} = W1(:, 1:20);

[grad_b, grad_W] = ComputeGradsNumSlow2(trainX(1:20,1:2), trainY(:,1:2), W, b, 0, 1e-05);
[own_gradb,own_gradw] = ComputeGradients(trainX(1:20,1:2), trainY(:,1:2), W, b);

% Check error of gradient
eps = 1e-10;

errorb1 = norm(grad_b{1}  - own_gradb{1})/max(eps,norm(own_gradb{1})+norm(grad_b{1}));
errorb2 = norm(grad_b{2}  - own_gradb{2})/max(eps,norm(own_gradb{2})+norm(grad_b{2}));

errorW1 = norm(grad_W{1}  - own_gradw{1})/max(eps,norm(own_gradw{1})+norm(grad_W{1}));
errorW2 = norm(grad_W{2}  - own_gradw{2})/max(eps,norm(own_gradw{2})+norm(grad_W{2}));

%% Trying to overfit training data to check gradient

% Initialzing parameters
rng(400)
[W,b] = initParams(nodes, d, K, bias);

eta = 0.01; 
epochs = 200; 
cost = zeros(1,epochs);
valcost = zeros(1,epochs); 

for i = 1:epochs
    [gradb, gradW] = ComputeGradients(trainX(:,1:100), trainY(:,1:100), W, b);
    W{1} = W{1} - eta*gradW{1};
    W{2} = W{2} - eta*gradW{2};
    b{1} = b{1} - eta*gradb{1};
    b{2} = b{2} - eta*gradb{2};
    cost(i) = ComputeCost(trainX(:,1:100), trainY(:,1:100), W, b, 0);
    valcost(i) = ComputeCost(validX, validY, W, b, 0);
end

% Plotting results
plot((1:1:epochs),cost, (1:1:epochs), valcost, 'LineWidth', 1.5)
title('Overfitting network to check gradient')
legend('Training data', 'Validation data')
xlabel('Epochs')
ylabel('Cost function')
set(gca,'FontSize',20)
set(gcf, 'Position',  [100, 100, 1000, 1000]);

%% Training network using cyclical learning rates

% Setting parameters
rng(400)
[W,b] = initParams(nodes, d, K, bias);
[~, n] = size(trainX);

eta_min = 1e-5; eta_max = 1e-1; ns = 500;
nbatch = 100; lambda = 0.01;

% Generating values for eta
etaup = eta_min + linspace(0, eta_max - eta_min, ns); % increasing part
etadown = eta_max - linspace(0, eta_max - eta_min, ns); % decreasing part
eta = [etaup, etadown];

W_curr = W; b_curr = b;

% Initializing arrays to record cost and loss 
cost_train = zeros(11,1); loss_train = zeros(11,1); acc_train = zeros(11,1);
cost_valid = zeros(11,1); loss_valid = zeros(11,1); acc_valid = zeros(11,1);

plotidx = 1;

for t = 0:2*ns-1
    % Generating batch
    j = mod(t,n/nbatch) + 1; 
    j_start = (j-1)*nbatch + 1;
    j_end = j*nbatch;
    inds = j_start:j_end;
    Xbatch = trainX(:, inds); 
    Ybatch = trainY(:, inds);
    
    [gradb, gradW] = ComputeGradients(Xbatch,Ybatch,W_curr,b_curr,lambda);
    
    % Updating b and W 
    b_curr{1} = b_curr{1} - eta(t+1)*gradb{1};
    b_curr{2} = b_curr{2} - eta(t+1)*gradb{2};
    
    W_curr{1} = W_curr{1} - eta(t+1)*gradW{1};
    W_curr{2} = W_curr{2} - eta(t+1)*gradW{2};
    
    % Recording values every 100th iteration to plot for sanity check
    if mod(t,100) == 0
        
    [costtrain, losstrain] = ComputeCost(trainX,trainY, W_curr, b_curr, lambda); 
    [costvalid, lossvalid] = ComputeCost(validX,validY, W_curr, b_curr, lambda);

    cost_train(plotidx) = costtrain; loss_train(plotidx) = losstrain;
    cost_valid(plotidx) = costvalid; loss_valid(plotidx) = lossvalid;
    
    acc_train(plotidx) = ComputeAccuracy(trainX,trainy, W_curr, b_curr); 
    acc_valid(plotidx) = ComputeAccuracy(validX,validy, W_curr, b_curr); 
    
    plotidx = plotidx + 1;
    end
end 

[costtrain, losstrain] = ComputeCost(trainX,trainY, W_curr, b_curr, lambda); 
[costvalid, lossvalid] = ComputeCost(validX,validY, W_curr, b_curr, lambda);

cost_train(plotidx) = costtrain; loss_train(plotidx) = losstrain;
cost_valid(plotidx) = costvalid; loss_valid(plotidx) = lossvalid; 

acc_train(plotidx) = ComputeAccuracy(trainX,trainy, W_curr, b_curr); 
acc_valid(plotidx) = ComputeAccuracy(validX,validy, W_curr, b_curr); 

% Compute final test accuracy
acc_valid = ComputeAccuracy(testX,testy, W_curr, b_curr); 

% Figures of cost, loss and accuracy
figure
plot(0:100:2*ns,cost_train, 0:100:2*ns,cost_valid, 'LineWidth', 1.5)
xlabel('update step')
ylabel('cost')
ylim([0,4])
legend('training', 'validation')
title('Cost plot')
set(gca,'FontSize',20)
set(gcf, 'Position',  [100, 100, 1000, 1000]);

figure 
plot(0:100:2*ns,loss_train, 0:100:2*ns,loss_valid, 'LineWidth', 1.5)
xlabel('update step')
ylabel('loss')
ylim([0,4])
legend('training', 'validation')
title('Loss plot')
set(gca,'FontSize',20)
set(gcf, 'Position',  [100, 100, 1000, 1000]);

figure 
plot(0:100:2*ns, acc_train, 0:100:2*ns, acc_valid, 'LineWidth', 1.5)
xlabel('update step')
ylabel('accuracy')
ylim([0,0.7])
legend('training', 'validation')
title('Accuracy plot')
set(gca,'FontSize',20)
set(gcf, 'Position',  [100, 100, 1000, 1000]);

%% Training network for real 

% Initializing parameters
rng(400)
[W,b] = initParams(nodes, d, K, bias);
[~, n] = size(trainX);

eta_min = 1e-5; eta_max = 1e-1; ns = 800;
nbatch = 100; lambda = 0.01;

% Generating values for eta
etaup = eta_min + linspace(0, eta_max - eta_min, ns); % increasing part
etadown = eta_max - linspace(0, eta_max - eta_min, ns); % decreasing part
cycles = 2;
eta = [etaup, etadown];
eta = repmat(eta,1,cycles);

W_curr = W; b_curr = b;

cost_train = zeros(11,1); loss_train = zeros(11,1); acc_train = zeros(11,1);
cost_valid = zeros(11,1); loss_valid = zeros(11,1); acc_valid = zeros(11,1);

plotidx = 1;

for t = 0:2*ns*cycles-1
    % Generating batch
    j = mod(t,n/nbatch) + 1; 
    j_start = (j-1)*nbatch + 1;
    j_end = j*nbatch;
    inds = j_start:j_end;
    Xbatch = trainX(:, inds); 
    Ybatch = trainY(:, inds);
    
    [gradb, gradW] = ComputeGradients(Xbatch,Ybatch,W_curr,b_curr,lambda);
    
    % Updating b and W 
    b_curr{1} = b_curr{1} - eta(t+1)*gradb{1};
    b_curr{2} = b_curr{2} - eta(t+1)*gradb{2};
    
    W_curr{1} = W_curr{1} - eta(t+1)*gradW{1};
    W_curr{2} = W_curr{2} - eta(t+1)*gradW{2};
    
    % Recording values every 480th iteration to plot for sanity check
    if mod(t,480) == 0
        
    [costtrain, losstrain] = ComputeCost(trainX,trainY, W_curr, b_curr, lambda); 
    [costvalid, lossvalid] = ComputeCost(validX,validY, W_curr, b_curr, lambda);

    cost_train(plotidx) = costtrain; loss_train(plotidx) = losstrain;
    cost_valid(plotidx) = costvalid; loss_valid(plotidx) = lossvalid;
    
    acc_train(plotidx) = ComputeAccuracy(trainX,trainy, W_curr, b_curr); 
    acc_valid(plotidx) = ComputeAccuracy(validX,validy, W_curr, b_curr); 
    
    plotidx = plotidx + 1;
    end
end 

[costtrain, losstrain] = ComputeCost(trainX,trainY, W_curr, b_curr, lambda); 
[costvalid, lossvalid] = ComputeCost(validX,validY, W_curr, b_curr, lambda);

cost_train(plotidx) = costtrain; loss_train(plotidx) = losstrain;
cost_valid(plotidx) = costvalid; loss_valid(plotidx) = lossvalid; 

acc_train(plotidx) = ComputeAccuracy(trainX,trainy, W_curr, b_curr); 
acc_valid(plotidx) = ComputeAccuracy(validX,validy, W_curr, b_curr); 

% Compute final test accuracy
acc_test = ComputeAccuracy(testX,testy, W_curr, b_curr); 

% Figures of cost, loss and accuracy
figure
plot(0:480:2*ns*cycles,cost_train, 0:480:2*ns*cycles,cost_valid, 'LineWidth', 1.5)
xlabel('update step')
ylabel('cost')
ylim([0,4])
legend('training', 'validation')
title('Cost plot')
set(gca,'FontSize',20)
set(gcf, 'Position',  [100, 100, 1000, 1000]);

figure
plot(0:480:2*ns*cycles,loss_train, 0:480:2*ns*cycles,loss_valid, 'LineWidth', 1.5)
xlabel('update step')
ylabel('loss')
ylim([0,4])
legend('training', 'validation')
title('Loss plot')
set(gca,'FontSize',20)
set(gcf, 'Position',  [100, 100, 1000, 1000]);

figure
plot(0:480:2*ns*cycles, acc_train, 0:480:2*ns*cycles, acc_valid, 'LineWidth', 1.5)
xlabel('update step')
ylabel('accuracy')
ylim([0,0.7])
legend('training', 'validation')
title('Accuracy plot')
set(gca,'FontSize',20)
set(gcf, 'Position',  [100, 100, 1000, 1000]);


%% Coarse search to set lambda
clear all; close all; clc

% Using all avaliable data
trainX = zeros(3072,10000*5);
trainY = zeros(10, 10000*5);
trainy = zeros(10000*5,1);
disp('Loading data')
for i = 1:5
    disp('...')
    filename = sprintf('data_batch_%d.mat', i);
    [fooX, fooY, fooy] = LoadBatch(filename);
    trainX(:,(1+(i-1)*10000:i*10000)) = fooX;
    trainY(:,(1+(i-1)*10000:i*10000)) = fooY;
    trainy((1+(i-1)*10000:i*10000),:) = fooy;
end

% Reserve 5000 images for validation
validX = trainX(:,45001:end); 
validY = trainY(:,45001:end); 
validy = trainy(45001:end,:);

trainX = trainX(:,1:45000);
trainY = trainY(:,1:45000);
trainy = trainy(1:45000,:);

[testX, testY, testy] = LoadBatch('test_batch.mat');

disp('Preparing data...')
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

disp('Setting parameters')
% Setting parameters
[K, ~] = size(trainY);
[d,n] = size(trainX);
nodes = 50; 
bias = 0;

rng(400)
[W,b] = initParams(nodes, d, K, bias);
[P,H] = EvalClassifier(trainX, W, b);

% Cyclical mini-batch parameters
eta_min = 1e-5; eta_max = 1e-1; 
nbatch = 100; cycles = 2;
ns = 2*floor(n/nbatch);

% Lambda search parameters
l_min = -5; l_max = -1;
no_values = 8;
l = linspace(l_min, l_max, no_values);
lambda = 10.^l;

parameters.eta_min = eta_min; parameters.eta_max = eta_max;
parameters.ns = ns; parameters.nbatch = nbatch; 
parameters.cycles = cycles;

acc_valid = zeros(1, no_values);

disp('Performing grid search')
for i = 1:length(lambda)
    disp('...')
    parameters.lambda = lambda(i); 
    [Wstar, bstar] = MiniBatch(trainX, trainY, W, b, parameters);
    acc_valid(i) = ComputeAccuracy(validX,validy, Wstar, bstar);
end

% Save results to file 
T = table(lambda',acc_valid');
writetable(T,'gridSearch.txt','Delimiter',' ')  
type 'gridSearch.txt'
 
%% Random search to set lambda 

disp('Setting parameters')
% Setting parameters
[K, ~] = size(trainY);
[d,n] = size(trainX);
nodes = 50; 
bias = 0;

rng(400)
[W,b] = initParams(nodes, d, K, bias);
[P,H] = EvalClassifier(trainX, W, b);

% Cyclical mini-batch parameters
eta_min = 1e-5; eta_max = 1e-1; 
nbatch = 100; cycles = 3;
ns = 2*floor(n/nbatch);


% Lambda search parameters
% Interval based on best performance from coarse search
l_min = log10(0.0019307); l_max = log10(0.00193068); 
no_values = 8;
l = l_min + (l_max-l_min)*rand(1,no_values);
lambda = 10.^l;

parameters.eta_min = eta_min; parameters.eta_max = eta_max;
parameters.ns = ns; parameters.nbatch = nbatch; 
parameters.cycles = cycles;

acc_valid = zeros(1, no_values);

disp('Performing random search')
for i = 1:length(lambda)
    disp('...')
    parameters.lambda = lambda(i); 
    [Wstar, bstar] = MiniBatch(trainX, trainY, W, b, parameters);
    acc_valid(i) = ComputeAccuracy(validX,validy, Wstar, bstar);
end

% Save results to file 
T = table(lambda',acc_valid');
writetable(T,'randomSearch.txt','Delimiter',' ')  
type 'randomSearch.txt'

%% Final training
clear all; close all; clc

% Using all avaliable data
trainX = zeros(3072,10000*5);
trainY = zeros(10, 10000*5);
trainy = zeros(10000*5,1);

disp('Loading data')
for i = 1:5
    disp('...')
    filename = sprintf('data_batch_%d.mat', i);
    [fooX, fooY, fooy] = LoadBatch(filename);
    trainX(:,(1+(i-1)*10000:i*10000)) = fooX;
    trainY(:,(1+(i-1)*10000:i*10000)) = fooY;
    trainy((1+(i-1)*10000:i*10000),:) = fooy;
end

% Reserve 1000 images for validation
validX = trainX(:,49001:end); 
validY = trainY(:,49001:end); 
validy = trainy(49001:end,:);

trainX = trainX(:,1:49000);
trainY = trainY(:,1:49000);
trainy = trainy(1:49000,:);

[testX, testY, testy] = LoadBatch('test_batch.mat');

disp('Preparing data...')

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

disp('Setting parameters')
% Setting parameters
[K, ~] = size(trainY);
[d,n] = size(trainX);
nodes = 50; 
bias = 0;

% Initializing weights and nodes 
rng(400)
[W,b] = initParams(nodes, d, K, bias);
[P,H] = EvalClassifier(trainX, W, b);

% Cyclical mini-batch parameters
eta_min = 1e-5; eta_max = 1e-1; 
nbatch = 100; cycles = 4;
ns = 2*floor(n/nbatch); lambda = 0.00193069772888325;

parameters.eta_min = eta_min; parameters.eta_max = eta_max;
parameters.ns = ns; parameters.nbatch = nbatch; 
parameters.cycles = cycles; parameters.lambda = lambda;

disp('Performing mini-batch...')
[Wstar, bstar, losstrain, lossvalid] = MiniBatch(trainX, trainY, validX, validY, ...
    W, b, parameters);

% Final test accuracy 
acc_test = ComputeAccuracy(testX,testy, Wstar, bstar);

plot(0:2*ns*cycles/10:2*ns*cycles,losstrain, 0:2*ns*cycles/10:2*ns*cycles,lossvalid, 'LineWidth', 1.5)
xlabel('update step')
ylabel('loss')
legend('training', 'validation')
title('Loss plot')
set(gca,'FontSize',20)
set(gcf, 'Position',  [100, 100, 1000, 1000]);

%% Functions 

function [W,b] = initParams(m, d, K, bias)
% Initializing the parameters for the network
% Parameters:
% m = number of nodes 
% d = dimension of the images
% K = number of labels 
% bias = 1 or 0. 0 = no bias, 1 = bias. 
% Returns:
% W1, size = mxd
% b1, size = mx1
% W2, size = Kxm
% b2, size = Kx1
% W1 and W2 stored in cell W
% b1 and b2 stored in cell b


W1 = (1/sqrt(d)).*randn(m,d);
W2 = (1/sqrt(m)).*randn(K,m);

if bias == 1
    b1 = (1/sqrt(d)).*randn(m,1);
    b2 = (1/sqrt(m)).*randn(K,1);
else
    b1 = zeros(m,1);
    b2 = zeros(K,1);
end

W = {W1,W2};
b = {b1,b2};

end 

function [P,H] = EvalClassifier(X, W, b)
% Each column of X corresponds to an image and has size dxn where d is the
% dimension of the image and n is the number of images
% W1, W2, b1 and b2 are the parameters of the network stored in cells.
% Returns P which contains the probability for each label for the image in
% the corresponding column of X. Has size Kxn

W1 = W{1}; W2 = W{2};
b1 = b{1}; b2 = b{2};

s1 = W1*X + b1; 
H = max(0,s1);
s = W2*H + b2;
exponent = exp(s);
P = exponent./sum(exponent,1);
end

function [J,loss] = ComputeCost(X, Y, W, b, lambda)
% Size dxn where each column of X corresponds to an image
% Each column of Y is either the Kxn one-hot ground truth label for
% corresponding column of X or is simply the 1xn vector of ground truth
% labels.
% Returns a scalar J corresponding the sum of the loss of the network's
% predictions for the images in X relative to the ground truth labels and
% the regularization term on W 


P = EvalClassifier(X,W,b);

W1 = W{1}; W2 = W{2}; 

[~,n] = size(X);

if ismember(0,Y) == 0 % 1xn 
    idx = sub2ind(size(P), Y, (1:1:n)');
    loss = 1/n*sum(-log(P(idx)));
    J = loss + lambda*(norm(W1,'fro')^2 + norm(W2,'fro')^2);
else % One-hot ground truth label
    loss = 1/n*sum(-log(sum(Y.*P)));
    J = loss + lambda*(norm(W1,'fro')^2 + norm(W2,'fro')^2);
end

end

function acc = ComputeAccuracy(X, y, W, b)
% Each column of X corresponds to an image, X is dxn
% y is the vector of ground truth values of length n
% Returns the scalar value 'acc' containing the accuracy 

P = EvalClassifier(X,W,b);

[~,I] = max(P,[],1); % Finding all matching labels
match = I == y';
acc = sum(match)/length(y);

end

function [gradb, gradW] = ComputeGradients(X,Y,W,b,lambda)
% Computes gradients for the weights W and bias b using forward and
% backward propagation
% W and b and the corresponding returned gradients, gradb and gradW, are on
% 1x2 cell form. 
% X and Y are the images and their labels. Lambda is the regularzation
% parameter.

W1 = W{1}; W2 = W{2};

[P,H] = EvalClassifier(X,W,b);

[~,n] = size(Y);

G = -(Y-P);

db2 = (1/n)*G*ones(n,1);
dW2 = (1/n)*G*H' + 2*lambda*W2;

G = W2'*G;
Ind = H > 0;
G = G.*Ind;

dW1 = (1/n)*G*X' + 2*lambda*W1;
db1 = (1/n)*G*ones(n,1);

gradW = {dW1, dW2};
gradb = {db1, db2};
end

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





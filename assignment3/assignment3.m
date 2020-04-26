%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%                                               %%%%
%%%% FUNCTIONS ARE FOUND AT THE END OF THIS SCRIPT %%%%
%%%%                                               %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc
% Load data 
[trainX,trainY,trainy] = LoadBatch('data_batch_1.mat');
[validX,validY,validy] = LoadBatch('data_batch_2.mat');
[testX, testY, testy] = LoadBatch('test_batch.mat');
labels = load('batches.meta.mat');
label_names = labels.label_names;

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

%% Using all avaliable data
clear all; close all; clc
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
disp('Done')

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

%%Initialize constants

% Initialize parameters  
[K, ~] = size(trainY); % K = number of labels
[d,n] = size(trainX);
nodes = [50 K]; % number of nodes in each layer
bias = 0; 
k = length(nodes);

rng(400)
[W,b] = initParams(k, nodes, n, d, bias);
[P,H] = EvalClassifier(trainX, W, b);

%% Testing gradients
grad_test_X = trainX(1:10,1:2); grad_test_Y = trainY(:,1:2);

% Initialize parameters  
[K, ~] = size(grad_test_Y); % K = number of labels
[d,n] = size(grad_test_X);
nodes = [50 50 20 K]; % number of nodes in each layer
bias = 0; 
k = length(nodes);

rng(400)
[W,b] = initParams(k, nodes, n, d, bias);

lambda = 0;

[gradb, gradW] = ComputeGradients(grad_test_X,grad_test_Y,W,b,lambda);

NetParams.W = W; NetParams.b = b; NetParams.use_bn = false;
Grads = ComputeGradsNumSlow(grad_test_X, grad_test_Y, NetParams, lambda, 1e-05);
gradbRef = Grads.b; gradWRef = Grads.W;

% Check error
eps = 1e-10;
layer = 1;

errorb = norm(gradbRef{layer}  - gradb{layer})/max(eps,norm(gradb{layer})+norm(gradbRef{layer}));
errorW = norm(gradWRef{layer}  - gradW{layer})/max(eps,norm(gradW{layer})+norm(gradWRef{layer}));

%% Trying to replicate results from assignment 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Make sure to load data before running this section %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize parameters  
[K, ~] = size(trainY); % K = number of labels
[d,n] = size(trainX);
nodes = [50 30 20 20 10 10 10 10 K]; % number of nodes in each layer
bias = 0; 
k = length(nodes);

rng(400)
[W,b] = initParams(k, nodes, n, d, bias);

% Set mini-batch parameters
eta_min = 1e-5; eta_max = 1e-1; nbatch = 100; 
ns = 5*45000/nbatch; lambda = 0.005; cycles = 2;

% Store mini-batch parameters
parameters.eta_min = eta_min; parameters.eta_max = eta_max;
parameters.ns = ns; parameters.nbatch = nbatch; 
parameters.cycles = cycles; parameters.lambda = lambda;

[Wstar,bstar, losstrain, lossvalid] = MiniBatch(trainX, trainY, validX, validY, W, b, parameters);

acc_test = ComputeAccuracy(testX,testy, Wstar, bstar); 

% Plotting evolution of loss function
plot(1:1:length(losstrain),losstrain, 1:1:length(lossvalid),lossvalid, 'LineWidth', 1.5)
grid on
xlabel('update step')
ylabel('loss')
ylim([0,3])
legend('training', 'validation', 'Fontsize',30)
title('Loss function - 9-layer network w/o batch normalization')
set(gca,'FontSize',30)
set(gcf, 'Position',  [100, 100, 1000, 700]);
saveas(gcf,[pwd '/Resultpics/9layerregular.png']);

%% Playground for batch norm

grad_test_X = trainX(1:20,1:3); grad_test_Y = trainY(:,1:3);


% Initialize parameters  
[K, ~] = size(grad_test_Y); % K = number of labels
[d,n] = size(grad_test_X);
nodes = [50 50 K]; % number of nodes in each layer
bias = 0; 
k = length(nodes);
lambda = 0;

rng(400)
[W,b,gamma,beta] = initParams(k, nodes, n, d, bias);

% Test classifier 
%[P,H,mu,v,s,shat] = EvalClassifier(grad_test_X, W, b, gamma, beta);
% [P,H] = EvalClassifier(trainX, W, b);

[gradb, gradW, gradgamma, gradbeta] = ComputeGradients(grad_test_X,grad_test_Y,W,b,lambda,...
    gamma, beta);

NetParams.W = W; NetParams.b = b; NetParams.use_bn = true;
NetParams.gammas = gamma; NetParams.betas = beta;

Grads = ComputeGradsNumSlow(grad_test_X, grad_test_Y, NetParams, lambda, 1e-05);
gradbRef = Grads.b; gradWRef = Grads.W;
gradgammaRef = Grads.gammas; gradbetaRef = Grads.betas;

% Check error
eps = 1e-10;
layer = 2;

errorb = norm(gradbRef{layer}  - gradb{layer})/max(eps,norm(gradb{layer})+norm(gradbRef{layer}));
errorW = norm(gradWRef{layer}  - gradW{layer})/max(eps,norm(gradW{layer})+norm(gradWRef{layer}));
errorgamma = norm(gradgammaRef{layer}  - gradgamma{layer})/max(eps,norm(gradgamma{layer})+norm(gradgammaRef{layer}));
errorbeta = norm(gradbetaRef{layer}  - gradbeta{layer})/max(eps,norm(gradbeta{layer})+norm(gradbetaRef{layer}));

%% Perform minibatch step
clc

% Initialize parameters  
[K, ~] = size(trainY); % K = number of labels
[d,n] = size(trainX);
%nodes = [50 30 20 20 10 10 10 10 K]; % number of nodes in each layer
nodes = [50 50 K];
bias = 0; 
k = length(nodes);

rng(400)
[W,b,gamma,beta] = initParams(k, nodes, n, d, bias);

% Set mini-batch parameters
eta_min = 1e-5; eta_max = 1e-1; nbatch = 100; 
ns = 5*45000/nbatch; lambda = 0.00651400417037734; cycles = 2;
lambda = 0.005;

% Store mini-batch parameters
parameters.eta_min = eta_min; parameters.eta_max = eta_max;
parameters.ns = ns; parameters.nbatch = nbatch; 
parameters.cycles = cycles; parameters.lambda = lambda;

[Wstar,bstar, losstrain, lossvalid, gammastar, betastar] = MiniBatch(trainX, trainY, validX, validY, ...
    W, b, parameters, gamma, beta);

acc_test = ComputeAccuracy(testX,testy, Wstar, bstar, gammastar, betastar); 

% Plotting evolution of loss function
plot(1:1:length(losstrain),losstrain, 1:1:length(lossvalid),lossvalid, 'LineWidth', 1.5)
grid on
xlabel('update step')
ylabel('loss')
ylim([0,3])
legend('training', 'validation', 'FontSize', 30)
title('Loss function - 3-layer network with batch normalization')
set(gca,'FontSize', 30)
set(gcf, 'Position',  [100, 100, 1000, 700]);
%saveas(gcf,[pwd '/Resultpics/9layerbn.png']);

%% Search for optimal lambda value 
clc 

% Initialize parameters  
[K, ~] = size(trainY); % K = number of labels
[d,n] = size(trainX);
%nodes = [50 30 20 20 10 10 10 10 K]; % number of nodes in each layer
nodes = [50 50 K];
bias = 0; 
k = length(nodes);

rng(400)
[W,b,gamma,beta] = initParams(k, nodes, n, d, bias);

% Lambda search parameters
l_min = log10(0.004); l_max = log10(0.006);
no_values = 8;
l = linspace(l_min, l_max, no_values);
lambda = 10.^l;

% Set mini-batch parameters
eta_min = 1e-5; eta_max = 1e-1; nbatch = 100; 
ns = 5*45000/nbatch; cycles = 2;

% Store mini-batch parameters
parameters.eta_min = eta_min; parameters.eta_max = eta_max;
parameters.ns = ns; parameters.nbatch = nbatch; 
parameters.cycles = cycles; parameters.lambda = lambda;


acc_valid = zeros(1, no_values);
acc_test = zeros(1, no_values);


disp('Performing grid search')
for i = 1:length(lambda)
    disp(['iteration ', int2str(i)])
    parameters.lambda = lambda(i); 
    [Wstar,bstar, losstrain, lossvalid, gammastar, betastar] = MiniBatch(trainX, trainY, validX, validY, ...
    W, b, parameters, gamma, beta);
    acc_valid(i) = ComputeAccuracy(validX,validy, Wstar, bstar, gammastar, betastar);
    acc_test(i) = ComputeAccuracy(testX,testy, Wstar, bstar, gammastar, betastar);
end

% Save results to file 
T = table(lambda',acc_valid',acc_test');
writetable(T,'gridSearch.txt','Delimiter',' ')  
type 'gridSearch.txt'

% best value found 
% lambda = 0.00719685673001152, valid accuracy 54.84%
% first search between 1e-5 and 1e-1
% lambda = 0.00707204172194891, accuracy 55.12 %, 53.72% test acc

%% Test sensitivity to initialization
clc

% Initialize parameters  
[K, ~] = size(trainY); % K = number of labels
[d,n] = size(trainX);
nodes = [50 30 20 20 10 10 10 10 K]; % number of nodes in each layer
%nodes = [50 50 K];
bias = 0; 
k = length(nodes);

sig = 1e-4;
bn = false;

rng(400)
[W,b,gamma,beta] = initParams(k, nodes, n, d, bias, sig);

% Set mini-batch parameters
eta_min = 1e-5; eta_max = 1e-1; nbatch = 100; 
ns = 5*45000/nbatch; lambda = 0.00707204172194891; cycles = 2;

% Store mini-batch parameters
parameters.eta_min = eta_min; parameters.eta_max = eta_max;
parameters.ns = ns; parameters.nbatch = nbatch; 
parameters.cycles = cycles; parameters.lambda = lambda;


if bn
    [Wstar,bstar, losstrain, lossvalid, gammastar, betastar] = MiniBatch(trainX, trainY, validX, validY, ...
        W, b, parameters, gamma, beta);
    acc_test = ComputeAccuracy(testX,testy, Wstar, bstar, gammastar, betastar); 
else
    [Wstar,bstar, losstrain, lossvalid] = MiniBatch(trainX, trainY, validX, validY, W, b, parameters);
    acc_test = ComputeAccuracy(testX,testy, Wstar, bstar); 
end

% Plotting evolution of loss function
plot(1:1:length(losstrain),losstrain, 1:1:length(lossvalid),lossvalid, 'LineWidth', 1.5)
grid on
xlabel('update step')
ylabel('loss')

legend('training', 'validation', 'FontSize', 30)
title({'Loss function - Constant initialzation w/o batch normalization', ['sig = ', num2str(sig), ' accuracy = ', num2str(acc_test)]})
set(gca,'FontSize', 30)
set(gcf, 'Position',  [100, 100, 1100, 700]);
saveas(gcf,[pwd '/Resultpics/9e4.png']);

%% Functions

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

function [J,loss] = ComputeCost(X, Y, Params, lambda)
% Computes cost and loss function given data 
% Inputs: 
% X = images 
% Y = labels of images X 
% Params = struct of data with 
%   W = cell of weight matrices 
%   b = cell of bias vectors 
%   gammas = cell of stretch vectors
%   beta = cell of shift vectors
%   use_bn = logical indicating whether to use batch norm. or not
% lambda = regularization
% 
% Returns
% J = value of cost function 
% loss = value of loss function 

W = Params.W; b = Params.b;

if Params.use_bn
    [P,~] = EvalClassifier(X,W,b,Params.gammas,Params.betas);
else
    [P,~] = EvalClassifier(X,W,b);  
end

[~,n] = size(X);

wSum = 0;
for l = 1:length(W)
    wSum = wSum + norm(W{l},'fro')^2;
end

if ismember(0,Y) == 0 % 1xn 
    idx = sub2ind(size(P), Y, (1:1:n)');
    loss = 1/n*sum(-log(P(idx)));
    J = loss + lambda*wSum;
else % One-hot ground truth label
    loss = 1/n*sum(-log(sum(Y.*P)));
    J = loss + lambda*wSum;
end

end

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

function acc = ComputeAccuracy(X, y, W, b, varargin)
% Inputs:
% X = matrix of images
% y = matrix of ground truth values of labels 
% W = cell of weight matrices for each layer
% b = cell of bias vectors for each layer
% (optional) gamma = cell of scale vectors for each layer
% (optional) beta = cell of shift vectors for each layer
%
% Returns:
% acc = scalar value of accuracy of data X using given parameters

if ~isempty(varargin)
    gamma = varargin{1};
    beta = varargin{2}; 
    [P,~] = EvalClassifier(X,W,b,gamma,beta);
else
    [P,~] = EvalClassifier(X,W,b);
end

[~,I] = max(P,[],1); % Finding all matching labels
match = I == y';
acc = sum(match)/length(y);

end

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


function G = BatchNormBackPass(G,S,mu,v,n)
% Function to batch normalization on backward pass
% Inputs:
% G = G_batch
% S = scores 
% mu = cell of means for each layer of unnormalized scores
% v = cell of variances for each layer of unnormalized scores
% n = number of images 
% 
% Returns:
% G = normalized G_batch

sigma1 = (v+eps).^(-0.5);
sigma2 = (v+eps).^(-1.5);

G1 = G.*(sigma1*ones(1,n));
G2 = G.*(sigma2*ones(1,n));

D = S - mu*ones(1,n);

c = (G2.*D)*ones(n,1);

G = G1 - (1/n)*(G1*ones(n,1))*ones(1,n) - (1/n)*D.*(c*ones(1,n));

end

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




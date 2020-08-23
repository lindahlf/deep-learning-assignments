clear all; close all; clc

% Import book
book_fname = 'goblet_book.txt';
fid = fopen(book_fname,'r');
book_data = fscanf(fid,'%c');
fclose(fid);

book_chars = unique(book_data); % all unique characters
K = length(book_chars); % number of unique characters
n_chars = length(book_data); % number of characters in book


% Initialize map containers
char_to_ind = containers.Map('KeyType', 'char', 'ValueType', 'int32');
int_to_char = containers.Map('KeyType', 'int32', 'ValueType', 'char');

% Add data to map containerss
for i = 1:K
    char_to_ind(book_chars(i)) = int32(i);
    int_to_char(int32(i)) = book_chars(i);
end

% Initialize parameters and store them in RNN
m = 100; % size of hidden state 
RNN.m = m; RNN.K = K;
RNN.b = zeros(m,1); 
RNN.c = zeros(K,1);

sig = 0.01; 
RNN.U = randn(m,K)*sig;
RNN.W = randn(m,m)*sig;
RNN.V = randn(K,m)*sig;

eta = 0.01; seq_length = 25; % length of output sequence
RNN.eta = eta; RNN.seql = seq_length;
epsilon = 1e-8; 



f = {'V','W', 'U', 'b', 'c'}; %relevant field names


%% Recall, trying to implement Forward and Backward pass functions 

% Variables for debugging
X_chars = book_data(1:seq_length);
Y_chars = book_data(2:seq_length+1);

X = hotEnc(X_chars, char_to_ind); % Input vector (hotenc of input sequence)
Y = hotEnc(Y_chars, char_to_ind); % Target output vector

h0 = zeros(m,1);

[A,H,O,P,L] = ForwardPass(X,Y,RNN,h0,true);

grads = BackwardPass(X,Y,RNN,P,H,A);
num_grads = ComputeGradsNum(X, Y, RNN, 1e-4);

% Check error
eps = 1e-10;

U_error = gradError(num_grads.U, grads.U, eps);
W_error = gradError(num_grads.W, grads.W, eps);
V_error = gradError(num_grads.V, grads.V, eps);

b_error = gradError(num_grads.b, grads.b, eps);
c_error = gradError(num_grads.c, grads.c, eps);



%% Testing functions 

% Variables for debugging
X_chars = book_data(1:seq_length);
Y_chars = book_data(2:seq_length+1);

X = hotEnc(X_chars, char_to_ind); % Input vector (hotenc of input sequence)
Y = hotEnc(Y_chars, char_to_ind); % Target output vector

h0 = zeros(m,1);

Ysyn = SynText(RNN,h0,X,seq_length); % Prediction of sequence
testText = ConvertText(Ysyn, int_to_char);

% Test ComputeLoss
[loss, P] = ComputeLoss(X, Y, RNN, h0, true);  

% Test gradients
num_grads = ComputeGradsNum(X, Y, RNN, 1e-4);
grads = ComputeGrads(X,Y,RNN,h0);

% Check error
eps = 1e-10;

U_error = gradError(num_grads.U, grads.U, eps);
W_error = gradError(num_grads.W, grads.W, eps);
V_error = gradError(num_grads.V, grads.V, eps);

b_error = gradError(num_grads.b, grads.b, eps);
c_error = gradError(num_grads.c, grads.c, eps);

%% AdaGrad 

hprev = zeros(m,1);
gamma = 0.9;  
smooth = 0;


% Initialize mvec 
for l = length(f)
    mvec.(f{l}) = zeros(size(RNN.(f{l})));
end

e = 1; % location in book 
iterations = 0;
smoothloss = 0;

while e < length(book_data) - seq_length - 1

    if e == 1
        hprev = zeros(m,1);
    end
     
    X_chars = book_data(e: e + seq_length - 1);
    Y_chars = book_data(e + 1 : e + seq_length);

    X = hotEnc(X_chars, char_to_ind); % Input vector (hotenc of input sequence)
    Y = hotEnc(Y_chars, char_to_ind); % Target output vector

    [A,H,O,P,~] = ForwardPass(X,Y,RNN,hprev,false);

    hprev = H(:,end);
    
    grads = BackwardPass(X,Y,RNN,P,H,A);


    for l = length(f)
        grads.(f{l}) = max(min(grads.(f{l}), 5), -5); % clip gradients
        mvec.(f{l}) = mvec.(f{l}) + grads.(f{l}).^2;
        RNN.(f{l}) = RNN.(f{l}) - (eta./sqrt(mvec.(f{l}) + epsilon)).*grads.(f{l});
    end
    
    if mod(e,10000) == 1
        L = ComputeLoss(Y,P);
        if e == 1
            smoothloss = L;
        else
            smoothloss = 0.999*smoothloss + 0.001*L;
        end
        
        disp(['Smooth loss: ', num2str(smoothloss)])
        disp([num2str((e/length(book_data))*100), ' %'])
        
    end
    
    e = e + seq_length;
    iterations = iterations + 1;
end

Ysyn = SynText(RNN,hprev, X(:,1), 200); % Prediction of sequence
testText = ConvertText(Ysyn, int_to_char);


%% Adagrad recall



epochs = 3;
e = 1; % location in book
iterations = floor(epochs*(n_chars-seq_length-1)/seq_length); % number of update steps
hprev = zeros(m,1);

% Initialize m in adagrad
for l = 1:length(f)
    m_ada.(f{l}) = zeros(size(RNN.(f{l})));
end

smooth_vec = zeros(1,iterations);

for i = 1:iterations

    if e > n_chars - seq_length - 1
        e = 1;
        hprev = zeros(m,1);
    end

    X_chars = book_data(e: e + seq_length - 1);
    Y_chars = book_data(e + 1 : e + seq_length);

    X = hotEnc(X_chars, char_to_ind); % Input vector (hotenc of input sequence)
    Y = hotEnc(Y_chars, char_to_ind); % Target output vector

    [A,H,O,P,L] = ForwardPass(X,Y,RNN,hprev,true);
    grads = BackwardPass(X,Y,RNN,P,H,A);

    if i == 1 
        smooth_loss = L;
    else
        smooth_loss = 0.999*smooth_loss + 0.001*L;
    end
    
    smooth_vec(i) = smooth_loss;

    for l = 1:length(f)
        grads.(f{l}) = max(min(grads.(f{l}), 5), -5); % Clip gradients

        % Adagrad step
        m_ada.(f{l}) = m_ada.(f{l}) + grads.(f{l}).^2;
        RNN.(f{l}) = RNN.(f{l}) - (eta./sqrt(m_ada.(f{l})+epsilon)).*grads.(f{l}); 
    end    

    
   
    if mod(i,10000) == 1
        Ysyn = SynText(RNN,hprev, X(:,1), 200);
        smooth_loss
        ConvertText(Ysyn, int_to_char)
        disp([num2str((i/iterations)*100), ' %'])
    end    
    
    hprev = H(:,end);
    e = e + seq_length; 
    
    
end



%% 

plot(itvec,smooth_vec)
title('Smooth loss as a function of iterations', 'FontSize', 18)
xlabel('Iterations', 'FontSize', 18)
ylabel('Smooth loss', 'FontSize', 18)






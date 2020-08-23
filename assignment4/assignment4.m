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


%% Testing gradients

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


%% Adagrad 

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



%% Plot smooth loss function 

plot(itvec,smooth_vec)
title('Smooth loss as a function of iterations', 'FontSize', 18)
xlabel('Iterations', 'FontSize', 18)
ylabel('Smooth loss', 'FontSize', 18)

%% Functions 

function hotText = hotEnc(charText, char_to_ind)
% Converts a sequence of characters to a one-hot representation

% Input:
% charText = sequence of characters
% char_to_ind = Map of character paired with correct integer


% Returns: 
% hotText = one-hot encoded representation of input charText

hotText = zeros(length(char_to_ind),length(charText));

for i = 1:length(charText)
    hotText(char_to_ind(charText(i)),i) = 1;
end

end

function Y = SynText(RNN, h0, x0, n)
% Function to synthesize text sequence
% 
% Inputs:
% RNN = struct containing network parameters
% h0 = initial hidden state
% x0 = initial character
% n = sequence length
% 
% Returns: 
% Y = matrix with one-hot encoding of sequence

% Import parameters
W = RNN.W; U = RNN.U; b = RNN.b; V = RNN.V; 
c = RNN.c; K = RNN.K;

h = h0; x = x0;

Y = zeros(K,n);

for t = 1:n
    
    [~,H,~,P,~] = ForwardPass(x,Y,RNN,h,false);
    % Select new character based on highest probability
    h = H(:,end);
    cp = cumsum(P); 
    r = rand;
    ixs = find(cp-r > 0);
    ii = ixs(1);
    
    Y(ii,t) = 1; % 
    x = Y(:,t); % xnext
end


end

 function charText = ConvertText(hotText, int_to_char)
% Function that converts one hot encoded text to character 

% Inputs: 
% hotText = one hot encoded text sequence
% int_to_char = Map of int paired with correct character

% Returns: 
% charText = charcter representation of hotText

[~,idx] = max(hotText); 
charText = char();

for i = 1:length(idx)
    curr = int_to_char(idx(i));
    charText = append(charText,curr);
end


end

function [A,H,O,P,L] = ForwardPass(X,Y,RNN,hprev,loss)
% Performs forward pass of a the network given its parameters and a sequence of text

% Inputs 
% X = sequence of text
% Y = targeted output text
% RNN = struct with network parameters 
% hprev = previous hidden layer
% loss = boolean indicating if loss is to be computed

% Returns: 
% The intermediary matrices A,H, and O
% P = matrix of probabilities for each character for each step
% L = value of loss function

W = RNN.W; U = RNN.U; V = RNN.V;
b = RNN.b; c = RNN.c; 
tau = size(X,2); m = round(RNN.m); K = round(RNN.K);

A = zeros(m,tau); 
H = zeros(m,tau+1);
H(:,1) = hprev;
O = zeros(K,tau);
P = zeros(K,tau);



for t = 1:tau
    A(:,t) = W*H(:,t) + U*X(:,t)+ b;
    H(:,t+1) = tanh(A(:,t));
    O(:,t) = V*H(:,t+1) + c;
    P(:,t) = softmax(O(:,t));
end

if loss == true
    L = ComputeLoss(Y,P);
else 
    L = 1; 
end
    

end

function grads = BackwardPass(X,Y,RNN,P,H,A)
% Function that performs the backward pass given the network and some text

% Inputs:
% X = sequence of text
% Y = targeted output text
% RNN = struct with network parameters 
% P = matrix of probabilities for each character for each step
% H = hidden layer 
% A = Intermediate matrix A

% Returns: 
% grads = struct with all gradients 



m = RNN.m; tau = RNN.seql; V = RNN.V; W = RNN.W;

dO = -(Y-P)';
dV = dO'*H(:,2:end)';

dH = zeros(tau,m);
dH(end,:) = dO(end,:)*V;

dA = zeros(tau,m);
dA(end,:) = dH(end,:)*diag(1-tanh(A(:,end)).^2);

for t = tau-1:-1:1
    dH(t,:) = dO(t,:)*V + dA(t+1,:)*W;
    dA(t,:) = dH(t,:)*diag(1-tanh(A(:,t)).^2);
end

dW = dA'*H(:,1:end-1)';
dU = dA'*X';

db = sum(dA',2);
dc = sum(dO',2);

grads.V = dV;
grads.W = dW;
grads.U = dU;
grads.b = db;
grads.c = dc;
end

function L = ComputeLoss(Y,P)
% Computes loss function for a given sequence of text 

% Inputs: 
% Y = labels for predicted sequence
% P = probability matrix 

% Returns:
% L = value of loss function

L = sum(-log(sum(Y.*P)));

end








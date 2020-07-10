clear all; close all; clc

% Import book
book_fname = 'goblet_book.txt';
fid = fopen(book_fname,'r');
book_data = fscanf(fid,'%c');
fclose(fid);

book_chars = unique(book_data); % all unique characters
K = length(book_chars); % number of unique characters

% Initialize map containers
char_to_ind = containers.Map('KeyType', 'char', 'ValueType', 'int32');
int_to_char = containers.Map('KeyType', 'int32', 'ValueType', 'char');

% Add data to map containerss
for i = 1:K
    char_to_ind(book_chars(i)) = int32(i);
    int_to_char(int32(i)) = book_chars(i);
end

% Initialize parameters and store them in RNN
m = 5; % size of hidden state 
RNN.m = m; RNN.K = K;
RNN.b = zeros(m,1); 
RNN.c = zeros(K,1);

sig = 0.01; 
RNN.U = randn(m,K)*sig;
RNN.W = randn(m,m)*sig;
RNN.V = randn(K,m)*sig;

eta = 0.1; seq_length = 25; % length of output sequence
RNN.eta = eta; RNN.seql = seq_length;



%% Testing functions 

% Variables for debugging
X_chars = book_data(1:seq_length);
Y_chars = book_data(2:seq_length+1);

X = hotEnc(X_chars, char_to_ind); % Input vector (hotenc of input sequence
Y = hotEnc(Y_chars, char_to_ind); % Target output vector

h0 = zeros(m,1);

Ysyn = SynText(RNN,h0,X,seq_length); % Prediction of sequence
testText = ConvertText(Ysyn, int_to_char);

% Test ComputeLoss
[loss, P] = ComputeLoss(X,Y,RNN,h0);  

% Test gradients
num_grads = ComputeGradsNum(X, Y, RNN, 1e-4);
grads = ComputeGrads(X,Y,RNN, h0);

% Check error
eps = 1e-10;

U_error = gradError(num_grads.U, grads.U, eps);
W_error = gradError(num_grads.W, grads.W, eps);
V_error = gradError(num_grads.V, grads.V, eps);



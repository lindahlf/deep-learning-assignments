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


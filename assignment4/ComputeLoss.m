function L = ComputeLoss(X,Y,RNN,h)
% Computes loss function for a given sequence of text using current
% network parameters

% Inputs:
% X = one-hot encoded representation of input seqence 
% Y = one-hot encoded representation of targeted ouput sequence
% RNN = struct of network parameters
% h = hidden state

% Returns: 
% L = loss function value 

% Import parameters
W = RNN.W; U = RNN.U; b = RNN.b; V = RNN.V; 
c = RNN.c; K = RNN.K;

a = W*h + U*X + b;
h = tanh(a);
o = V*h + c;
p = softmax(o);

L = sum(-log(sum(Y.*p)));
    

end


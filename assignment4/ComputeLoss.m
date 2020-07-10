function [L, varargout] = ComputeLoss(X,Y,RNN,h0)
% Computes loss function for a given sequence of text using current
% network parameters

% Inputs:
% X = one-hot encoded representation of input seqence 
% Y = one-hot encoded representation of targeted ouput sequence
% RNN = struct of network parameters
% h = hidden state

% Returns: 
% L = loss function value 
% p = 
% a 
% h

% Import parameters
W = RNN.W; U = RNN.U; b = RNN.b; V = RNN.V; 
c = RNN.c; K = round(RNN.K); seql = round(RNN.seql); m = round(RNN.m);

% a = W*h + U*X + b;
% h = tanh(a);
% o = V*h + c;
% p = softmax(o);

% Idiot approach

a = zeros(m,seql); h = zeros(m,seql);
o = zeros(K,seql); p = zeros(K,seql);


for i = 1:seql
    a(:,i) = W*h(:,i) + U*X(:,i) + b;
    h(:,i) = tanh(a(:,i));
    o(:,i) = V*h(:,i) + c;
    p(:,i) = softmax(o(:,i));
end
    


L = sum(-log(sum(Y.*p)));
varargout{1} = p;
varargout{2} = a; 
varargout{3} = h;
    

end


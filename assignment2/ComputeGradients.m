function [dW1, dW2, db1, db2] = ComputeGradients(X,Y,W,b)

[P,H] = EvalClassifier(X,W,b);

[~,n] = size(Y);

G = -(Y-P);

db2 = (1/n)*G;
dW2 = (1/n)*G*H';

g = g*W2;
g = g*diag(

end


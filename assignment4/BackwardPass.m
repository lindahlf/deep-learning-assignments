function grads = BackwardPass(X,Y,RNN,P,H,A)
% Function that performs the backward pass given the network and some text

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


function grads = ComputeGrads(X,Y,RNN,H)
% Compute gradients for RNN

% Import parameters
W = RNN.W; U = RNN.U; b = RNN.b; V = RNN.V; 
c = RNN.c; K = RNN.K; seql = RNN.seql; m = RNN.m;

[~,P,a,H] = ComputeLoss(X,Y,RNN,H);

G = -(Y-P)';

grads.V = G'*H';

dh = zeros(seql, m); % dL/dh
da = zeros(seql, m); % dL/da

dh(end,:) = G(end,:)*V;
da(end,:) = dh(end,:)*diag(1 - tanh(a(:,end)).^2);

for i = seql-1:-1:1
    dh(i,:) = G(i,:)*V + da(i+1,:)*W;
    da(i,:) = dh(i,:)*diag(1 - tanh(a(:,i)).^2);
end

dummy_H = zeros(m,seql); 
dummy_H(:, 2:end) = H(:, 2:end); 

grads.W = da'*dummy_H'; 

grads.U = da'*X';




end


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


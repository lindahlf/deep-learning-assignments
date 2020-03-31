function [W,b] = initParams(m, d, K, bias)
% Initializing the parameters for the network
% Parameters:
% m = number of nodes 
% d = dimension of the images
% K = number of labels 
% bias = 1 or 0. 0 = no bias, 1 = bias. 
% Returns:
% W1, size = mxd
% b1, size = mx1
% W2, size = Kxm
% b2, size = Kx1
% W1 and W2 stored in cell W
% b1 and b2 stored in cell b



W1 = (1/sqrt(d)).*randn(m,d);
W2 = (1/sqrt(m)).*randn(K,m);

if bias == 1
    b1 = (1/sqrt(d)).*randn(m,1);
    b2 = (1/sqrt(m)).*randn(K,1);
else
    b1 = zeros(m,1);
    b2 = zeros(K,1);
end

W = {W1,W2};
b = {b1,b2};

end 




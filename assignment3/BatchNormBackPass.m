function G = BatchNormBackPass(G,S,mu,v,n)
% Function to batch normalization on backward pass
% Inputs:
% G = G_batch
% S = scores 
% mu = cell of means for each layer of unnormalized scores
% v = cell of variances for each layer of unnormalized scores
% n = number of images 
% 
% Returns:
% G = normalized G_batch

sigma1 = (v+eps).^(-0.5);
sigma2 = (v+eps).^(-1.5);

G1 = G.*(sigma1*ones(1,n));
G2 = G.*(sigma2*ones(1,n));

D = S - mu*ones(1,n);

c = (G2.*D)*ones(n,1);

G = G1 - (1/n)*(G1*ones(n,1))*ones(1,n) - (1/n)*D.*(c*ones(1,n));

end


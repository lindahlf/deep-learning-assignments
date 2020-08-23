function L = ComputeLoss(Y,P)
% Computes loss function for a given sequence of text 

L = sum(-log(sum(Y.*P)));

end


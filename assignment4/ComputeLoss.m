function L = ComputeLoss(Y,P)
% Computes loss function for a given sequence of text 

% Inputs: 
% Y = labels for predicted sequence
% P = probability matrix 

% Returns:
% L = value of loss function

L = sum(-log(sum(Y.*P)));

end


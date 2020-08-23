function error = gradError(num_grad, grad, eps)
% Function to compute error between analytical 
% and numerical gradient

% Inputs: 
% num_grad = numerical gradient
% grad = analytical gradient 
% eps = relative size of error 

% Returns: 
% error = error between numerical and analytical gradient

error = norm(grad - num_grad)/max(eps,norm(grad)+norm(num_grad)); 


    

end


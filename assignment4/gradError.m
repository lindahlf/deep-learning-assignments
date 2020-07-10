function error = gradError(num_grad,grad, eps)
% Function to compute error between analytical 
% and numerical gradient

error = norm(grad - num_grad)/max(eps,norm(grad)+norm(num_grad)); 

end


function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
J = 0;
temp0 = 0;
temp1 = 0;
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    %for i=1:m
    %    temp0 = temp0 + ((X(i,:)*theta)-y(i,:));
    %    temp1 = temp1 + ((X(i,:)*theta)-y(i,:))*X(i,2);
    %end
    %temp0 = theta(1) - (alpha)*(1/m)*temp0;
    %temp1 = theta(2) - (alpha)*(1/m)*temp1;
    %theta(1) = temp0;
    %theta(2) = temp1;


    %J = computeCost(X,y,theta)
    h = X*theta;
    errors = h - y;
    theta = theta - ((X'*errors)*(1/m)*(alpha))


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end

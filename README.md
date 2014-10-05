coursera-ml-007
===============
mulit-variable linear regression

cost functions
J = (sum(((X * theta) - y).^2))/(2*m); OR
J = (((X*theta) - y)'*((X*theta) - y))/(2*m);

gradient descent (choose aplpha, many iterations, simultaneously update)
theta()=theta()-((alpha/m)*X'*(X*theta()-y));

normal equation (<10000 features, compute expensive (O(n^3)))
theta=pinv(X’*X)*(X’*y);

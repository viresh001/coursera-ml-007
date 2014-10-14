coursera-ml-007
===============
mulit-variable linear regression

hθ(x) = θ'x;

cost functions
J = (sum(((X * theta) - y).^2))/(2*m); OR
J = (((X*theta) - y)'*((X*theta) - y))/(2*m);

gradient descent (choose aplpha, many iterations, simultaneously update)
theta()=theta()-((alpha/m)*X'*(X*theta()-y));

normal equation (<10000 features, compute expensive (O(n^3)))
theta=pinv(X’*X)*(X’*y);

logistic regression

hθ(x) = 1/(1+e(-θ'x)); 

cost function
J(θ) = -1/m*sum(y*log(hθ(x)) + (1-y)log(1-hθ(x)))

gradient descent (choose aplpha, many iterations, simultaneously update)
theta()=theta()-((alpha/m)*X'*(hθ(x)-y)); 


coursera-ml-007
===============
LINEAR REGRESSION

Cost Functions
J = (sum(((X * theta) - y).^2))/(2*m); OR
J = (((X*theta) - y)'*((X*theta) - y))/(2*m);

Gradient Descent
theta()=theta()-((alpha/m)*X'*(X*theta()-y));

Normal Equation
theta=pinv(X’*X)*(X’*y);

coursera-ml-007

MULTI-VARIABLE LINEAR REGRESSION

hθ(x) = θ'x;

cost functions

J = (sum(((X * theta) - y).^2))/(2*m); OR
J = (((X*theta) - y)'*((X*theta) - y))/(2*m);

gradient descent (choose aplpha, many iterations, simultaneously update)

theta()=theta()-((alpha/m)*X'*(X*theta()-y));

normal equation (<10000 features, compute expensive (O(n^3)))

theta=pinv(X’*X)*(X’*y);

LOGISTIC REGRESSION

hθ(x) = 1/(1+e(-θ'x)); 

decision boundary

θ'x == 0

cost function

J(θ) = -1/m*sum(y*log(hθ(x)) + (1-y)log(1-hθ(x)))

code

ht = sigmoid(X*theta);
J = (1/m)*((-y'*log(ht)) - (1-y')*log(1-ht));

gradient descent (choose aplpha, many iterations, simultaneously update)

theta()=theta()-((alpha/m)*X'*(hθ(x)-y));

code

grad = (1/m)*(X'*(ht-y));

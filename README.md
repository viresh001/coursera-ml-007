coursera-ml-007 summary

matrix & vector multiplication

A[m,n]*B[n,o] = C[m,o] - multiply each m (row) in A with each o (column) in B (for vectors, o = 1)

get array row maximuns/minimuns: [max/min, max/min_index] = max/min(A,[],2);

get array column maximuns/minimuns: [max/min, max/min_index] = max/min(A,[],1);

MULTI-VARIABLE LINEAR REGRESSION

hθ(x) = θ'x;

x = features & θ = paramaters

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

REGULARIZED LOGISTIC REGRESSION

code

theta_reg = [0;theta(2:size(theta))];

ht = sigmoid(X*theta);

cost function

J = ((1/m)*((-y'*log(ht)) - (1-y')*log(1-ht))) + (lambda/(2*m))*(theta_reg'*theta_reg);

gradient descent

grad = ((1/m)*(X'*(ht-y))) + ((lambda/m)*theta_reg);

NEURAL NETWORKS - REPRESENTATION

feedforward propagation

n = 0: Input Layer

n = last layer:  Output Layer

n = other:  Hidden Layers

for any layer n (where n > 1):

z(n) = θ(n-1)'*a(n-1);

a(n) = g(z(n)); 

add column of 1s for each n to last layer -1

a(n) = [ones(m, 1) a(n)];

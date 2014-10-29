coursera-ml-007 notes

matrix & vector multiplication
A[m,n]*B[n,o] = C[m,o] - multiply each m (row) in A with each o (column) in B (for vectors, o = 1)

get array row maximuns/minimuns: [max/min, max/min_index] = max/min(A,[],2);

get array column maximuns/minimuns: [max/min, max/min_index] = max/min(A,[],1);

unroll arrays:  Array3 = [Array1(:) ; Array2(:)]; and use reshape to rebuilt

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
J(θ) = (-1/m)*sum(y*log(hθ(x)) + (1-y)log(1-hθ(x)))

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

a(n) = "activation" in layer n
θ(n) = matrix of weights (parameters) controlling function mapping from layer n to layer (n+1)

if a network has s(n) units in layer j and s(n+1) units in layer n+1, then θ(n) will be of dimension (s(n+1)) X (s(n) + 1)

feedforward propagation

n = 0: Input Layer
n = last layer:  Output Layer
n = other:  Hidden Layers

for any layer n (where n > 1):
z(n) = θ(n-1)'*a(n-1);
a(n) = g(z(n)); 
add column of 1s for each n to last layer -1
a(n) = [ones(m, 1) a(n)];

recode y to logical vector:
y contains 0 & 1 data
A_logical = zeros(m, num_labels);
for i = 1:m
  y_logical(i, y(i,1)) = 1;
end

neural networks cost function

sum across examples(m rows) and classes(K)
J = (1/m)*sum(sum((-y_logical).*log(hx) - (1-y_logical).*log(1-hx)));

add regularization to cost function - for each theta
theta1_reg = (lambda/(2*m))*sum(sum((Theta1(:, 2 :end).^2)));
theta2_reg = (lambda/(2*m))*sum(sum((Theta2(:, 2 :end).^2)));
theta3_reg = ...

J = J + theta1_reg + theta2_reg + ...

%back propagation ("error" per layer)
delta_OutputLayer = hx - y_logical;

delta at all hidden layers (l=2,... OutputLayer-1) (IGNORE BIAS UNIT)
delta(l) = delta(l+1)*Theta(l);
delta(l) = delta(l)(:,2:end).*sigmoidGradient(z(l));
theta_gradient(l) = theta_gradient(l) + ((1/m)*(delta(l+1)'*a(l)));

%add regularization to back propagation (IGNORE BIAS UNIT)
theta_reg(l) = theta_gradient(l);
theta_reg(l)(:,1) = 0;
theta_gradient(l) = theta_gradient(l) + ((lambda/m)*theta_reg(l));

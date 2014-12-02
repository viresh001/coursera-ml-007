coursera-ml-007 notes

matrix & vector multiplication
A[m,n]*B[n,o] = C[m,o] - multiply each m (row) in A with each o (column) in B (for vectors, o = 1)

get array row maximuns/minimuns: [max/min, max/min_index] = max/min(A,[],2);

get array column maximuns/minimuns: [max/min, max/min_index] = max/min(A,[],1);

unroll arrays:  Array3 = [Array1(:) ; Array2(:)]; and use reshape to rebuilt

ALGORITHMS
Supervised learning algorithms needs labeled examples (x,y)
unsupervised learning algorithms need only the input (x)

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
y_logical = zeros(m, num_labels);
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

random initialization of parameters for symmetry breaking
ESPILON_INIT = sqrt(6)/sqrt(L_in+L_out);
W = rand(L_out, 1+L_in)*2*ESPILON_INIT-ESPILON_INIT;
L_in = incoming layer connections
L_out = outgoing connections

back propagation ("error" per layer)
delta_OutputLayer = hx - y_logical;

delta at all hidden layers (l=2,... OutputLayer-1) (IGNORE BIAS UNIT)
delta(l) = delta(l+1)*Theta(l);
delta(l) = delta(l)(:,2:end).*sigmoidGradient(z(l));
theta_gradient(l) = theta_gradient(l) + ((1/m)*(delta(l+1)'*a(l)));

add regularization to back propagation (IGNORE BIAS UNIT)
theta_reg(l) = theta_gradient(l);
theta_reg(l)(:,1) = 0;
theta_gradient(l) = theta_gradient(l) + ((lambda/m)*theta_reg(l));

use theta_gradient[] as input to methods like fmincg to learn a good set of parameters

ADVICE FOR APPLYING MACHINE LEARNING

Model Section and Train/Validation/Test Sets
- Split data into 3 sets (RANDOM training set (M(train) 60%, RANDOM cross validtion set (M(cv) 20%, RANDOM test set (M(test)) 20%)
- Calculate Jtrain(θ), Jcv(θ), Jtest(θ)
- Pick model with lowest Jcv(θ)

Choosing regularization (lamba)
- try a set of lamba values
- Learn θ from training set and minimize J(θ)
- Compute test set Jcv(θ), pick lamba with smallest Jcv(θ)
- small lamba -> high variance (overfitting) in Jcv(θ)
- large lamba -> high bias (underfitting) in Jcv(θ)

Diagnosing Bias vs Variance
- High bias (underfit) problem:  Jcv(θ) AND Jtrain(θ) both HIGH
- High variance (overfit) problem Jcv(θ) is HIGH and Jtran(θ) is LOW

Learning Curves
- Plot Jtrain(θ) and Jcv(θ) vs traing set size (m)
- as m goes from small->large, Jtrain(θ) goes from small->large
- as m goes from small->large, Jcv(θ) goes from large->smaller

- High Bias (Underfitting)
  - as m goes from small->large, Jcv(θ) goes from large->smaller but flattens out quickly (best possible fit)
  - as m goes from small->large, Jtrain(θ) goes from small->large quickly, ~= Jcv(θ)
  - high value of Jtrain(θ)~=Jcv(θ)
  - getting more training data, DOES NOT help
  - try getting additional features 
  - try adding polynomila features 
  - try decreasing lamba 

 - High Variance (Overfitting)
  - as m goes from small->large, Jtrain(θ) goes from small->increasess (slowly) 
  - as m goes from small->large, Jcv(θ) goes from large->small
  - Jcv(θ) - Jtrain(θ) =  significant
  - getting more training data, DOES help (converges Jvc & Jtrain)
  - try smaller set of features
  - try increasing lamba

- For neutal networks try with > 1 hidden layers and check for lowest Jcv(θ)

MACHINE LEARNING SYSTEM DESIGN

Recommended Approach
- Start with a simple algorithm that you can implement quick and test on cross-validation data
- Plot learning curves to identify bias (underfitting)/variance (overfitting)
- Error Analysis - manually examine the examples in cross validation set for errors
- Error Metrics - single real number evaluate/validate algorithms  - compare cross validation errors
- Error Metrics for Skewed Classes:
  - Accuracy = (true positives + true negatives) / (total examples)
  - Precision (true positives/(true positives + false positives))
  - Recall (true positvies/(true positives + fale negatives))
  - Use high precision and high recall
  - precision/recall comparison:
    - Calculate F(1)-Score: 2*((P*R)/(P+R)) on cross validation set and pick MAX value
    - Recall is the percentage of true positives in relation to both true positives and false negatives. In other words, the       percentage of items containing malicious content that where marked as containing malicious content. 
    - Precision is the percentage of true positives in relation to both true positives and false positives. In other words,        the percentage of log events that were marked as containing malicious content that actually contained malicious              content.

Large data rationale (when is getting larger data sets better?)
- Given the input x, can a human expert confidently predict y?
- Use logistic/linear regression with many features; neural network with many hidden units
  - produces low bias algoritms (Jtrain will be small)
  - very large training set (unlikely to overfit) - produces low variance
  - Jtrain(θ)~=Jtest(θ)


MULTIVARIATE GAUSSIAN ALGO CODE

%calculate mean for each feature
mu = (1/m) .* sum(X);

%calculate variance for each feature
%work around broadcasting warning
mu1 = mu' * ones(1,m);
sigma2 = (1/m) .* sum((X - mu1') .^2);

X = bsxfun(@minus, X, mu(:)');
p = (2 * pi) ^ (- k / 2) * det(Sigma2) ^ (-0.5) * exp(-0.5 * sum(bsxfun(@times, X * pinv(Sigma2), X), 2));

COLLABORATIVE FILTERING ALGO CODE

%calculate the cost function (J) but only where R ==1
J = (1/2)*sum((((X*Theta') - Y).^2)(R==1));

X_grad = (((X*Theta') - Y) .* R)*Theta;
Theta_grad = (((X*Theta') - Y) .* R)'*X;

%calculate regularization for Theta and X
reg_Theta = (lambda/2)*sum(sum((Theta .^ 2)));
reg_X = (lambda/2)*sum(sum((X .^ 2)));

J = J + reg_Theta + reg_X;

%calculate regularization for Theta_grad and X_grad
reg_Theta_grad = lambda*Theta;
reg_X_grad = lambda*X;

X_grad = X_grad + reg_X_grad;
Theta_grad = Theta_grad + reg_Theta_grad;
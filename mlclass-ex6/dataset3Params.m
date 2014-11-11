function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

range = [0.01 0.03 0.1 0.3 1 3 10 30];

min_error = inf;
C_final = C;
sigma_final = sigma;

% iterate each value of range 
for C = range
	for sigma = range
		%short hand for the gaussianKernel function @(x1, x2) get fed to gaussianKernel in scmTrain
	    gaussianKernelFunction = @(x1, x2) gaussianKernel(x1, x2, sigma);
		model= svmTrain(X, y, C, gaussianKernelFunction);
	    predict =  svmPredict(model, Xval, yval);
	    cur_error =  mean(double(predict~=yval));
	    if(cur_error< min_error)
	    	C_final = C;
	        sigma_final = sigma;
	        min_error = cur_error;
	    end
	 end
end

C = C_final;
sigma = sigma_final;







% =========================================================================

end

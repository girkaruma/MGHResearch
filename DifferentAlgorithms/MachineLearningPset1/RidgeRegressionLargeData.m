function RidgeRegressionLargeData()
    X = csvread('x_test.csv');
    Y = csvread('y_test.csv');
    lambda = 1;
    optimalW = largeRegression(X, Y, lambda);
    computeError(optimalW, X, Y)
end
function optimalW = largeRegression(X, Y, lambda)
    xSize = size(transpose(X)*X);
    identity = eye(xSize(2), xSize(1));
    optimalW = inv(lambda*identity + transpose(X)*X)*transpose(X)*Y;
end
function error = computeError(w, X, Y)
    error = 0;
    yFunc = w'*X';
    for i=1:length(Y)
        error = error + (yFunc(i) - Y(i))^2;
    end
end

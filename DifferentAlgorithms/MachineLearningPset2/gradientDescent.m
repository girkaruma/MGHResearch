function minimum = gradientDescent(initialGuess, stepSize, convergenceThreshold)
    %disp('init');
    [laterGuess, initialGuess] = gradientDescentCalculation(initialGuess, stepSize);
    n = 0;
    while abs(objectiveFunction(laterGuess) - objectiveFunction(initialGuess)) >= convergenceThreshold
        initialGuess = laterGuess;
        [laterGuess, initialGuess] = gradientDescentCalculation(initialGuess, stepSize);
        n = n + 1;
    end
    minimum = laterGuess;
    objectiveFunction(minimum)
end

function [laterGuess, initialGuess] = gradientDescentCalculation(initialGuess, stepSize)
    laterGuess = initialGuess - stepSize * gradientFunction(initialGuess);
end

function objValue = objectiveFunction(initialGuess)
    data = importdata(strcat('data/data_','stdev1','_train.csv'));
    X = data(:,1:2);
    Y = data(:,3);
    w = initialGuess(1:2);
    w0 = initialGuess(3);
    lambda = 0;
    objValue = logistic_regression(X, Y, w, w0, lambda);
end

function error_logistic_regression = logistic_regression(X, Y, w, w0, lambda)
    error_logistic_regression = 0;
    for i = 1:length(Y)
        %size(Y(i))
        %size(X(i,:))
        size(w);
        %size(w0)
        error_i = log10(1 + exp(-Y(i)*(X(i, :)*w + w0)));
        error_logistic_regression = error_logistic_regression + error_i;
    end
    error_logistic_regression = error_logistic_regression + lambda * (w'*w);
    %pause;
end

function gradient = gradientFunction(initialGuess)
    gradient = computeGradientCentralDifferences(initialGuess', 0.01);
end
function gradient = computeGradientCentralDifferences(point, stepSize)
    gradient = zeros(length(point), 1);
    for i=1:length(point)
        pointAdd = transpose(point);
        pointSubtract = transpose(point);
        pointAdd(i, 1) = pointAdd(i, 1) + 0.5 * stepSize;
        pointSubtract(i, 1) = pointSubtract(i, 1) - 0.5 * stepSize;
        %size(objectiveFunction(pointAdd));
        %objectiveFunction(pointAdd);
        gradient(i, 1) = (objectiveFunction(pointAdd) - objectiveFunction(pointSubtract))/stepSize;
    end
end

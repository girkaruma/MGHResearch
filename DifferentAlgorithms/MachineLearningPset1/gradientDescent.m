function minimum = gradientDescent(initialGuess, stepSize, convergenceThreshold)
    [laterGuess, initialGuess] = gradientDescentCalculation(initialGuess, stepSize);
    n = 0;
    while abs(objectiveFunction(laterGuess) - objectiveFunction(initialGuess)) >= convergenceThreshold
        initialGuess = laterGuess;
        [laterGuess, initialGuess] = gradientDescentCalculation(initialGuess, stepSize);
        n = n + 1;
        laterGuess;
    end
    n;
    minimum = laterGuess;
end

function [laterGuess, initialGuess] = gradientDescentCalculation(initialGuess, stepSize)
    laterGuess = initialGuess - stepSize * gradientFunction(initialGuess)';
end

function objValue = objectiveFunction(initialGuess)
    %x = initialGuess(1);
    %y = initialGuess(2);
    %objValue = x^2;
    %objValue = (x+4)*(x-3)*(x+1)*(x-2);
    objValue = LeastAbsoluteDeviation(initialGuess);
%     M = 5;
%     data = importdata('curvefitting.txt');
%     X = data(1,:);
%     Y = data(2,:);
%     objValue = computeSumOfSquaresError(X, Y, M, initialGuess);
end

function gradient = gradientFunction(initialGuess)
    %x = initialGuess(1);
    %y = initialGuess(2);
    gradient = computeGradientCentralDifferences(initialGuess, 0.01);
    %gradient(1) = 4*x^3-30*x+10;
    %gradient(2) = 0;
    %gradient = 2*x;
end
function gradient = computeGradientCentralDifferences(point, stepSize)
    gradient = zeros(length(point), 1);
    for i=1:length(point)
        pointAdd = transpose(point);
        pointSubtract = transpose(point);
        pointAdd(i, 1) = pointAdd(i, 1) + 0.5 * stepSize;
        pointSubtract(i, 1) = pointSubtract(i, 1) - 0.5 * stepSize;
        gradient(i, 1) = (objectiveFunction(pointAdd) - objectiveFunction(pointSubtract))/stepSize;
    end
end
function error = computeSumOfSquaresError(X, Y, M, w)
    error = 0;
    N = length(X);
    for i=1:N
        hypothesis = 0;
        for j=0:M
            hypothesis = hypothesis + w(j+1)*basisFunction(X(i), j);
        end
        error = error + (Y(i) - hypothesis)^2;
    end
    error = 0.5 * error;
end
function val = basisFunction(x, m)
    val = x^m;
end
% function ladDerivative = LADDerivative(initialGuess)
%     data = importdata('regress_train.txt');
%     X = data(1,:);
%     Y = data(2,:);
%     N = length(Y);
%     M = 3;
%     lambda = 0;
%     ladDerivative = lambda*initialGuess;
%     for i=1:N
%         phiN = zeros(M+1, 1);
%         for j=0:M
%             phiN(j+1,1) = X(i)^j; 
%         end
%         if (Y(i) - transpose(initialGuess)*phiN' > 0)
%             ladDerivative = ladDerivative - 0.5 * phiN';
%         else
%             ladDerivative = ladDerivative + 0.5 * phiN';
%         end
%     end
% end

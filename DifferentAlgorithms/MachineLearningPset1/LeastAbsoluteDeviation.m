function realError = LeastAbsoluteDeviation(w)
    data = importdata('regress_validate.txt');
    X = data(1,:);
    Y = data(2,:);
    %X = [0 1 2 3];
    %Y = [0 1 2 3];
    M = 3;
    %lambda = 2.71828^(0-18);
    lambda = 0.01;
    realError = computeLeastAbsoluteDeviation(X, Y, M, w, lambda);
end
function error = computeLeastAbsoluteDeviation(X, Y, M, w, lambda)
    error = 0;
    N = length(X);
    for i=1:N
        hypothesis = 0;
        for j=0:M
            hypothesis = hypothesis + w(j+1)*basisFunction(X(i), j);
        end
        error = error + abs(Y(i) - hypothesis);
    end
    weightMagnitude = 0;
    for i=1:length(w)
        weightMagnitude = weightMagnitude + w(i)^2;
    end
    error = error + lambda * weightMagnitude;
    error = 0.5 * error;
end
function val = basisFunction(x, m)
    val = x^m;
end
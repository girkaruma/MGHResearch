function testGradientDescent()
 min = gradientDescent([1 1 1 1], 0.0001, 0.001)
 M = 3;
 data = importdata('regress_validate.txt');
 X = data(1,:);
 Y = data(2,:);
% min = [0.8015    0.8738    0.5090];
 computeLeastAbsoluteDeviation(X, Y, M, [0.7753    0.6890    0.7310    0.0286], 0)
%fun = @(x)(x(1)+4)*(x(1)-3)*(x(1)+1)*(x(1)-2);
%[x,fval,exitflag,output] = fminunc(fun,-10)

    figure;
    plot(X, Y, 'o', 'MarkerSize', 10);
    xlabel('x');
    ylabel('y');
    hold all
    x = -3:.01:2;
    y = min*designMatrix(x,M);
    plot(x,y)
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
    %error = error + lambda * weightMagnitude;
    error = 0.5 * error;
end
function phi = designMatrix(x, order)
    M = order;
    phi = zeros(M+1, length(x));
    for i=1:length(x)
        for j=0:M
            phi(j+1,i)=basisFunction(x(i),j);
        end
    end
end
function val = basisFunction(x, m)
    val = x^m;
end
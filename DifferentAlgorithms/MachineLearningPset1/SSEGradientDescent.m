function SSEGradientDescent()
 min = gradientDescent([1 1 1 1 1 1], 0.0001, 0.001)
 M = 5;
 data = importdata('curvefitting.txt');
 X = data(1,:);
 Y = data(2,:);
 computeSumOfSquaresError(X, Y, M, min)

    figure;
    plot(X, Y, 'o', 'MarkerSize', 10);
    xlabel('x');
    ylabel('y');
    hold all
    x = -3:.01:2;
    y = min*designMatrix(x,M);
    plot(x,y)
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
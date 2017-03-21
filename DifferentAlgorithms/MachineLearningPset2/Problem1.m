function Problem1()
    %lambdas = [-100 -10 -1 -0.1 -0.01 0 0.01 0.1 1 2 4 6 8 10 100];
    %lambdas = [-5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8 9 10];
    lambdas = [0 0.01 0.1 1 2 5 10];
    errorMat = zeros(7,1);
    dataSets = {'stdev1', 'stdev2', 'stdev4', 'nonsep'};
    for i=1:length(lambdas)
        [a, b] = lr_test('titanic', exp(lambdas(i)));
        errorMat(i, 1) = b;
    end
    
    %errorTable = zeros(11, 8);
%         for i = 1:length(lambdas)
%             [c1, c2] = lr_test('stdev1', exp(lambdas(i)));
%             errorTable(i, 1) = c1;
%             errorTable(i, 2) = c2;
%         end
%         for i = 1:length(lambdas)
%             [c1, c2] = lr_test('stdev2', exp(lambdas(i)));
%             errorTable(i, 3) = c1;
%             errorTable(i, 4) = c2;
%         end
%         for i = 1:length(lambdas)
%             [c1, c2] = lr_test('stdev4', exp(lambdas(i)));
%             errorTable(i, 5) = c1;
%             errorTable(i, 6) = c2;
%         end
%         for i = 1:length(lambdas)
%             [c1, c2] = lr_test('nonsep', exp(lambdas(i)));
%             %errorTable(i, 7) = c1;
%             %errorTable(i, 8) = c2;
%         end
    errorMat
end


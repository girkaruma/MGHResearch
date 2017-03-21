function Problem2()
optim_ver = ver('optim');
optim_ver = str2double(optim_ver.Version);
if optim_ver >= 6
    opts = optimset('Algorithm', 'interior-point-convex');
else
    opts = optimset('Algorithm', 'interior-point', 'LargeScale', 'off', 'MaxIter', 2000);
end
% H = [5 6 0 -4; 6 8 0 -2; 0 0 0 0; -4 -2 0 13];
% f = [-1 -1 -1 -1]';
% Aeq = [1 1 -1 -1];
% beq = 0;
% lb = [0 0 0 0]';
% ub = [1 1 1 1]';
% sol = quadprog(H, f, [], [], Aeq, beq, lb, ub, [], opts);
% weights = [0 0];
% Y = [1 1 -1 -1]';
% X = [1 2; 2 2; 0 0; -2 3];
% for i = 1:length(sol)
%     weights = weights + Y(i) * sol(i) * X(i, :);
% end
% w0 = getIntercept(X, Y, sol);
% errorMat = zeros(7, 1);
% C = [0.01 0.1 1 2 5 10 100];
% for i=1:length(C)
%     [a, b, c] = svm_test('titanic', C(i), 0);
%     errorMat(i, 1) = c;
% end
% errorMat
% C = [0.01 0. 1 10 100];
% bw = [0.1 1 10];
% stdev1 = zeros(15, 2);
% count = 1;
% for j = 1:length(bw)
% for i = 1:length(C)    
%         [geo, sv] = svm_test('nonsep', i, j);
%         stdev1(count, 1) = geo;
%         stdev1(count, 2) = sv;
%         count = count + 1;
%     end
% end
% stdev1
svm_test('stdev2', 1, 1);
%svm_test1('stdev1', 1);
%x = quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options)
end


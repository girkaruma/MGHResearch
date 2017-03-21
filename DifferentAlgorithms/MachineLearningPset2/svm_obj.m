function [ w, intercept] = svm_obj(C,data)
%SVM_obj This function train the SVM weight for given dataset
%   C: upper bound for the alpha or the penalty for the slack varibe
%   data: the name of a file that contains both Xs and Ys, with Ys at the
%   last colume

%init data
table = importdata(data);

[k, j] = size(table);
X = table(:,1:j-1);
Y = table(:,j);
% X = [1 2; 2 2; 0 0; -2 3];
% Y = [1 1 -1 -1]';

for i = 1:j-1
    Ph_X(:,i)=X(:,i).*Y;
end

H = Ph_X*Ph_X';

f = ones(1,length(Y)).*(-1);
Aeq = Y';
beq = 0;
lb = zeros(length(Y),1);
ub = C.*ones(length(Y),1);

optim_ver = ver('optim');
optim_ver = str2double(optim_ver.Version);
if optim_ver >= 6
    opts = optimset('Algorithm', 'interior-point-convex');
else
    opts = optimset('Algorithm', 'interior-point', 'LargeScale', 'off', 'MaxIter', 2000);
end
sol = quadprog(H,f,[],[],Aeq,beq,lb,ub,[],opts);

temp = sol.*Y;
w= zeros(1,j-1);
for i=1:length(Y)
    w = w + temp(i).*X(i,:);
end

%calculate intercept
count = 0;
for i = 1:length(Y)
    if sol(i) > 0.01
        count = count +1;
        support_vector(count,:) = X(i,:);
        support_alpha(count,:) = sol(i);
        support_label(count,:) = Y(i,:);
    end
end
amtm = support_alpha.*support_label;

support_K = support_vector*support_vector';

final_K = zeros(length(amtm),length(amtm));

A = ones(length(amtm),1);

B = ones(1,length(amtm));

for i = 1:length(amtm)
    final_K(:,i) = amtm.*support_K(:,i);
end

inner_sum = final_K*A;

pre_outer_sum = support_label-inner_sum;

nm=length(amtm);
intercept = B*pre_outer_sum/nm;
scatter(X(:,1),X(:,2),50,1+Y);
hold on;
x1 = (-3:0.5:2);
x2 = -w(1)/w(2)*x1 - intercept;

plot(x1,x2,'r');

end

function errorRate = getClassificationErrorRate(X, Y, w, w0)
    error = 0;
    for i=1:length(Y)
        val = predictSVM(X(i, :)', w, w0);
        if val <= 0
            val = -1;
        else
            val = 1;
        end
        if val ~= Y(i)
            error = error + 1;
        end
    end
    errorRate = error/(1.0 * length(Y));
end


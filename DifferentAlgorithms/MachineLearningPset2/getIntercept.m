function intercept = getIntercept(X, Y, sol, C)
    count = 0;
    for i = 1:length(Y)
        %if sol(i) > 0.01
        if sol(i) > 10^(0-5)*C
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
end


function plots()
%     scatter([1     2     5    10    25    50   100   150   200], [0.6667    0.0100    0.0100    0.0167    0.0067    0.0333    0.1767    0.6667    0.5867], 100, 'r');
%     hold on
%     scatter([1     2     5    10    25    50   100   150   200], [0.6667    0.6433    0.6667    0.3333    0.2633    0.1300    0.0567    0.3500    0.1233], 100, 'b');
%    scatter([log2(0.0001) log2(0.001) log2(0.01) log2(0.1) log2(1) log2(exp(1))], [0.0920  0.1300 0.0900 0.0720 0.0740 0.5620]);
    %scatter(log2([1 0.01 0.001 0.0001]), [0.7400 0.1160 0.1060 0.1060]);
    %scatter(log2([1 0.1 0.01 0.001]), [0.3700 0.0720 0.12 0.6720]);
    scatter(log2([0.0001 0.001 0.01 0.1]), [0.1120 0.1160 0.2180 0.7660]);
end

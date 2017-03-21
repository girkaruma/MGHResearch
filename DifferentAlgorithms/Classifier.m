labels = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
%images = ['Images/Image1.png'; 'Images/Image2.png'; 'Images/Image3.png'; 'Images/Image4.png'; 'Images/Image5.png' 'NoImages/NoImage1.png'; 'NoImages/NoImage2.png'; 'NoImages/NoImage3.png'; 'NoImages/NoImage4.png'; 'NoImages/NoImage5.png'];
% features = zeros(2, 10);
% for i = 1:40 
%     if i <= 20
%         C = {'Images/Image', int2str(i), '.png'};
%     else 
%         j = i - 20;
%         C = {'NoImages/NoImage', int2str(j), '.png'};
%     end
%     [area, boxes] = NoPreProcessing(strjoin(C, ''))
%     features(1, i) = boxes;
%     features(2, i) = area;
% end
% features = features';
% features = [939	1269575;
% 686	159057;
% 1342 255028;
% 71	191799;
% 115	240120;
% 128	76329;
% 26	149730;
% 7	80028;
% 56	99750;
% 62	176750;
% 53	58844;
% 78	109440;
% 114	247878;
% 58	551826;
% 334	1583397;
% 12	178816;
% 24	147496;
% 12	101112;
% 15	60525;
% 13	159203;
% 211	29600;
% 198	55200;
% 196	41832;
% 26	37697;
% 7	58779;
% 18	32430;
% 37	23491;
% 155	92504;
% 3	29415;
% 17	33681;
% 79	8288;
% 13	27405;
% 14	19224;
% 14	9504;
% 6	19845;
% 145	59087;
% 213	56628;
% 50	8255;
% 72	22575;
% 52	71137];

% features = [0	0;
% 12	11832;
% 41	6600;
% 3	9028;
% 8	3549'
% 7	3700;
% 16	5830;
% 3	5445;
% 1	4788;
% 9	2520;
% 2	9010;
% 19	7250;
% 8	2976;
% 15	3588;
% 12	39223;
% 11	3192;
% 4	2320;
% 5	5720;
% 2	5874;
% 6	3444;
% 10	2624;
% 4	2028;
% 7	2387;
% 3	1617;
% 1	1554;
% 2	2312;
% 2	1650;
% 8	2002;
% 1	2343;
% 5	1634;
% 6	2025;
% 4	2370;
% 2	1794;
% 1	1475;
% 3	1800;
% 12	4543;
% 15	3726;
% 15	2378;
% 15	2914;
% 26	2835
% ];

% features = [44	1269575;
% 24	951539;
% 81	1778715;
% 21	227247;
% 12	434907;
% 5	174000;
% 2	163737;
% 2	82134;
% 7	141360;
% 2	242535;
% 4	107045;
% 11	312481;
% 16	288301;
% 31	602921;
% 50	1583397;
% 2	202270;
% 2	173582;
% 5	103410;
% 5	66443;
% 2	176341;
% 12	199133;
% 7	236457;
% 6	223011;
% 4	69575;
% 2	60645;
% 2	43475;
% 1	49235;
% 3	209253;
% 1	30369;
% 2	39894;
% 1	64629;
% 1	39440;
% 2	22672;
% 2	15295;
% 2	28199;
% 1	199875;
% 2	255945;
% 12	53261;
% 6	74290;
% 10	102399];

features1 = [0	0;
12	11832;
41	6600;
3	9028;
8	3549'
7	3700;
16	5830;
3	5445;
1	4788;
9	2520;
2	9010;
19	7250;
8	2976;
15	3588;
12	39223;
11	3192;
4	2320;
5	5720;
2	5874;
6	3444;
10	2624;
4	2028;
7	2387;
3	1617;
1	1554;
2	2312;
2	1650;
8	2002;
1	2343;
5	1634;
6	2025;
4	2370;
2	1794;
1	1475;
3	1800;
12	4543;
15	3726;
15	2378;
15	2914;
26	2835
];

features2 = [939	1269575;
686	159057;
1342 255028;
71	191799;
115	240120;
128	76329;
26	149730;
7	80028;
56	99750;
62	176750;
53	58844;
78	109440;
114	247878;
58	551826;
334	1583397;
12	178816;
24	147496;
12	101112;
15	60525;
13	159203;
211	29600;
198	55200;
196	41832;
26	37697;
7	58779;
18	32430;
37	23491;
155	92504;
3	29415;
17	33681;
79	8288;
13	27405;
14	19224;
14	9504;
6	19845;
145	59087;
213	56628;
50	8255;
72	22575;
52	71137];

features3 = [44	1269575;
24	951539;
81	1778715;
21	227247;
12	434907;
5	174000;
2	163737;
2	82134;
7	141360;
2	242535;
4	107045;
11	312481;
16	288301;
31	602921;
50	1583397;
2	202270;
2	173582;
5	103410;
5	66443;
2	176341;
12	199133;
7	236457;
6	223011;
4	69575;
2	60645;
2	43475;
1	49235;
3	209253;
1	30369;
2	39894;
1	64629;
1	39440;
2	22672;
2	15295;
2	28199;
1	199875;
2	255945;
12	53261;
6	74290;
10	102399];


% nbGau = fitcnb(features1, labels, 'DistributionNames','kernel', 'Kernel','box');
% nbGauResubErr = resubLoss(nbGau)
% cp = cvpartition(labels,'KFold',10)
% nbGauCV = crossval(nbGau, 'CVPartition',cp);
% nbGauCVErr = kfoldLoss(nbGauCV)

%pred = features1(:,1:2);
resp = (1:40)'<=20;
mdlNB = fitcnb(features1, resp, 'DistributionNames','kernel', 'Kernel','box');
[~,score_nb] = resubPredict(mdlNB)
[Xnb,Ynb,Tnb,AUCnb] = perfcurve(resp,score_nb(:,mdlNB.ClassNames),'true');
AUCnb
plot(Xnb,Ynb)
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC Curve for Naive Bayes Classification')
% 
% [numberOfBoxedRegions,areaOfLargestBoxedRegion] = meshgrid(min(features1(:,1)):max(features1(:,1)),min(features1(:,2)):max(features1(:,2)));
% numberOfBoxedRegions = numberOfBoxedRegions(:);
% areaOfLargestBoxedRegion = areaOfLargestBoxedRegion(:);
% label = predict(nbGau, [numberOfBoxedRegions areaOfLargestBoxedRegion]);
% gscatter(numberOfBoxedRegions,areaOfLargestBoxedRegion,label,'grb','sod')



% t = fitctree(features1, labels,'PredictorNames',{'SL' 'SW' });
% [x,y] = meshgrid(min(features1(:,1)):max(features1(:,1)),min(features1(:,2)):max(features1(:,2)));
% x = x(:);
% y = y(:);
% [grpname,node] = predict(t,[x y]);
% gscatter(x,y,grpname,'grb','sod')
% view(t,'Mode','graph');
% dtResubErr = resubLoss(t)
% cp = cvpartition(labels,'KFold',10)
% cvt = crossval(t,'CVPartition',cp);
% dtCVErr = kfoldLoss(cvt)
% Solve a Pattern Recognition Problem with a Neural Network
% Script generated by NPRTOOL
%
% This script assumes these variables are defined:
%
%   cancerInputs - input data.
%   cancerTargets - target data.
labels = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
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

inputs = features3';
targets = labels;

% Create a Pattern Recognition Network
hiddenLayerSize = 20;
net = patternnet(hiddenLayerSize);


% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;


% Train the Network
net.trainFcn = 'trainbr';
[net,tr] = train(net,inputs,targets);

% Test the Network
outputs = net(inputs);
errors = gsubtract(targets,outputs);
performance = perform(net,targets,outputs)

tInd = tr.testInd;
tstOutputs = net(inputs(:,tInd));
tstPerform = perform(net,targets(:,tInd),tstOutputs)

% View the Network
%view(net)

% Plots
% Uncomment these lines to enable various plots.
figure, plotperform(tr)
% figure, plottrainstate(tr)
% figure, plotconfusion(targets,outputs)
% figure, ploterrhist(errors)
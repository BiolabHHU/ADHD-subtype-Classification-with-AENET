
% Include dependencies
addpath('./lib');                      % dependencies
addpath('./methods');            % FS methods
addpath(genpath('./lib/drtoolbox'));


% Load the data and select features for classification
load(['NYU_data' '.mat'])
X_temp = inform.brain_conn;                    % FC's upper triangle matrix without diagonal elements
Y_temp = inform.tag_C;
Ytemp = inform.tag

X = [];

for i=1:max(size(X_temp))    
   temp = X_temp{1,i}';
   X = [temp; X];
end

Y = nominal(ismember(Y_temp,1)); 


X_train = double(X);
Y_train = (double(Y)-1)*2-1; 
%  labels:  neg_class: -1, pos_class: +1
 numclass=size(unique(Y_train),1);
X= X_train(:,1:4005)
ranking = spider_wrapper(X,Y_train,5,lower('rfe'))
r=SVM_RFE(X_train,Y_train);
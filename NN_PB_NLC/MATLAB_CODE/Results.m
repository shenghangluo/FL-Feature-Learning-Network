% % 0 dBm
% % FO (1000 epoch)
% BER_min =
% 
%     0.0120
% 
% 
% Q_max =
% 
%     7.0770
%     
% % SO (20 neurons in second hidden layer, 1000 epoch)
% BER_min =
% 
%     0.0090
% 
% 
% Q_max =
% 
%     7.4843
clear all
close all
clc
prices = [10328 26712 126192 37208 11352];
bar(prices)
set(gca,'xticklabel',{'RNN-FO-PB-NLC','RNN-SO-PB-NLC','DNN-SO-PB-NLC(1314 triplets)', 'DNN-SO-PB-NLC', 'DNN-FO-PB-NLC'});
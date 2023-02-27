clear all
close all
clc

longstate = csvread('long_state.csv');
Window_size = 25;
hidden_unit = 25;
state = zeros(2^16,Window_size*hidden_unit);
for ii = 1:2^16
    state(ii, 1:Window_size*hidden_unit) = reshape(longstate(ii:Window_size+ii-1, :).', [1, Window_size*hidden_unit]);
end
save('Processed_state.mat', 'state')
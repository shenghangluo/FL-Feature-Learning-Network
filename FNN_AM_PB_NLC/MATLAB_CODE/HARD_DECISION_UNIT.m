function [hard_decision_X,hard_decision_Y]=HARD_DECISION_UNIT(x_data,y_data, M)


hard_decision_X_temp=qamdemod(x_data.', M,  'OutputType', 'integer', 'UnitAveragePower', true);
hard_decision_Y_temp=qamdemod(y_data.', M,  'OutputType', 'integer', 'UnitAveragePower', true);

hard_decision_X=qammod(hard_decision_X_temp.', M,  'InputType', 'integer', 'UnitAveragePower', true);
hard_decision_Y=qammod(hard_decision_Y_temp.', M,  'InputType', 'integer', 'UnitAveragePower', true);

end
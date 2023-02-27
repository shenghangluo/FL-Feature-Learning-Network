clc
clear all

Tx_Pow_dBm_ = [];
q_factor_db_cdc_ = [];


load('Results_n3_dBm_2021_11_22_18_31');
Tx_Pow_dBm_ = [Tx_Pow_dBm_, Tx_Pow_dBm];
q_factor_db_cdc_ = [q_factor_db_cdc_, q_factor_db_cdc];

load('Results_n3_dBm_2021_11_22_18_31');
Tx_Pow_dBm_ = [Tx_Pow_dBm_, Tx_Pow_dBm];
q_factor_db_cdc_ = [q_factor_db_cdc_, q_factor_db_cdc];


load('Results_n3_dBm_2021_11_22_18_31');
Tx_Pow_dBm_ = [Tx_Pow_dBm_, Tx_Pow_dBm];
q_factor_db_cdc_ = [q_factor_db_cdc_, q_factor_db_cdc];


plot(Tx_Pow_dBm_, q_factor_db_cdc_, '-or');


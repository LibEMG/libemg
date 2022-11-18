clear variables;
clc
window_size = 200;
window_inc  = 25;
data_location = "data/emg_data_myo.csv";
data = readmatrix(data_location);
feature_list = {'MAV';
                'ZC';
                'SSC';
                'WL';
                'LS';
                'MFL';
                'MSR';
                'WAMP';
                'RMS';
                'IEMG';
                'DASDV';
                'VAR';
                'M0';
                'M2';
                'M4';
                'SPARSI';
                'IRF';
                'WLR';
                'AR'; 
                'CC';
                'LD';
                'MAVFD';
                'MAVSLP';
                'MDF';
                'MNF';
                'MNP';
                'MPK';
                'SKEW';
                'KURT';
                'RMSPHASOR';
                'WLPHASOR';
                'MZP';
                'PAP';
                'TM';
                'SM';
                'SAMPEN';
                'FUZZYEN'};
feature_functions = cellfun( @(a) "get" + lower(a) + "feat",feature_list);

for f =1:length(feature_functions)
    eval("feature_values = " + feature_functions(f) + "(data, window_size, window_inc);")
    csvwrite("data\matlab_"+feature_list{f}+".csv",feature_values);
end



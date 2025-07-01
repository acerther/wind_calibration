clear all
close all

% Define file name
filename    = '250626.txt';
StartTime   = '2025-06-26 20:32:20';
EndTime     = '2025-06-26 20:36:20';
y_pos        = 2 ;




% datenum(StartTime)


z_pos   = [1.6+0.1 2.2+0.15 2.8+0.15 3.4+0.15 4.0+0.15 4.6+0.15 5.2+0.15 5.8+0.15]-1.7;

z_pos_index   = [5 6 3 8 7 4 2 1];

z_pos = z_pos(z_pos_index);


%% Read data into array
% Read the data as a table, treat '-----' as missing
opts = detectImportOptions(filename, 'Delimiter', '\t');
opts = setvaropts(opts, opts.VariableNames, 'TreatAsMissing', '-----');
opts = setvaropts(opts, opts.VariableNames, 'EmptyFieldRule', 'auto');

% Read the table
T1 = readtable(filename, opts);

% Convert time column to datetime
T1.SensorSampleTime = datetime(T1.SensorSampleTime, 'InputFormat', 'yyyy-MM-dd HH:mm:ss');


T1 = cleanTableToNumeric(T1);



channelName = {'WS2 ', 'WS3 ', 'WS4 ', 'WS5 ', 'WS6 ', 'WS7 ', 'WS8 ', 'WS T2 '};


WS_all = table2array(T1(:,[2:8 11]));   % 选取风速信号列
WD     = table2array(T1(:,[12]));   % 选取风向
Time   = T1.SensorSampleTime;

Time_num = datenum(Time);




% %%
% figure()
% plot(T1.SensorSampleTime,T1.x__1_m_s)
% 
% 
% %%  the wind speed from the first channel
% figure()
% plot(T1.x__1_m_s)



%% You need to adjust the information below
index_start = find(Time_num>datenum(StartTime),1);
index_end   = find(Time_num>=datenum(EndTime),1);

index{1} = [index_start index_end];
% index{2} = [6350 6617];
% index{3} = [6683 6984];
% index{4} = [7660 7967];
% index{5} = [8044 8345];
% y_pos   = [2.0 1.5 1.0 0.5 0];


%%
WS_mean  = nan(length(y_pos),length(z_pos));
for it = 1:length(index)
    
    WS_mean(it,:) = mean(WS_all(index{it}(1):index{it}(2),:),1);
     
end



%%
figure()
plot(Time,WS_all)




%%
% Plot the points (optional)
figure;
hold on

for it = 1:length(y_pos)
    for ih = 1:length(z_pos)
        scatter(y_pos(it), z_pos(ih), 50, WS_mean(it,ih), 'filled');  % Color points by wind speed
    end
end
colorbar;
xlabel('Y Position');
ylabel('Z Position');
title('Horizontal Wind Speed [m/s]');



% Add text annotations at each (x, y) position
for it = 1:length(y_pos)
    for ih = 1:length(z_pos)
        text(y_pos(it), z_pos(ih), [channelName{ih} ' ' num2str(WS_mean(it,ih),'%.2f')], ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', ...
        'FontSize', 10, 'Color', 'k');
    end
end

axis equal 
ylim([-0.5 4.8])


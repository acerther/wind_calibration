function T_clean = cleanTableToNumeric(T_in)
% CLEANTABLETONUMERIC Converts string-like numeric entries to doubles and replaces invalid entries with NaN.
% Works with tables where values may be stored as strings or cell arrays.
%
% Input:
%   T_in - input table with strings, cells, or numeric entries
% Output:
%   T_clean - numeric version with non-convertible strings replaced by NaN

    T_clean = T_in;  % Copy structure

    for col = 1:width(T_in)
        colData = T_in{:, col};

        % Convert to string array first
        if iscell(colData)
            colStr = string(colData);
        elseif isstring(colData)
            colStr = colData;
        elseif ischar(colData)
            colStr = string(colData);
        elseif isnumeric(colData)
            continue  % Already numeric, no action needed
        else
            % Skip unsupported types
            warning("Skipping column %s: unsupported type.", T_in.Properties.VariableNames{col});
            continue
        end

        % Convert strings like '1.5' to numbers, invalids to NaN
        numCol = str2double(colStr);

        % Assign back ¡ª make sure it's a numeric column
        T_clean.(T_in.Properties.VariableNames{col}) = numCol;
    end
end

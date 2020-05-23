 function charText = ConvertText(hotText, int_to_char)
% Function that converts one hot encoded text to character 

% Inputs: 
% hotText = one hot encoded text sequence
% int_to_char = Map of int paired with correct character

% Returns: 
% charText = charcter representation of hotText

[~,idx] = max(hotText); 
charText = char();

for i = 1:length(idx)
    curr = int_to_char(idx(i));
    charText = append(charText,curr);
end


end


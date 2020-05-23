function hotText = hotEnc(charText, char_to_ind)
% Converts a sequence of characters to a one-hot representation

% Input:
% charText = sequence of characters
% char_to_ind = Map of character paired with correct integer


% Returns: 
% hotText = one-hot encoded representation of input charText

hotText = zeros(length(char_to_ind),length(charText));

for i = 1:length(charText)
    hotText(char_to_ind(charText(i)),i) = 1;
end

end


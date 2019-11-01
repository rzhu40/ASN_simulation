function [weight] = getWeight(measure, signal, steps)
% get training weight with measured data, target and number of steps feed
% in.

[training_length, E] = size(measure);
lhs = zeros(training_length-steps, E*steps+2);
rhs = signal(steps+1:training_length);

lhs(:,1) = 1;
% lhs(:,2) = signal(steps:training_length-1);
lhs(:,2) = 0;
for i = 0:steps-1
    lhs(:, i*E+3:(i+1)*E+2) = measure(steps-i:training_length-i-1,:);
end 

weight = regress(rhs, lhs);

end


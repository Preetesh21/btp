% define transfer functions
s = tf('s');
C =@(Kp,a) Kp*(s + a)/s;
G = tf(1,[1 8 15])
% define parameter combinations
Kp = 8:2:14;
a = 3;

P = tf(16,[1,4,16])

% create figure
figure
ah = axes;
hold(ah,'on');
step(P)
stepinfo(P)
[y,t]=step(P);
sserror = abs(1-y(end))
% loop through given combinations of Kp and Ki
for ik = 1:length(Kp)
    H = feedback(C(Kp(ik),a)*G,1);
    step(H)
    stepinfo(H)
    [y,t]=step(H);
    sserror = abs(1-y(end))
end
% create cell array of legend entries
cLegend = arrayfun(@(Kp) sprintf('Kp = %i, a = 3',Kp),Kp,'UniformOutput',false);
% add legend to figure
legend(ah,cLegend)
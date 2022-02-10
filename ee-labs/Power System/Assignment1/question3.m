i=0;
t=[];
while(i<24)
t(end+1)=i;
i=i+0.005;
end
b=linspace(0,24,25);
p1=@(t) 6*exp(-((t-2)^2)/17);
p2=@(t) 5*exp(-((t-9)^2)/16)
p3=@(t) 4*exp(-((t-23)^2)/19)
p4=@(t,b) 5.5*exp(-((t-b)^2)/18)

p1_power=[];
for i =1:length(t)
p1_power(end+1)=(p1(t(i)));
end

p2_power=[];
for i =1:length(t)
p2_power(end+1)=(p2(t(i)));
end

p3_power=[];
for i =1:length(t)
p3_power(end+1)=(p3(t(i)));
end

p4_power=[];
for i=1:length(b)
p4_bpower=[];
for j=1:length(t)
p4_bpower(end+1)=p4(t(j),b(i));
end
p4_power=[p4_power;p4_bpower];
end

max_p1=max(p1_power);
max_p2=max(p2_power);
max_p3=max(p3_power);

max_power123=p1_power+p2_power+p3_power;

df_ans=[]
for i=1:length(b)
current_p4=p4_power(i,:);
df=(max_p1+max_p2+max_p3+max(current_p4))/(max(max_power123+current_p4));
df_ans(end+1)=df;
end
[M,I]=max(df_ans)
fprintf('The value of b is=%d',I)

p4_power=p4_power(I,:);

plot(t,p1_power);
figure();
plot(t,p2_power);
figure();
plot(t,p3_power);
figure();
plot(t,p4_power);
figure();
tot=p1_power+p2_power+p3_power+p4_power;
plot(t,tot);
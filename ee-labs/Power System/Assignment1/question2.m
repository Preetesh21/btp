data=[0 8 0;
8 9 300;
9 17 9000;
17 19 300; 
19 24 0];
power=data(:, 3);
Duration=data(:,2) - data(:,1);
Demand=power.*Duration;
Total_power_demand=sum(Demand);
Total_power2=sum(power.*Duration);
DATA=[data(:,1) data(:,2) power];
Average_load=Total_power2/sum(Duration);
peak_load=max(power);
Daily_LF=Average_load/peak_load;
L=length(data);
timeinterval=data(:,1:2) ; 
tA = sort(reshape(timeinterval,1,2*L));
for n = 1:L
  PA(2*n-1) = power(n);
  PA(2*n) = power(n);
end

fprintf('Load Factor of A= %d.\n',Daily_LF);

dataB=[0 6 0;6 8 1500; 8 10 10000; 10 17 1500;
17 20 12000; 20 22 5000; 22 24 0];
powerB=dataB(:, 3);
DurationB=dataB(:,2) - dataB(:,1);
DemandB=powerB.*DurationB;
Total_power_demandB=sum(DemandB);
Total_power2B=sum(powerB.*DurationB);
DATAB=[dataB(:,1) dataB(:,2) powerB];
Average_loadB=Total_power2B/sum(DurationB);
peak_loadB=max(powerB);
Daily_LFB=Average_loadB/peak_loadB;
LB=length(dataB);
timeintervalB=dataB(:,1:2) ; 
tB = sort(reshape(timeintervalB,1,2*LB));
for n = 1:LB
  PB(2*n-1) = powerB(n);
  PB(2*n) = powerB(n);
end
fprintf('Load Factor of B= %d.\n',Daily_LFB);

t=0;
i=1;
j=1;
ends=0;
data3_start=[];
data3_end=[];
data3_power=[];
aax=data(:, 3);
bbx=dataB(:,3);
endA=data(:,2);
endB=dataB(:,2);
while t<24
start=(ends);
ends=min(endA(i),endB(j));
tot=aax(i)+bbx(j);
data3_start=[data3_start,start];
data3_end=[data3_end,ends];
data3_power=[data3_power,tot];
if(endA(i)<endB(j))
i=i+1;
elseif(endA(i)>endB(j))
j=j+1;
else
i=i+1;
j=j+1;
end
t=ends;
end

DurationC=data3_end-data3_start;
DemandC=data3_power.*DurationC;
Total_power_demandC=sum(DemandC);
Total_power2C=sum(data3_power.*DurationC);
Average_loadC=Total_power2C/sum(DurationC);
peak_loadC=max(data3_power);
Daily_LFC=Average_loadC/peak_loadC;
LC=length(data3_end);
timeintervalC=[data3_start data3_end] ; 
tC = sort(reshape(timeintervalC,1,2*LC));
for n = 1:LC
  PC(2*n-1) = data3_power(n);
  PC(2*n) = data3_power(n);
end
plot(tB,PB,tA,PA);

xlabel(['Timer, Hr']);
ylabel(['Power, KW']);
title('Daily Load Curve for first and second area');
figure();
fprintf('Load Factor= %d.\n',Daily_LFC);
max_overall=max(data3_power);
maxA=max(power);
maxB=max(powerB);
load_diversity_factor=(maxA+maxB)/max_overall;
fprintf('Load Diversity Factor=%d.\n',load_diversity_factor);

plant_utilization_factor=max_overall/20000;
fprintf('Plant Utilization Factor=%d.\n',plant_utilization_factor);

plant_capacity_factor=Average_loadC/20000;
fprintf('Plant Capacity Factor=%d.\n',plant_capacity_factor);

plant_usage_factor=Total_power_demandC/(20000*16);
fprintf('Plant Use Factor=%d.\n',plant_usage_factor);

plot(tC,PC);

xlabel(['Timer, Hr']);
ylabel(['Power, KW']);
title('Daily Load Curve for overall area');
figure();

xx=reshape(data3_power,[9,1]);
yy=reshape(DurationC,[9,1]);
zz=[xx yy];
zz=sortrows(zz,-1);
zz(:,2)=cumsum(zz(:,2));
yy=zz(:,2);
yy=[0;yy];
yy(end)=[];
zz=[zz yy];
zz=[zz(:,1) zz(:,3) zz(:,2)];
L=length(zz);data3_power=sort(data3_power,'descend');
timeinterval=zz(:,2:3) ; 
tttt = sort(reshape(timeinterval,1,2*L));
for n = 1:L
  Pp(2*n-1) = data3_power(n);
  Pp(2*n) = data3_power(n);
end
plot(tttt,Pp);
title('Load Duration Curve for overall System');
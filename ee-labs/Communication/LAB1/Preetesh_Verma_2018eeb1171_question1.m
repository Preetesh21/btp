freq=1;% 1 Hz
t=linspace(0,3,1024)/freq; %from 0 to 5 periods in 1024 steps
A=5;%amplitude
x=A*sin(freq*t*2*pi); % sample sine wave with given amplitude and frequency
figure % used for creating subplots within the same plot.
subplot(2,2,1)
plot(t,x)
xlabel('seconds');
ylabel('Amplitude');
title('Original Sine Signal');

fy=1; % signal frequency in Hz
wy=2*pi*fy; % signal frequency in rad/s 

fs=0.5; % sampling frequency in Hz
tiv=1/fs ;% time interval between samples
t=0:tiv:(3); % time interval set
y=A*sin(wy*t); % signal data set
subplot(2,2,2)
stem(t,y,'-kd');
xlabel('seconds');
ylabel('Amplitude');
title('Undersampled Signal');

Fs=2*1; %Sampling frequency=2Hz
t1=0.25:1/Fs:3; %time interval where 1/Fs is sampling period.
x1=5*sin(2*pi*1*t1); %sampled signal
subplot(2,2,3)
stem(t1,x1);
xlabel('Time(s)');
ylabel('Amplitude');
title('Sampled at Nyquist rate'); %Fs=2fm

fs=10; % sampling frequency in Hz
tiv=1/fs; % time interval between samples
t=0:tiv:(3); % time interval set
y=A*sin(wy*t); % signal data set
subplot(2,2,4)
stem(t,y,'-kd');
xlabel('seconds');
ylabel('Amplitude');
title('Oversampled Signal');

% Aliiasing Effect-- Because the sampling frequency was too low, a high-frequency cosine looked like a low-frequency cosine after we sampled it.
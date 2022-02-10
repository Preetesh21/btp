fm=1; %frequency of sine wave=1Hz
%Part(a)
Fs=2*1; %Sampling frequency=2Hz
t1=0.25:1/Fs:3; %time interval where 1/Fs is sampling period.
x1=5*sin(2*pi*fm*t1); %sampled signal
y1=fft(x1); %Discrete fourier transform of sampled signal x1.
f = (0:length(y1)-1)*Fs/length(y1); %frequency interval
subplot(2,2,1)
stem(f,abs(y1))
xlabel('Frequency (Hz)')
ylabel('Magnitude')
title('Sampled at Nyquist rate');


%Part(b)
Fs=10*1; %Sampling frequency=10Hz
t2=0:1/Fs:3; %time interval where 1/Fs is sampling period.
x2=5*sin(2*pi*fm*t2); %sampled signal
y2=fft(x2); %Discrete fourier transform of sampled signal x2.
f = (0:length(y2)-1)*Fs/length(y2); %frequency interval
subplot(2,2,2)
stem(f,abs(y2))
xlabel('Frequency (Hz)')
ylabel('Magnitude')
title('Oversampling');


%Part(c)
Fs=0.5; %Sampling frequency=1.5Hz
t3=0:1/Fs:3; %time interval where 1/Fs is sampling period.
x3=5*sin(2*pi*fm*t3); %sampled signal
y3=fft(x3); %Discrete fourier transform of sampled signal x2.
f = (0:length(y3)-1)*Fs/length(y3); %frequency interval
subplot(2,2,3)
stem(f,abs(y3))
xlabel('Frequency (Hz)')
ylabel('Magnitude')
title('Undersampling');

% Aliiasing Effect-- Because the sampling frequency was too low, a high-frequency cosine looked like a low-frequency cosine after we sampled it.
%------ Quesiton1 (Modulation)
clc;
%part(a)

Am=5;                 %amplitude of message signal
Fm=1;                 %frequency of message signal
t=0:0.01:3;          %time vector
m=Am*sin(2*pi*Fm*t);  %message signal

subplot(2,2,1)
plot(t,m);            %plotting message signal as subplot 1
xlabel('Time(s)');
ylabel('Amplitude');
title('Message signal');

%part(b)

u=0.9;                %modulation index
Fc=10;                %frequency of carrier signal
Ac=Am/u;              %amplitude of carrier signal
c=Ac*cos(2*pi*Fc*t);  %carrier signal

s=(1+u*sin(2*pi*Fm*t)).*c;        %modulated signal

subplot(2,2,2)
plot(t,s,t,(Ac+(m)),t,(-Ac-(m))); %plotting modulated signal as subplot 2
xlabel('Time(s)');
ylabel('Amplitude');
title('Modulated signal');

%plotting message signal in frequency domain
N=length(t);
y=fft(m,N);
z=y(1:(floor(N/2)+1));
k=0:floor(N/2);

subplot(2,2,3)
stem(k,abs(z));
xlabel('k');
ylabel('Magnitude');
title('Message signal in Frequency Domain');

%plotting modulated signal in frequency domain
y1=fft(s,N);
z1=y1(1:(floor(N/2)+1));
k1=0:floor(N/2);

subplot(2,2,4)
stem(k1,abs(z1));
xlabel('k');
ylabel('Magnitude');
title('Modulated signal in Frequency Domain')

%plots are in digital domain.So, we are converting the frequencies obtained from plots to analog domain so that we can verify them with theoretical results.

k=3; %frequency obtained from message signal curve corresponding to peaks
digital_freq_in_rad=(2*pi/N)*k;
analog_freq_in_rad=digital_freq_in_rad/0.01;
analog_freq_in_Hertz=analog_freq_in_rad/(2*pi);
disp('Frequency of message signal(Hz), Fm = ')
disp(analog_freq_in_Hertz);

k1=[27,30,33]; %frequencies obtained from modulated signal curve corresponding to peaks
digital_freq_in_rad_s=(2*pi/N)*k1;
analog_freq_in_rad_s=digital_freq_in_rad_s/0.01;
analog_freq_in_Hertz_s=analog_freq_in_rad_s/(2*pi);
disp('Frequency components in modulated signal(Hz) = Fc-Fm, Fc, Fc+Fm = ')
disp(analog_freq_in_Hertz_s);

%Part(c)

%So, in the frequency domain we have got Upper sideband = Fc+Fm = 11Hz, Lower
%sideband = Fc-Fm = 9Hz and the carrier frequency component, Fc = 1Hz which
%is as per expected results and the same is diplayed in command window.

%-----------------------------------------------------

%Qyestion 2  (Demodulation)

z = amdemod(s, Fc, 10*Fc);
figure(2)

subplot(2,1,1);
plot(t,z-5);
title('Demodulated signal with Fc=10Hz');

err = immse(m,z-5);
disp(err);

z = amdemod(s, 40, 10*Fc);

subplot(2,1,2);
plot(t,z);
title('Demodulated signal with Fc=40Hz')
err = immse(m,z-5);
disp(err);

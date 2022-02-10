clear;
figure
% Designing the sine wave
Am=5; % Amplitude
Fs=8; % Sampling
tm=0:1/Fs:1; % Time Frame Considered
Ts=1; % Frequency
freq=1/Ts;
y1=Am*sin(2*pi*freq*tm);

% Defining N which is the number of bits
N=8;
step=2*Am/2^N; % number of steps
% Finding the min and max quantization level
minimumi=-Am+step/2;
maximumi=Am-step/2;
n=length(y1);
quant_sig=zeros(1,n);
% Main loop to allocate one of the quantization level to each sample
for i=minimumi:step:maximumi
    for j=1:n
        if(((i-step/2)<y1(j)) && (y1(j)<(i+step/2)))
            quant_sig(j)=i;
        elseif(y1(j)>=maximumi)
            quant_sig(j)=maximumi;
        elseif(y1(j)<=minimumi)
            quant_sig(j)=minimumi;
        end
    end
end
subplot(2,1,1);
hold on;
plot(tm,quant_sig);
stem(tm,quant_sig);
xlabel("Time");
ylabel("Amplitude"); 
title('Plot of Signal Sampled at fs=8Hz and N = 8');
% Calculating the estimated quantization error and the mean square
% error: using the formula given in the attached document.
estimated_error=step^2/12;
mean_square_error=immse(y1,quant_sig);
disp("Estimated error for Fs=8Hz and N=8:")
disp(estimated_error);
disp("Calculated error for Fs=8Hz and N=8:")
disp(mean_square_error);

% Defining N which is the number of bits
N=4;
step=2*Am/2^N; % number of steps
% Finding the min and max quantization level
minimumi=-Am+step/2;
maximumi=Am-step/2;
n=length(y1);
quant_sig=zeros(1,n);
% Main loop to allocate one of the quantization level to each sample
for i=minimumi:step:maximumi
    for j=1:n
        if(((i-step/2)<y1(j)) && (y1(j)<(i+step/2)))
            quant_sig(j)=i;
        elseif(y1(j)>=maximumi)
            quant_sig(j)=maximumi;
        elseif(y1(j)<=minimumi)
            quant_sig(j)=minimumi;
        end
    end
end
subplot(2,1,2);
hold on;
plot(tm,quant_sig);
stem(tm,quant_sig);

xlabel("Time");
ylabel("Amplitude"); 
title('Plot of Signal Sampled at fs=8Hz and N = 4');
% Calculating the estimated quantization error and the mean square
% error: using the formula given in the attached document.
estimated_error=step^2/12;
mean_square_error=immse(y1,quant_sig);
disp("Estimated error for Fs=8Hz and N=4:")
disp(estimated_error);
disp("Calculated error for Fs=8Hz and N=4:")
disp(mean_square_error);
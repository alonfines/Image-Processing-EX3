clc;
close all;
clear all;

% Displaying the names of the submitters:
disp('Alon Finestein - 318354057')
disp('Binyamin Pardilov - 316163914')

%% ---------------1.1 Writing your own functions---------------------------
% 1.1.3
beatles_norm = imread_double_norm('beatles.png');
gray_beatles = rgb2gray(beatles_norm);
imshow(gray_beatles)

% % 1.1.4
fft_beatles = dip_fft2(gray_beatles);
shifted_fft_beatles = dip_fftshift(fft_beatles);
amp_fft_beatles = abs(shifted_fft_beatles);
phase_fft_beatles = atan2(imag(shifted_fft_beatles), real(shifted_fft_beatles));

figure;
subplot(1,2,1);
imagesc(log(amp_fft_beatles));
colormap(subplot(1,2,1),"gray");
colorbar;
title("log of Amplitude")

subplot(1,2,2);
imagesc(phase_fft_beatles);
colormap(subplot(1,2,2),"gray");
colorbar;
title("Phase")

ifft_beatles = dip_ifft2(fft_beatles);

figure;
imagesc(real(ifft_beatles));
colormap("gray")
title('dip ifft2 image')

% test
[m,n]=size(gray_beatles);
cropped_img = ifft_beatles(1:m,1:n);

diff_val = abs(gray_beatles-real(cropped_img));
max_diff = max(max(diff_val));

%%------------------ Functions --------------------------------------------

% 1.1.1
function F = dip_fft2(I)
    % Check if input is complex, if not convert it to complex
    if ~isreal(I)
        I = complex(I);
    end

    % Ensure that input dimensions are powers of 2
    [M, N] = size(I);
    desiredSize = 2^nextpow2(max(M, N));
    if M ~= desiredSize || N ~= desiredSize
        I = padarray(I, [desiredSize-M, desiredSize-N], 0, 'post');
    end

    % Compute 1D FFT along columns
    F_col = zeros(size(I));
    for m = 1:size(I, 1)
        F_col(m, :) = dip_fft(I(m, :));
    end
    
    % Compute 1D FFT along rows
    F = zeros(size(I));
    for n = 1:size(I, 2)
        F(:, n) = dip_fft(F_col(:, n).');
    end
end

function Y = dip_fft(X)
    % Radix-2 FFT algorithm
    N = length(X);
    if N <= 1
        Y = X;
    else
        X_even = dip_fft(X(1:2:end));
        X_odd = dip_fft(X(2:2:end));
        factor = exp(-2i * pi / N * (0:N/2-1));
        Y = [X_even + factor .* X_odd, X_even - factor .* X_odd];
    end
end

function I = dip_ifft2(FFT)
    % Check if input is complex, if not convert it to complex
    if ~isreal(FFT)
        FFT = complex(FFT);
    end

    % Compute 1D IFFT along columns
    I_col = zeros(size(FFT));
    for m = 1:size(FFT, 1)
        I_col(m, :) = dip_ifft(FFT(m, :));
    end
    
    % Compute 1D IFFT along rows
    I = zeros(size(FFT));
    for n = 1:size(FFT, 2)
        I(:, n) = dip_ifft(I_col(:, n).');
    end

    % Normalizing by 1/MN
    [m,n] = size(FFT);
    I = I./(m*n);
end

function Y = dip_ifft(X)
    % Radix-2 FFT algorithm
    N = length(X);
    if N <= 1
        Y = X;
    else
        X_even = dip_ifft(X(1:2:end));
        X_odd = dip_ifft(X(2:2:end));
        factor = exp(2i * pi / N * (0:N/2-1));
        Y = [X_even + factor .* X_odd, X_even - factor .* X_odd];
    end
end

% 1.2
function shiftedFFT = dip_fftshift(FFT)
    % Get size of the input FFT
    [M, N] = size(FFT);
    
    % Calculate the indices for shifting
    p = ceil(M/2);
    q = ceil(N/2);
    
    % Shift the quadrants
    shiftedFFT = FFT;
    shiftedFFT(1:p, 1:q) = FFT(p+1:end, q+1:end);
    shiftedFFT(p+1:end, q+1:end) = FFT(1:p, 1:q);
    shiftedFFT(1:p, q+1:end) = FFT(p+1:end, 1:q);
    shiftedFFT(p+1:end, 1:q) = FFT(1:p, q+1:end);
end

function norm_img = imread_double_norm(file_name)

    % Read the image file
    reg_img = imread(file_name);
    
    % Check if the image is grayscale or RGB
    if size(reg_img, 3) == 1
        % Grayscale image
        double_img = double(reg_img);
    else
        % RGB image
        double_img = double(reg_img) / 255; % Normalize each channel
    end
    
    % Normalize the image to the range [0, 1]
    norm_img = (double_img - min(double_img(:))) / (max(double_img(:)) - min(double_img(:)));
end
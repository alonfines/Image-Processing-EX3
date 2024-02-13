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

diff_val = abs(gray_beatles-real(ifft_beatles));
max_diff = max(max(diff_val));

%%------------------ Functions --------------------------------------------
% 1.1.1
function F = dip_fft2(I)
     % Get dimensions of the input matrix
    [M, N] = size(I);
    
    % Create matrices for row and column indices
    u = 0:M-1;
    v = 0:N-1;
    
    % Compute the Fourier matrix for rows and columns
    F_row = exp(-2i * pi / M * u' * u);
    F_col = exp(-2i * pi / N * v' * v);
    
    % Perform 2D DFT using matrix multiplication
    F = F_row * I * F_col;
end

function I = dip_ifft2(FFT)
    % Get dimensions of the input matrix
    [M, N] = size(FFT);
    
    % Create matrices for row and column indices
    u = 0:M-1;
    v = 0:N-1;
    
    % Compute the inverse Fourier matrix for rows and columns
    F_row = exp(2i * pi / M * u' * u);
    F_col = exp(2i * pi / N * v' * v);
    
    % Perform 2D IDFT using matrix multiplication
    I = (1/(M*N)) * F_row * FFT * F_col;
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
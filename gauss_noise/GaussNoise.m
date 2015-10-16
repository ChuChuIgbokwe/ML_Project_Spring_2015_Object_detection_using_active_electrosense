function [ volserror ] = GaussNoise( pNoise,loadfile,savefile)
%GaussNoise adds Gaussian random noise to the loadfile and saves it to
%savefile. The first argument should be an integer between 0 and 100. The
%second and third argument should be strings in the form of 'filename.mat'

%  the random noise is based on a Gaussion distribution with standard
%  deviation=pNoise% x (maximum value of the row of voltages)
%   Therefore the noise level is bounded and
%  can be controlled by the argument to the function pNoise

load(loadfile); %load the specified data file.

percentNoise=pNoise/100; %convert given noise level to percentage

for i=1:numel(vols(:,1))
    data=vols(i,:);
   for k=1:numel(data)
    data(k) = data(k) + random('Normal',0,percentNoise*max(data));
  end
  
  volserror(i,:) = data;
end

%newfile = 'circ_ar1_noise.mat';
save(savefile, 'volserror');
%plot the zero degree original data and noisy data
x=linspace(0,101,101);
hold all;
plot(x,volserror(1,:));
plot(x,vols(1,:));
end


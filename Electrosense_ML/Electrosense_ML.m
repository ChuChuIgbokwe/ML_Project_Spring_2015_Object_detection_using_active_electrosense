function Electrosense_ML()
%Set the directory where this file and the simulation data is located
directory='C:\Users\Kathleen\Documents\MATLAB\Final_Project\Electrosense_ML';

AddNoise(directory);   %Add Noise to the simulated data
FeatureExtract(directory);  %extract Amplitude data

end

function AddNoise(dirName)
    %Uses GaussNoise() to process all simulation file in directory and
    %create 10 noisy files
files = dir( fullfile(dirName,'circ_ar*.mat') );   %# list all *.mat files
files = {files.name}';                      %'# file names

data = cell(numel(files),1);                %# store file contents
GaussNoise(10,files{1},'test.mat');
for k=1:numel(files)
    fname = fullfile(dirName,files{k});     %# full path to file
    savename=fullfile(dirName,(files{k}(6:(numel(files{k})-4))));
     for i=1:10
       savefile=strcat(savename,'_Noise',num2str(i),'.mat');
       GaussNoise(10,fname,savefile);
    end
    
    end
end

function GaussNoise( pNoise,loadfile,savefile)
%GaussNoise adds Gaussian random noise to the loadfile and saves it to
%savefile. The first argument should be an integer between 0 and 100. The
%second and third argument should be strings in the form of 'filename.mat'

%  the random noise is based on a Gaussion distribution with standard
%  deviation=pNoise% x (maximum value of the row of voltages)
%   Therefore the noise level is bounded and
%  can be controled by the argument to the function pNoise

rawdata=load(loadfile); %load the specified data file.

percentNoise=pNoise/100; %convert given noise level to percentage

for i=1:numel(rawdata.vols(:,1))
    data=rawdata.vols(i,:);
   for k=1:numel(data)
    data(k) = data(k) + random('Normal',0,percentNoise*max(data));
  end
  
  volserror(i,:) = data;
end

%newfile = 'circ_ar1_noise.mat';
save(savefile, 'volserror');
clear;
end

function FeatureExtract(dirName)
for j=1:4
    arfile=strcat('ar',num2str(j),'*Noise*.mat');
files = dir( fullfile(dirName,arfile) );   %# list all *.mat files
files = {files.name}';                      %'# file names
data = cell(numel(files),1);                %# store file contents
for k=1:numel(files)
    fname = fullfile(dirName,files{k});     %# full path to file
    %clear data;
    data=load(fname);
      hipeak=max(data.volserror(1,:));      %this can b adjusted according
      lopeak=min(data.volserror(1,:));      %the format of the .mat file
      amp =(hipeak-lopeak)/2;
      AMPMAT(k,j)=amp;
end
end

AR=[1 2 3 4];
processedfile=fullfile(dirName,'ar1_ar2_ar3_ar4.mat');
save(processedfile,'AR', 'AMPMAT'); %output file
%clear;
    end
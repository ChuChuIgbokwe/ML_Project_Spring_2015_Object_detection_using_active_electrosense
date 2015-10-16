 function Electrosense_ML()
%Set the directory where this file and the simulation data is located
directory='C:\Users\Kathleen\Documents\MATLAB\Final_Project\Electrosense_Project';

%AddNoise(directory,'circ_ar*.mat');   %Add Noise to the simulated data
%AddNoise(directory,'clean*.mat');  %Add Noise to the AR2.5 data, the test set
%FeatureExtract(directory,'ar1_ar2_ar3_ar4.mat',[1 2 3 4]);  %extract Amplitude data
%FeatureExtract(directory,'ar2.5.mat',[2.5])

compareModels_poly(directory,'ar1_ar2_ar3_ar4_real.mat');
xstar = compareModels_root(directory,'ar1_ar2_ar3_ar4_real.mat')

determineAR(directory,'ar2.5_real.mat')


end
%***Data Preprocessing*******
function AddNoise(dirName,wildcard)
    %Uses GaussNoise() to process all simulation file in directory and
    %create 10 noisy files
files = dir( fullfile(dirName,wildcard) );   %# list all *.mat files
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

function FeatureExtract(dirName,savename,ar)
global AMPMAT AR
for j=1:(numel(ar))
    arfile=strcat('*ar',num2str(ar(j)),'*Noise*.mat');
files = dir( fullfile(dirName,arfile) );   %# list all *.mat files
files = {files.name}';                      %'# file names
data = cell(numel(files),1);                %# store file contents
clear AMPMAT;
clear AR;
for k=1:numel(files)
    fname = fullfile(dirName,files{k});     %# full path to file
    %clear data;
    data=load(fname);
      hipeak=max(data.volserror(1,:));      %this can b adjusted according
      lopeak=min(data.volserror(1,:));      %the format of the .mat file
      amp =(hipeak-lopeak)/2;
      AMPMAT(k,j)=amp;
     % AR(j)=ar(j);
end
end
 
 AR=ar;
 processedfile=fullfile(dirName,savename);
 save(processedfile,'AR', 'AMPMAT'); %output file
end
%******Data Preprocessing******
 
function testdata = load_data(dirName,filename)
    filepath = fullfile(dirName,filename);
    data=load(filepath);

    % Organize data
    mV_AMP = data.AMPMAT*1000; % use mV for amplitude as opposed to V
    AR = data.AR;
    testdata = [];
    for i=1:size(mV_AMP,2)
        dataARi = [AR(i)*ones(size(mV_AMP,1),1),mV_AMP(:,i)];
        testdata = [testdata; dataARi];
    end
end


%******Polynomial Modeling*****
function compareModels_poly(dirName,filename)
% Use cross validation (3-fold) to generate models for various polynomial
% fits for aspect ratio (AR) vs mV observed. 

% Variable - poly_degs gives maximum deg of poly
poly_degs = 1:3;

% load data etc.,
testdata = load_data(dirName,filename);
a = testdata(:,1);
b = testdata(:,2);

% split points into 3 equal sized sets and plot
c = split_data(a,b);

% do 3-fold cross-validation
cross_validate(a,b,c,poly_degs);  
end

%----------------------Local Functions ------------------------------------

% splits data into 3 training/testing sets
function c = split_data(a,b)
    % split data into 3 equal sized sets
    K = length(b);
    order = randperm(K);
    c = ones(K,1);
    K = round((1/3)*K);
    c(order(K+1:2*K)) =2;
    c(order(2*K+1:end)) = 3;

    % plot train/test sets for each cross-validation instance
    for j = 1:3
        a_r = a(find(c == j));
        b_r = b(find(c == j));
        a_t = a(find(c~=j));
        b_t = b(find(c~=j));
        
        figure(1)
        subplot(2,3,j)
        box on
        plot_pts(a_r,b_r,a_t,b_t)
        
        figure(3)
        subplot(1,3,j)
        box on
        plot_pts(a_r,b_r,a_t,b_t)
    end
end

%plot the data 
function plot_pts(a_r,b_r,a_e,b_e)
    % plot train
    hold on
    plot(a_e,b_e,'o','MarkerEdgeColor',[0.1 0.8 1],'MarkerFaceColor',[0.1 0.8 1],'MarkerSize',3)
    % plot test
    hold on
    plot(a_r,b_r,'o','MarkerEdgeColor',[1 0.7 0],'MarkerFaceColor',[1 0.7 0],'MarkerSize',3)
    set(gcf,'color','w');
    box on
    %axis square
end

%plot the polynomial
function plot_poly(x,color)
    model = [0:0.001:1];
    deg = length(x) - 1;
    out = [];
    for k = 1:deg;
        out = [out ; x(k+ 1)*model.^k];
    end
    out = sum(out,1) + x(1);
    plot(model,out,color,'LineWidth',1.25)
end

%perform cross validation
function cross_validate(a,b,c,poly_degs)  
    %%% performs 3-fold cross validation

    % solve for weights and collect test errors
    test_errors = [];
    train_errors = [];
    for i = 1:length(poly_degs)
        % generate features
        deg = poly_degs(i);
        A = [];
        for j = 1:deg
            A = [A  a.^j];
        end
        
        train_resids = [];
        test_resids = [];
        for j = 1:3
            A_1 = A(find(c ~= j),:);
            b_1 = b(find(c ~= j));
            A_2 = A(find(c==j),:);
            b_2 = b(find(c==j));
            A_1 = [ones(size(A_1,1),1) A_1];
            A_2 = [ones(size(A_2,1),1) A_2];
            x = linsolve(A_1,b_1);
            resid = norm(A_2*x - b_2)/numel(b_2);
            test_resids = [test_resids resid];
            resid = norm(A_1*x - b_1)/numel(b_1);
            train_resids = [train_resids resid];
        end
        test_errors = [test_errors; test_resids];
        train_errors = [train_errors; train_resids];
    end
    
    % find best parameter per data-split
    for i = 1:3
        
        %%% find the best performer (per split) and plot it %%%
        [val,j] = min(test_errors(:,i));
        
        % build features
        deg = poly_degs(j);
        A = [];
        for k = 1:deg
            A = [A  a.^k];
        end
        A_1= A(find(c ~= i),:);
        A_1 = [ones(size(A_1,1),1) A_1];
        b_1 = b(find(c ~= i));
        x =linsolve(A_1,b_1);   
        
        % output model
        figure(1)
        subplot(2,3,i) 
        hold on
        model =  [0:0.001:1]';
        out = [];
        for j = 1:deg;
            out = [out  x(j + 1)*model.^j];
        end
        out = sum(out,2) + x(1);
        plot(model,out,'b','LineWidth',1.25)
        
        %%% find the worst performer (per split) and plot it %%%
        [val,j] = max(test_errors(:,i));
        
        % build features
        deg = poly_degs(j);
        A = [];
        for k = 1:deg
            A = [A  a.^k];
        end
        A_1 = A(find(c ~= i),:);
        A_1 = [ones(size(A_1,1),1) A_1];
        b_1 = b(find(c ~= i));
        x =linsolve(A_1,b_1);        
        hold on
        
        % output model
        subplot(2,3,i)
        model =  [0:0.001:1]';
        out = [];
        for j = 1:deg;
            out = [out  x(j + 1)*model.^j];
        end
        out = sum(out,2) + x(1);
        hold on
        plot(model,out,'r','LineWidth',1.25)
                
        % clean up plot
        axis([0 1 -2 2])
        %axis square 
        box on
        xlabel('a','Fontsize',14,'FontName','cmmi9')
        ylabel('b','Fontsize',14,'FontName','cmmi9')
        set(get(gca,'YLabel'),'Rotation',0)
        set(gca,'YTickLabel',[])
        set(gca,'YTick',[])
        set(gcf,'color','w');
        set(gca,'FontSize',12); 
    end
    test_ave = mean(test_errors');
    [val,j] = min(test_ave);
    j
    % build features
    deg = poly_degs(j);
    A = [];
    for k = 1:deg
        A = [A  a.^k];
    end
    A = [ones(size(A,1),1) A];
    x =linsolve(A,b);
    
    % output model
    model =  [0:0.001:1]';
    out = [];
    for j = 1:deg;
        out = [out  x(j + 1)*model.^j];
    end
    out = sum(out,2) + x(1);
    hold on
    subplot(2,3,5)
    plot(model,out,'m','LineWidth',1.25)
    
  % clean up plot
    axis([0 1 -2 2])
    %axis square 
    box on
    xlabel('a','Fontsize',14,'FontName','cmmi9')
    ylabel('b','Fontsize',14,'FontName','cmmi9')
    set(get(gca,'YLabel'),'Rotation',0)
    set(gca,'YTickLabel',[])
    set(gca,'YTick',[])
    set(gcf,'color','w');
    set(gca,'FontSize',12); 

    % plot training and testing errors
    figure(2)
    plot(1:max(poly_degs),mean(test_errors'),'--','Color', [1 0.7 0])
    hold on
    plot(1:max(poly_degs),mean(train_errors'),'--','Color',[0.1 0.8 1])
    hold on
    plot(1:max(poly_degs),mean(test_errors'),'o','MarkerEdgeColor',[1 0.7 0],'MarkerFaceColor',[1 0.7 0])
    hold on
    plot(1:max(poly_degs),mean(train_errors'),'o','MarkerEdgeColor',[0.1 0.8 1],'MarkerFaceColor',[0.1 0.8 1])
   
    legend('testing error','training error')
    set(gcf,'color','w');
    set(gca,'FontSize',12); 
    set(gca,'xtick',0:max(poly_degs))
    box on
    %axis square
    axis([0.5 10 0 max(mean(test_errors'))])
    
    xlabel('polynomial degree','Fontsize',14,'FontName','cmr10')
    ylabel('error','Fontsize',14,'FontName','cmr10')
    set(get(gca,'YLabel'),'Rotation',90)
    set(gca,'YTickLabel',[])
    set(gca,'YTick',[])
    set(gcf,'color','w');
    set(gca,'FontSize',12); 
end

%*******End Polynomial Modeling***

%*******Root Model fitting*******
function xstar = compareModels_root(dirName,filename)
testdata = load_data(dirName,filename); %from another .m file

figure(4)
plot(testdata(:,1),testdata(:,2),'.');
axis([0 5 0.3 1]);
title('Training Data');

figure(5)
xstar1 = linear_fitting(testdata(:,1),testdata(:,2));

figure(6)
xstar2 = sqrt_fitting(testdata(:,1),testdata(:,2));

%chosen model coefficients: 
xstar = xstar1;
end 

%%%--------------Local Functions-------------------------------------------

%linear fitting: equation of y = x1*a + x0
function xstar = linear_fitting(a,b)

D = [ones(size(a)), a];
    
%{
%check to verify that cost function will be concave
Hessian = 2*(D')*D;
eig(Hessian);        %Eigenvalues are nonnegative
%}

%Least square optimization problem
xstar = pinv(D'*D)*D'*b;

%plot new dataline for model
dataline = (0:0.01:5);
Dnew = [ones(size(dataline))', dataline'];
b_new = Dnew*xstar;

%error squared (original model's error): 
error = sum((D*xstar-b).^2);

plot(dataline, b_new);
hold on;
plot(a,b,'.');
title(sprintf('Original Space - Linear fitting\nError: %f', error));
xlabel('Aspect Ratio');
ylabel('Amplitude (mV)');
hold off;
end 

%square root fitting: equation of b = x1*sqrt(a) + x0
function xstar = sqrt_fitting(a,b)

D = [ones(size(a)), sqrt(a)];
    
%{
%check to verify that cost function will be concave
Hessian = 2*(D')*D;
eig(Hessian);        %Eigenvalues are nonnegative
%}

%Least square optimization problem
xstar = pinv(D'*D)*D'*b;

%plot new dataline for model in the ORIGINAL space
dataline = (0:0.01:5);
Dnew = [ones(size(dataline))', (dataline').^(1/2)];
b_new = Dnew*xstar;

%error squared (original model's error): 
error = sum((D*xstar-b).^2);

plot(dataline, b_new);
hold on;
plot(a,b,'.');
title(sprintf('Original Space - Square root fitting\nError: %f', error));
xlabel('Aspect Ratio');
ylabel('Amplitude (mV)');
hold off;
end 

%******End Root Model fitting*****

% Determine the Aspect Ratio (AR) of an object with unknown AR, based on
% the object's electronic signature in mV. In this example, we know the AR
% to be 2.5. Let's see how close our model can predict this value. 

function determineAR(dirName,testfile)
% HARD CODED - this is the set of coefficients for the linear model
% that we picked from all the models
xstar = [0.1652 0.1860];

bias = xstar(1);
slope = xstar(2);
data = load(fullfile(dirName,testfile));%'ar2.5.mat');
mV_AMP = data.AMPMAT*1000;

%using mV_AMP = x1*AR + x0, we can estimate what the AR will be for a given
%mV_AMP value. (x0 = bias, x1 = slope)
AR = [];
for i=1:size(mV_AMP,1)
ARi = (mV_AMP(i) - bias)/slope;
AR = [AR ARi];
end 

%average the data to get our estimated aspect ratio: 
estimated_AR = sum(AR)/10
end
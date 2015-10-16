% Determine the Aspect Ratio (AR) of an object with unknown AR, based on
% the object's electronic signature in mV. In this example, we know the AR
% to be 2.5. Let's see how close our model can predict this value. 

function determineAR()
% HARD CODED - this is the set of coefficients for the linear model
% that we picked from all the models
xstar = [0.1652 0.1860];

bias = xstar(1);
slope = xstar(2);
data = load('ar2.5.mat');
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
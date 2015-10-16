function xstar = compareModels_root()
testdata = load_data(); %from another .m file

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
function testdata = load_data()
    data = load('C:\Users\Sandra\Dropbox\EECS495_project\cross_validation\ar1_ar2_ar3_ar4.mat'); 

    % Organize data
    mV_AMP = data.AMPMAT*1000; %use mV for amplitude as opposed to V
    AR = data.AR;
    testdata = [];
    for i=1:size(mV_AMP,2)
        dataARi = [AR(i)*ones(size(mV_AMP,1),1),mV_AMP(:,i)];
        testdata = [testdata; dataARi];
    end
end
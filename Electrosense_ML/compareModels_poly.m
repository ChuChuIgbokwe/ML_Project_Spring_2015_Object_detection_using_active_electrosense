function compareModels_poly()
% Use cross validation (3-fold) to generate models for various polynomial
% fits for aspect ratio (AR) vs mV observed. 

% Variable - poly_degs gives maximum deg of poly
poly_degs = 1:3;

% load data etc.,
testdata = load_data();
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
lb = [-1,0, -1, 0];
ub = [1,300, 1, 300];
A = [];
b = [];
Aeq = [];
beq = [];
% x0 = [cos(1.721), 0.01, cos(0.988), 0.01]; %cos, ns
x0 = [cos(1.521), 175, cos(1.04), 0.01]; %cos, ns
options = optimoptions('fmincon','Display','iter');
tic;
[x,fval,exitflag,output] = fmincon(@obj_func,x0,A,b,Aeq,beq,lb,ub, [], options);
toc;

fprintf('AoA (deg): ');
disp(rad2deg(acos(x(1:2:end))));
fprintf('ToF (ns): ');
disp(x(2:2:end));

% tofs = 0:0.5:100;
% for ii=1:length(tofs)
%     cost(ii) = obj_func([cos(2.015), tofs(ii)]);
% end
% figure;
% plot(cost);


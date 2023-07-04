%unction[x1,y1,x2,y2] = project_pat_rec()
x1 = 2 + 6*rand(400,1);
y1 = 1 + 1*rand(400,1);
figure(1)
scatter(x1,y1,'b')
hold on
x2 = 6 + 2*rand(100,1);
y2 = 2.5 + 3*rand(100,1);
scatter(x2,y2,'r')
ylim([0,10])
xlim([0,10])
title('Plot of uniformly distributed elements')
legend({'400 elements of class 1','100 elements of class 2'},'Location','northwest') 
class1 = transpose([x1(:),y1(:)]);
class2 = transpose([x2(:),y2(:)]);
hold on
%B1Func(x1,x2,y1,y2,class1,class2);




%B1-MAXIMUM LIKELIHOOD ESTIMATORS
%The following equations are the result of ?L/?? and ?L/?(?^2)
mean1 = (1/400)*sum(class1,2);
mean2 = (1/100)*sum(class2,2);
plot(mean1(1),mean1(2),'kx');
plot(mean2(1),mean2(2),'kx');
sub1 = class1 - mean1;%Subtract for mean1[1] and mean1[2] x1 and y1
sub1sq  = power(sub1,2);
sub2 = class2 - mean2;
sub2sq = power(sub2,2);
%covariance1 = (1/399)*sum(sub1sq,2);
%covariance2 = (1/99)*sum(sub2sq,2);
%covariance1b = (1/400)*sum(sub1sq,2);
%covariance2b= (1/100)*sum(sub2sq,2);
covMatrix1 = (1/400)*sub1*transpose(sub1);
covMatrix2 = (1/100)*sub2*transpose(sub2);
%B2 EUCLEDIAN DISTANCE
%For 1st class
%dist1mean1 = sqrt(sum(sub1sq));
sub12 = class1 - mean2;
sub12sq = power(sub12,2);
%dist1mean2 = sqrt(sum(sub12sq));
numWrongClassification1 = 0;
for i = 1:400
  dist1mean1 = norm(class1(:,i) - mean1);
  dist1mean2 = norm(class1(:,i) - mean2);
  if(dist1mean1 > dist1mean2)
  %disp(i)
  plot(class1(2*i-1),class1(2*i),'rx')
    ++numWrongClassification1;
  end
end
%disp(class1(4));
wrongClassification1 = (numWrongClassification1/400)*100;
hold on
%For 2nd class
%dist2mean2 = sqrt(sum(sub2sq));
sub21 = class2 - mean1;
sub21sq = power(sub21,2);
%%dist2mean1 = sqrt(sum(sub21sq));
numWrongClassification2 = 0;
for i = 1:100
  dist2mean2 = norm(class2(:,i) - mean2);
  dist2mean1 = norm(class2(:,i) - mean1);
  if(dist2mean1 < dist2mean2)
  %disp(i)
  plot(class2(2*i-1),class2(2*i),'bx')
    ++numWrongClassification2;
  end
end
wrongClassification2 = (numWrongClassification2/100)*100;
hold off




%B3 MAHALANOBIS DISTANCE
figure(2)
scatter(x1,y1,'b')
hold on
scatter(x2,y2,'r')
ylim([0,10])
xlim([0,10])
title('Mahalanobis Distance classification')
legend({'Blue x: Wrongly classified elements of classs 2 to class 1','Red x: Wrongly classified elements of classs 1 to class 2'},'Location','northwest')
hold on
sw = (covMatrix1 + covMatrix2)/2;
swInv = inv(sw);
%For 1st class
sub1T = transpose(sub1);
mahalanobisDist11 = zeros(1,400);
for i =1:400
 mahalanobisDist11(:,i) = power((sub1T(i,:)*swInv*sub1(:,i)),0.5);
end

sub12T = transpose(sub12);
mahalanobisDist12 = zeros(1,400);
for i =1:400
 mahalanobisDist12(:,i) = power((sub12T(i,:)*swInv*sub12(:,i)),0.5);
end
mahWrongClssfctn1 = 0;
for i = 1:400
  if(mahalanobisDist11(i) > mahalanobisDist12(i))
  disp(i)
  plot(class1(2*i-1),class1(2*i),'rx')
    ++mahWrongClssfctn1;
  end
end
wrngPecentage1 = (mahWrongClssfctn1/400)*100;
hold on
%For 2nd class
sub2T = transpose(sub2);
mahalanobisDist22 = zeros(1,100);
for i =1:100
 mahalanobisDist22(:,i) = power((sub2T(i,:)*swInv*sub2(:,i)),0.5);
end

sub21T = transpose(sub21);
mahalanobisDist21 = zeros(1,100);

for i =1:100
 mahalanobisDist21(:,i) = power((sub21T(i,:)*swInv*sub21(:,i)),0.5);
end
mahWrongClssfctn2 = 0;
for i = 1:100
  if(mahalanobisDist22(i) > mahalanobisDist21(i))
  %disp(i)
  plot(class2(2*i-1),class2(2*i),'bx')
    ++mahWrongClssfctn2;
  end
end
wrngPercentage2 = (mahWrongClssfctn2/100)*100;
hold off





%B4 BAYESIAN CLASSIFIER
figure(3)
scatter(x1,y1,'b')
hold on
scatter(x2,y2,'r')
ylim([0,10])
xlim([0,10])
title('Bayesian classification')
legend({'Blue x: Wrongly classified elements of classs 2 to class 1','Red x: Wrongly classified elements of classs 1 to class 2'},'Location','northwest')
hold on
covMatrix1Inv = inv(covMatrix1);
covMatrix2Inv = inv(covMatrix2);

bayesianWrngClssfd1 = 0;
pdf11 = zeros(1,400);
pdf12 = zeros(1,400);
for i =1:400
  pdf11(:,i) = (1/((2*pi)*power(det(covMatrix1),0.5)))*exp(-0.5*(sub1T(i,:)*covMatrix1Inv*sub1(:,i)));
  pdf12(:,i) = (1/((2*pi)*power(det(covMatrix2),0.5)))*exp(-0.5*(sub12T(i,:)*covMatrix2Inv*sub12(:,i)));
end

for i = 1:400
  if(pdf11(i) < pdf12(i))
  disp(i)
  plot(class1(2*i-1),class1(2*i),'rx')
  ++bayesianWrngClssfd1;
  end
end
wrngPecentage1 = (bayesianWrngClssfd1/400)*100;
hold on
bayesianWrngClssfd2 = 0;
pdf22 = zeros(1,100);
pdf21 = zeros(1,100);
for i =1:100
  pdf22(:,i) = (1/((2*pi)*power(det(covMatrix2),0.5)))*exp(-0.5*(sub2T(i,:)*covMatrix2Inv*sub2(:,i)));
  pdf21(:,i) = (1/((2*pi)*power(det(covMatrix1),0.5)))*exp(-0.5*(sub21T(i,:)*covMatrix1Inv*sub21(:,i)));
end

for i = 1:100
  if(pdf22(i) < pdf21(i))
  %disp(i)
  plot(class2(2*i-1),class2(2*i),'bx')
    ++bayesianWrngClssfd2;
  end
end
wrngPercentage2 = (bayesianWrngClssfd1/100)*100;

hold off






%C1 PCA ANALYSIS

mtr = [class1 class2];

mean = (1/500)*sum(mtr,2);

class1shifted = class1 - mean;
class2shifted = class2 - mean;
x1shifted = x1 - mean(1);
y1shifted = y1 - mean(2);

x2shifted = x2 - mean(1);
y2shifted = y2 - mean(2);

combMatrix = [class1shifted class2shifted];
figure(4)
scatter(x1shifted,y1shifted, 'b');
hold on
scatter(x2shifted,y2shifted,'r');
ylim([-5,5])
xlim([-5,5])
title('PCA')

%legend('elements of class1','elements of class2','AutoUpdate','on');
hold on
autocorrelationMatrix = (1/500)*combMatrix * transpose(combMatrix);
[eigenvectors,eigenvalues] = eig(autocorrelationMatrix);

if(eigenvalues(1)>eigenvalues(4))
  wProj = eigenvectors(:,1)
else
  wProj = eigenvectors(:,2)
end


wProjLength = sqrt(power(wProj(1),2) + power(wProj(2),2));

projMatrix = zeros(2,500);
for i = 1:500
  %projMatrix(:,i) = (transpose(wProj)*combMatrix(:,i))/wProjLength;
  projMatrix(:,i) = (dot(combMatrix(:,i),wProj)/norm(wProj)^2)*wProj;
end
%pause(2);
for i = 1:500 
  if(i <= 400)
  plot(projMatrix(2*i-1),projMatrix(2*i),'gx')
  else
  plot(projMatrix(2*i-1),projMatrix(2*i),'mx')
  end
end

%plot(projMatrix(999),projMatrix(1000),'kx');
%legend('Projected elements of class1','Projected elements of class2');

%C2 CLASSIFICATION BASED ON THE EUCLEDIAN DISTANCE
meanOfprojElem1 = (1/400)*sum(projMatrix(:,[1:400]),2);
meanOfprojElem2 = (1/100)*sum(projMatrix(:,[401:500]),2);
%pause(2);
for(i = 1:400)
dist11 = norm(projMatrix(:,i) - meanOfprojElem1);
dist12 = norm(projMatrix(:,i) - meanOfprojElem2);
if(dist11 > dist12)
  disp(i)
  plot(projMatrix(2*i-1),projMatrix(2*i),'cx')
end
end

for(i = 401:500)
dist21 = norm( projMatrix(:,i) - meanOfprojElem2);
dist22 = norm( projMatrix(:,i) - meanOfprojElem2);
if(dist22 > dist21)
  disp(i)
  plot(projMatrix(2*i-1),projMatrix(2*i),'kx')
end
end

%legend({'Wrongly classified elements of class1','Wrongly classified elements of class1'});
hold off

%C3 LDA ANALYSIS
figure(5)
scatter(x1,y1, 'b');
hold on
scatter(x2,y2,'r');
ylim([0,10])
xlim([0,10])
title('LDA')
hold on
wProjLDA = swInv*(mean1 - mean2);
projMatrixLDA1 = zeros(2,400);
projMatrixLDA2 = zeros(2,100);
for i = 1:400
    projMatrixLDA1(:,i) = (dot(class1(:,i),wProjLDA)/norm(wProjLDA)^2)*wProjLDA;
    %plot(projMatrixLDA(2*i-1),projMatrixLDA(2*i),'gx');
end

for i = 1:100
    projMatrixLDA2(:,i) = (dot(class2(:,i),wProjLDA)/norm(wProjLDA)^2)*wProjLDA;
   % plot(projMatrixLDA(2*i-1),projMatrixLDA(2*i),'mx');
end


for i = 1:400 
  plot(projMatrixLDA1(2*i-1),projMatrixLDA1(2*i),'gx')
end
for i = 1:100 
  plot(projMatrixLDA2(2*i-1),projMatrixLDA2(2*i),'mx')
end

%C4 LDA CLASSIFICATION

meanOfprojElem1 = (1/400)*sum(projMatrixLDA1,2);
meanOfprojElem2 = (1/100)*sum(projMatrixLDA2,2);
%pause(2);
for(i = 1:400)
dist11 = norm(projMatrixLDA1(:,i) - meanOfprojElem1);
dist12 = norm(projMatrixLDA1(:,i) - meanOfprojElem2);
if(dist11 > dist12)
  disp(i)
  plot(projMatrixLDA1(2*i-1),projMatrixLDA1(2*i),'cx')
end
end

for(i = 1:100)
dist21 = norm( projMatrixLDA2(:,i) - meanOfprojElem2);
dist22 = norm( projMatrixLDA2(:,i) - meanOfprojElem2);
if(dist22 > dist21)
  disp(i)
  plot(projMatrixLDA2(2*i-1),projMatrixLDA2(2*i),'kx')
end
end

hold off






%D1 LEAST SQUARE ERRORS
figure(6)
scatter(x1,y1, 'b');
hold on
scatter(x2,y2,'r');
ylim([0,10])
xlim([0,10])
title('LSE')
hold on


y_labels = [ones(1,400) -ones(1,100)];
helpMatrix = [1 0;0 1;0 0];
comb_classes = [class1 class2];
increasedMatrix = helpMatrix * comb_classes;
X = increasedMatrix + [0;0;1];
weights = inv(X*transpose(X))*X*transpose(y_labels);
%X = [class1 class2];
x_lse = linspace(0,1000);
y_lse = (weights([1])/(-weights([2])))*x_lse + weights([3])/(-weights([2]));
%e = weights*t;
plot(x_lse,y_lse);
res = transpose(weights)*X;
%J = sum(power((transpose(y_labels) -  transpose(res)),2));
hold on

for(i = 1:400)
if(res([i]) < 0)
  plot(class1(2*i-1),class1(2*i),'rx')
end
end
for(i = 1:100)
if(res([400+i]) > 0)
  plot(class2(2*i-1),class2(2*i),'bx')
end
end

hold off










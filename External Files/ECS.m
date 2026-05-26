clear all
close all
DATA1=importdata("PL_2.86GeV_240904.dat");
% DATA2=importdata("PL_V1_TestA25.dat");
% DATA3=importdata("PL_V1_240603.dat");
V0=150
r56=-0.22
deltaE0=57.2
dati1=DATA1.data;
X=dati1(:,1);
XP=dati1(:,2);
Y=dati1(:,3);
YP=dati1(:,4);


Z=(dati1(:,5)-1*mean(dati1(:,5)))*10^-3;
% FIRST rf bucket
scelta2=find(Z<0.011&Z>-0.015);
P=dati1(:,6);
histogram2(Z,P,[200,200],'FaceColor','flat')

Z1=Z+r56*(P-mean(P))/mean(P);
pi=3.14159;
E0=linspace(2000,3000,1000);
P1=P+V0*sin(2*pi*Z1/0.1499)
for ii=1:1000
    scelta = find(P1<E0(ii)+deltaE0&P1>E0(ii)-deltaE0);
    [a,b]=size(scelta);
    number1(ii)=a;
    POPOL{ii}=scelta;
end
for ii=1:1000
    scelta = find(P<E0(ii)+deltaE0&P>E0(ii)-deltaE0);
    [a,b]=size(scelta);
    number(ii)=a;
end
plot(E0,number/max(size(P)))
hold on
plot(E0,number1/max(size(P)))
xlabel('E0 (MeV)')
ylabel('Fraction of the beam in E0+/-deltaE0')
legend ('No ECS','ECS')
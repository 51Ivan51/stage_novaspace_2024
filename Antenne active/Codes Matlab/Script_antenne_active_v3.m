clc
clear all
close all
%% Generation des coordonnées UV


UV.nbMappingSample = 201; % Number of mapping samples 
UV.thetaMax = 90;  % Max angle of interest (°)
UV.varUV = sind(UV.thetaMax); %limits for the far field observation in UV coordinates
UV.u = linspace(-UV.varUV,UV.varUV,UV.nbMappingSample);
UV.v = UV.u;
[UV.gridU,UV.gridV] = meshgrid(UV.u,UV.v);
U=UV.gridU;
V=UV.gridV;

UU=reshape(U,1,201*201);
VV=reshape(V,1,201*201);
[~,index_reduced] = find(sqrt(UU.^2 + VV.^2)<0.81);  %contient les indices des points à l'intérieur du cercle de rayon 0.81.
indexes=1:(201*201);
removed_indexes = setdiff(indexes,index_reduced);


x=transpose(0:360);
circle=0.8121*[sind(x),cosd(x)];

% Grapique 2D des coordonnés du cercle
circle_x=circle(:,1);
circle_y=circle(:,2);
figure; 
plot(circle_x, circle_y); 
title('Graphique 2D des coordonnées du cercle'); 
grid on; 



%% CONSTRUCTION GRDS 

load('GRD_INITIAL.mat');
load('GEOMETRY');
ANTENNA_GEOMETRY
% Plot antenna geometry
figure; 
plot(ANTENNA_GEOMETRY(:,2),ANTENNA_GEOMETRY(:,3),'o')
xlabel('X'); 
ylabel('Y');
title('Graphique 2D des coordonnées de l antenne'); 
grid on;



k = 2*pi;
etaelm = 0.85;
celem = 58.9;
tmp = celem/(0.6369*etaelm);
tmp = celem/(.5*etaelm);
elm    = 12./(tmp.^2);
 
Constante_UV = 10.^((-elm*(asind(  sqrt((U).^2+(V).^2)   )).^2));

GRDS=zeros(522,201,201);
factor_scaling = 100/(2*pi);


for i=1:522
    xi = ANTENNA_GEOMETRY(i,2)/factor_scaling;
    yi = ANTENNA_GEOMETRY(i,3)/factor_scaling;
    dUV = exp(2*pi*1i*(xi.*U + yi.*V ));
    GRDS(i,:,:) = GRD_INIT.*dUV.*Constante_UV;
end


% Plot juste le gain de chaque antenne
coords = ANTENNA_GEOMETRY(:,2:3);  
weights =ANTENNA_GEOMETRY(:,4);
figure;
scatter(coords(:, 1), coords(:, 2), 50, weights, 'filled');
colorbar;  
xlabel('Coordonnée X');
ylabel('Coordonnée Y');
title('Gain de chaque antenne');
c = colorbar;
c.Label.String = 'Poids';
grid on;


% Affichage de GRD_INIT
figure;
imshow(real(GRD_INIT), []);
colorbar; 
xlabel('X');
ylabel('Y');
title('GRD_INIT');

% Affichage de GRDS d'une antenne
figure;
imshow(real(squeeze(GRDS(100,:,:))), []);
colorbar; 
xlabel('X');
ylabel('Y');
title('GRDS d une antenne');


%% TEST BEAM AT NADIR
w=ANTENNA_GEOMETRY(:,4);

dU=0.3;
dV=0.2;
w= exp( 1i*2*pi*(-ANTENNA_GEOMETRY(:,2)/10 * dU + ANTENNA_GEOMETRY(:,3)/10 * dV ) ) ;

GRD_resultant = zeros(201,201);

for i=1:522
    GRD_resultant = GRD_resultant + w(i) .* squeeze(GRDS(i,:,:));
end


% Plot Target
pcolor(linspace(-1,1,201),linspace(-1,1,201),20*log10(abs(GRD_resultant)))
caxis([7 50])
shading interp
hold on
plot(circle(:,2),circle(:,1),'r','linewidth',3)






%% DEFINITION SCENARIOS
% Position des beams _ pb
% Position des porteuses _ pp
% Backoff _ bo


%scenarios = dlmread('scenario_batch.txt');









%% INITIALISATION DES POIDS DE N BEAMS - FULL SPECTRUM
factor_scaling = 100/(2*pi);
transfer_matrix_feeds_grd=reshape(GRDS,522,201*201);
%Beam_positions=[-0.4 -0; -0.2 0 ; 0.2 0; 0.4 0; 0 -0.4 ; 0 -0.2 ; 0 0.2; 0 0.4 ];

i=68;
Nbeams =scenarios(i,1);

Beam_positions= reshape(scenarios(i,2:Nbeams*2+1),Nbeams,2);
CF= [repmat(0.85e9,8,1);repmat(1.7e9,8,1)];
GAIN = -12;

[a,b] = size(Beam_positions);
Nbeams = a;

weights=[];
% BEAM1
for i=1:Nbeams
dU=Beam_positions(i,1);
dV=Beam_positions(i,2);
w=ANTENNA_GEOMETRY(:,4);
w= w .* exp( 1i*2*pi*(-ANTENNA_GEOMETRY(:,2)/factor_scaling * dU + ANTENNA_GEOMETRY(:,3)/factor_scaling *dV) )
weights=[weights,w];
end

weights_sspa=weights;

%% DEFINITION COURONNES

mapping_crown1 = CROWNS(:,1);
mapping_crown2 = CROWNS(:,2);
mapping_crown3 = CROWNS(:,3);


%% PLOT RESULTS 

%display_func= @surf;
display_func= @pcolor;
% display_func= @contourf;

scaleUV = linspace(-1,1,201);
%[a,b] = size(signal_recu_full_spectrum);
%XX= sum(signal_recu_full_spectrum(15:end,:),1)/a;
% figure(1)
% display_func(scaleUV ,scaleUV ,20*log10(abs(squeeze(reshape(signal_recu_full_spectrum(end,:,:),201,201)))));
% %pcolor(scaleUV ,scaleUV ,20*log10(abs(squeeze(reshape(XX,201,201)))));
% shading interp;
% caxis([20 50])
% maxi_power = max(max(20*log10(abs(squeeze(reshape(signal_recu_full_spectrum(end,:,:),201,201))))));
% %maxi_power = max(max(20*log10(abs(squeeze(reshape(XX,201,201))))));
% caxis([maxi_power - 30 maxi_power])
% 
% colormap('jet')
% hold on
% plot(circle(:,2),circle(:,1),'r','Linewidth',3)
% title('Full Spectrum Power')


figure(2)
display_func(scaleUV ,scaleUV ,20*log10(abs(squeeze(reshape(signal_recu_bande1(end,:,:),201,201)))));
%pcolor(scaleUV ,scaleUV ,20*log10(abs(squeeze(reshape(XX,201,201)))));
shading interp;
caxis([20 50])
maxi_power = max(max(20*log10(abs(squeeze(reshape(signal_recu_bande1(end,:,:),201,201))))));
%maxi_power = max(max(20*log10(abs(squeeze(reshape(XX,201,201))))));
caxis([maxi_power - 30 maxi_power])
caxis([12 42])

colormap('jet')
hold on
plot(circle(:,2),circle(:,1),'r','Linewidth',3)
title('17.8 - 18.6 GHz')

figure(3)
display_func(scaleUV ,scaleUV ,20*log10(abs(squeeze(reshape(signal_recu_bande2(end,:,:),201,201)))));
%pcolor(scaleUV ,scaleUV ,20*log10(abs(squeeze(reshape(XX,201,201)))));
shading interp;
caxis([20 50])
maxi_power = maxi_power-6;
%maxi_power = max(max(20*log10(abs(squeeze(reshape(XX,201,201))))));
caxis([maxi_power - 30 maxi_power])
caxis([12 42])

colormap('jet')
hold on
plot(circle(:,2),circle(:,1),'r','Linewidth',3)
title('18.6 - 18.8 GHz')

figure(4)
display_func(scaleUV ,scaleUV ,20*log10(abs(squeeze(reshape(signal_recu_bande3(end,:,:),201,201)))));
%pcolor(scaleUV ,scaleUV ,20*log10(abs(squeeze(reshape(XX,201,201)))));
shading interp;
caxis([20 50])
maxi_power = max(max(20*log10(abs(squeeze(reshape(signal_recu_bande3(end,:,:),201,201))))));
%maxi_power = max(max(20*log10(abs(squeeze(reshape(XX,201,201))))));
caxis([maxi_power - 30 maxi_power])
caxis([12 42])

colormap('jet')
hold on
plot(circle(:,2),circle(:,1),'r','Linewidth',3)
title('18.8 - 19.3 GHz')

figure(5)
display_func(scaleUV ,scaleUV ,20*log10(abs(squeeze(reshape(signal_recu_bande4(end,:,:),201,201)))));
%pcolor(scaleUV ,scaleUV ,20*log10(abs(squeeze(reshape(XX,201,201)))));
shading interp;
caxis([20 50])
maxi_power = max(max(20*log10(abs(squeeze(reshape(signal_recu_bande4(end,:,:),201,201))))));
%maxi_power = max(max(20*log10(abs(squeeze(reshape(XX,201,201))))));
caxis([maxi_power - 30 maxi_power])
caxis([12 42])

colormap('jet')
hold on
plot(circle(:,2),circle(:,1),'r','Linewidth',3)
title('19.3 - 19.7 GHz')

figure(6)
display_func(scaleUV ,scaleUV ,20*log10(abs(squeeze(reshape(signal_recu_bande5(end,:,:),201,201)))));
%pcolor(scaleUV ,scaleUV ,20*log10(abs(squeeze(reshape(XX,201,201)))));
shading interp;
caxis([20 50])
maxi_power = max(max(20*log10(abs(squeeze(reshape(signal_recu_bande5(end,:,:),201,201))))));
%maxi_power = max(max(20*log10(abs(squeeze(reshape(XX,201,201))))));
caxis([maxi_power - 30 maxi_power])
caxis([12 42])

colormap('jet')
hold on
plot(circle(:,2),circle(:,1),'r','Linewidth',3)
title('19.7 - 20.2 GHz')

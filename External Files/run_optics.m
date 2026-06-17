RF_Track;

do_RF = false;

% Define Lattice
L = Lattice('dr.tws');

% Define bunch
T = Bunch6d_twiss();
T.beta_x = 9.989319917;
T.alpha_x = -0.0002540930971;
T.beta_y = 2.418837296;
T.alpha_y = 7.379939621e-05;
T.sigma_t = 0; % mm/c
T.sigma_pt = 0; % permille
T.emitt_x = 0.001; % mm.mrad
T.emitt_y = 0.001; % mm.mrad

Pref = 2859.999954; % MeV/c
B0 = Bunch6d_QR (RF_Track.electronmass, 0.0, +1, Pref, T, 1000);

% Remove the RF structure
if do_RF == false
    RF = L.get_rf_elements();
    for rf = RF
        rf{1}.replace_with(Drift(rf{1}.get_length()));
    end
end

% Tracking
tic
B1 = L.track(B0);
toc

% Plots
T = L.get_transport_table('%S %emitt_x %emitt_y');
B = L.get_transport_table('%S %beta_x %beta_y');
X = L.get_transport_table('%S %mean_x %mean_y');

%% Plots
figure(1)
clf
plot(T(:,1), T(:,2), 'b-', 'linewidth', 2, ...
     T(:,1), T(:,3), 'r-', 'linewidth', 2);
legend('emitt x', 'emitt y');
xlabel('S [m]');
ylabel('emitt [mm.mrad]');
print -dpng -S'1200,800' -F:12 plot_FCCee_DR_emitt.png

figure(2)
clf
plot(B(:,1), B(:,2), 'b-', 'linewidth', 2, ...
     B(:,1), B(:,3), 'r-', 'linewidth', 2);
legend('\beta x', '\beta y');
xlabel('S [m]');
ylabel('\beta [m]');
print -dpng -S'1200,800' -F:12 plot_FCCee_DR_beta.png

figure(3)
clf
hold on
scatter(B0.get_phase_space('%x'), B0.get_phase_space('%Px'));
scatter(B1.get_phase_space('%x'), B1.get_phase_space('%Px'), 'r');
title('initial and final horizontal phase space');
xlabel('x [mm]');
ylabel('P_x [MeV/c]');

figure(4)
clf
hold on
scatter(B0.get_phase_space('%y'), B0.get_phase_space('%Py'));
scatter(B1.get_phase_space('%y'), B1.get_phase_space('%Py'), 'r');
title('initial and final vertical phase space');
xlabel('y [mm]');
ylabel('P_y [MeV/c]');

figure(5)
clf
hold on
scatter(B0.get_phase_space('%dt'), B0.get_phase_space('%E'));
scatter(B1.get_phase_space('%dt'), B1.get_phase_space('%E'), 'r');
title('initial and final vertical phase space');
xlabel('dt [mm/c]');
ylabel('E [MeV]');

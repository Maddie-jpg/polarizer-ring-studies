import xtrack as xt
import numpy as np

# ============================================================
# Environment
# ============================================================

env = xt.Environment()

env.particle_ref = xt.Particles(
    mass0=xt.ELECTRON_MASS_EV,
    kinetic_energy0=3e9,
)

# ============================================================
# Ring parameters
# ============================================================

Ncells = 24

Lb = 0.4
Lq = 0.2
Ld = 0.5

theta = 2*np.pi/(2*Ncells)

# ============================================================
# Variables
# ============================================================

env.vars({
    'kQF': 3.0,
    'kQD': -2.5,
})

# ============================================================
# Elements
# ============================================================

env.new(
    'QF',
    xt.Quadrupole,
    length=Lq,
    k1='kQF'
)

env.new(
    'QD',
    xt.Quadrupole,
    length=Lq,
    k1='kQD'
)

env.new(
    'DR',
    xt.Drift,
    length=Ld
)

env.new(
    'B',
    xt.Bend,
    length=Lb,
    angle=theta,
    k0_from_h=True,
    edge_entry_angle=theta/2,
    edge_exit_angle=theta/2,
    edge_entry_model='full',
    edge_exit_model='full'
)

# ============================================================
# TME cell
# ============================================================

cell = env.new_line(
    components=[

        env.new('QF1','QF'),

        env.new('D1','DR'),

        env.new('B1','B'),

        env.new('MIDB',xt.Marker),

        env.new('D2','DR'),

        env.new('QD1','QD'),

        env.new('D3','DR'),

        env.new('B2','B'),

        env.new('D4','DR'),

        env.new('QF2','QF'),

    ]
)

# ------------------------------------------------------------
# Place marker at center of first dipole
# ------------------------------------------------------------

cell.insert(
    env.new('B1_CENTER', xt.Marker),
    at=Lb/2,
    from_='B1'
)

# ============================================================
# Theoretical minimum emittance conditions
# ============================================================

beta_tme = Lb/(2*np.sqrt(15))
disp_tme = Lb*theta/24

print('Target beta_x =', beta_tme)
print('Target Dx     =', disp_tme)

# ============================================================
# Match cell
# ============================================================
'''
opt = cell.match(
    method='4d',
    solve=False,

    vary=[
        xt.Vary('kQF', step=1e-4),
        xt.Vary('kQD', step=1e-4),
    ],

    targets=[

        xt.TargetSet(
            betx=beta_tme,
            alfx=0.0,
            dx=disp_tme,
            dpx=0.0,
            at='B1_CENTER',
            tol=1e-8
        ),

        xt.TargetSet(
            alfx=0,
            alfy=0,
            at=xt.END,
            tol=1e-8
        )
    ]
)

opt.solve()'''

# ============================================================
# Cell optics
# ============================================================

tw_cell = cell.twiss(method='4d')

print()
print('Cell phase advance')
print('mux =', tw_cell.mux[-1])
print('muy =', tw_cell.muy[-1])

# ============================================================
# Build ring
# ============================================================

ring = env.new_line()

for ii in range(Ncells):
    ring += cell.copy(name=f'CELL_{ii}')

env.lines['ring'] = ring

# ============================================================
# Radiation
# ============================================================

ring.configure_radiation(model='mean')

# ============================================================
# Ring twiss
# ============================================================

tw = ring.twiss(
    method='4d',
    eneloss_and_damping=True
)

print()
print('================================================')
print('Tunes')
print('================================================')
print('Qx =', tw.qx)
print('Qy =', tw.qy)

print()
print('================================================')
print('Natural emittance')
print('================================================')
print('emit_x =', tw.eq_gemitt_x, ' m.rad')

print()
print('================================================')
print('Damping times')
print('================================================')
print('tau_x =', tw.damping_constants_s[0], ' s')
print('tau_y =', tw.damping_constants_s[1], ' s')
print('tau_z =', tw.damping_constants_s[2], ' s')

# ============================================================
# Plot
# ============================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

plt.plot(tw.s, tw.betx, label='betx')
plt.plot(tw.s, tw.bety, label='bety')

plt.xlabel('s [m]')
plt.ylabel('beta [m]')
plt.legend()

plt.tight_layout()
plt.show()
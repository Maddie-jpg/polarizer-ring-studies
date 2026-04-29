#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Created on Mon Apr 27 15:14:35 2026



"""



import numpy              as np

import matplotlib.pyplot  as plt



# %% Read coordinates and some statistics to get emittances and Twiss param's

Eref = 2860.

#f = open('/Users/ccarli/Documents/FCC/PolarizerRing/FromPosLinac/Beam_3GHzOption_2.86GeV_20260421.dat', 'r')

f = open('/home/mwatson/Documents/laughing-octo-bassoon/Macro-particle distributions/PositronBeam_2p86GeV/Beam_3GHzOption_2.86GeV_20260421.dat', 'r')

lines = f.readlines()

f.close()

# long position in ns and momentum offset in 1.e-3

coords = [ (lambda x: [float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4])/(299.7925), 1000*(float(x[5])/Eref - 1)])(line.split(' ')) 

           for line in lines[1:] ]

nptsall = len(coords)



xav, xpav, xx, xxp, xpxp = 0., 0., 0., 0., 0.

yav, ypav, yy, yyp, ypyp = 0., 0., 0., 0., 0.

tav, plav                = 0., 0.

for part in coords:

    xav += part[0]; xpav += part[1]; yav += part[2]; ypav += part[3]

    xx += part[0]**2; xxp += part[0]*part[1]; xpxp += part[1]**2

    yy += part[2]**2; yyp += part[2]*part[3]; ypyp += part[3]**2

    tav += part[4]; plav += part[5]

    

xav, xpav, xx, xxp, xpxp = xav/nptsall, xpav/nptsall, xx/nptsall, xxp/nptsall, xpxp/nptsall

yav, ypav, yy, yyp, ypyp = yav/nptsall, ypav/nptsall, yy/nptsall, yyp/nptsall, ypyp/nptsall

tav, plav = tav/nptsall, plav/nptsall

epsH = ((xx-xav**2)*(xpxp-xpav**2) - (xxp-xav*xpav)**2)**.5

betHL, alfHL = (xx-xav**2)/epsH, -(xxp-xav*xpav)/epsH

epsV = ((yy-yav**2)*(ypyp-ypav**2) - (yyp-yav*ypav)**2)**.5

betVL, alfVL = (yy-yav**2)/epsV, -(yyp-yav*ypav)/epsV



print( f" Horizontal eps ={epsH:7.5f} um, beta ={betHL:7.5f} m and alf ={alfHL:7.5f}" )

print( f"   based on var(x) ={xx-xav**2:8.5f} mm^2, var(x') ={xpxp-xpav**2:8.5f} mrad^2", end="" )

print( f" and cov(x,x') ={xxp-xav*xpav:8.5f} mm mrad" )

print( f" Vertical   eps ={epsV:7.5f} um, beta ={betVL:7.5f} m and alf ={alfVL:7.5f}" )

print( f"   based on var(y) ={yy-yav**2:8.5f} mm^2, var(y') ={ypyp-ypav**2:8.5f} mrad^2", end="" )

print( f" and cov(y,y') ={yyp-yav*ypav:8.5f} mm mrad" )



# %%

def addplots( offs, xdata, ydata, xlims, ylims, fig ):

    

    amain = fig.add_axes( [.5/21 + offs, .5/7, 4/21, 4/7] )

    amain.set_xlim( xlims[0], xlims[1] ); amain.set_ylim( ylims[0], ylims[1] )

    amain.set_xlabel( xlims[2] ); amain.set_ylabel( ylims[2] )

    amain.scatter( xdata, ydata, c='C0', s=.2)

    

    ahor  = fig.add_axes( [.5/21 + offs, 5./7, 4/21, 1/7], sharex=amain)

    ahor.tick_params(axis="x", labelbottom=False)

    ahor.hist( xdata, np.linspace( xlims[0], xlims[1], 100 ) )

    avert = fig.add_axes( [5./21 + offs, .5/7, 1/21, 4/7], sharey=amain)

    avert.tick_params(axis="y", labelleft=False)

    avert.hist( ydata, np.linspace( ylims[0], ylims[1], 100 ), orientation='horizontal' )

    

    return amain

    

fig = plt.figure( figsize = (21, 7) )

ax = addplots( 0./3., [item[0] for item in coords], [item[1] for item in coords],

              [-10., 10., "x [mm]"], [-6., 6., "x' [mrad]"], fig )

ax.plot( [(epsH*betHL)**.5*np.cos(mu) for mu in np.linspace(0, 2*np.pi, 51)], 

         [(epsH/betHL)**.5*(np.sin(mu) - alfHL*np.cos(mu)) for mu in np.linspace(0, 2*np.pi, 51)], c='C1'  )

ay = addplots( 1./3., [item[2] for item in coords], [item[3] for item in coords],

              [-10., 10., "y [mm]"], [-6., 6., "y' [mrad]"], fig)

ay.plot( [(epsV*betVL)**.5*np.cos(mu) for mu in np.linspace(0, 2*np.pi, 51)], 

         [(epsV/betVL)**.5*(np.sin(mu) - alfVL*np.cos(mu)) for mu in np.linspace(0, 2*np.pi, 51)], c='C1'  )

addplots( 2./3., [item[4] - tav for item in coords], [item[5] for item in coords],

         [-.1, .4, "Tau [ns]"], [-200., 100., "delta [1e-3]"], fig)



#fig.savefig('/Users/ccarli/Documents/FCC/PolarizerRing/FromPosLinac/PhSpaces.pdf')

fig.savefig('./PhSpaces.pdf')

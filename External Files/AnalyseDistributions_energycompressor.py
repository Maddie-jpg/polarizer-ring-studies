#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Created on Mon Apr 27 15:14:35 2026



"""



import numpy              as np

import matplotlib.pyplot  as plt

import sys



# %% Read coordinates and some statistics to get emittances and Twiss param's

Eref = 2860.

#f = open('/Users/ccarli/Documents/FCC/PolarizerRing/FromPosLinac/Beam_3GHzOption_2.86GeV_20260421.dat', 'r')

f = open('/Users/maddiewatson/Library/CloudStorage/OneDrive-Personal/University/Research year/polarizer-ring-studies/Macro-particle distributions/PositronBeam_2p86GeV/Beam_3GHzOption_2.86GeV_20260421.dat', 'r')

#f = open('./Beam_3GHzOption_2.86GeV_20260421.dat', 'r')

lines = f.readlines()

f.close()

# all positions in m (as well longitudinal), angles in rad and energy in MeV

# =====>>> sign of longitudinal position to be checked!!!

coords = [ (lambda x: [.001*float(x[0]), .001*float(x[1]), .001*float(x[2]), .001*float(x[3]),

                       .001*float(x[4]), float(x[5])])(line.split(' ')) 

           for line in lines[1:] ]

nptsall = len(coords)



xav, xpav, xx, xxp, xpxp = 0., 0., 0., 0., 0.

yav, ypav, yy, yyp, ypyp = 0., 0., 0., 0., 0.

zav, Enav                = 0., 0.

for part in coords:

    xav += part[0]; xpav += part[1]; yav += part[2]; ypav += part[3]

    xx += part[0]**2; xxp += part[0]*part[1]; xpxp += part[1]**2

    yy += part[2]**2; yyp += part[2]*part[3]; ypyp += part[3]**2

    zav += part[4]; Enav += part[5]

    

xav, xpav, xx, xxp, xpxp = xav/nptsall, xpav/nptsall, xx/nptsall, xxp/nptsall, xpxp/nptsall

yav, ypav, yy, yyp, ypyp = yav/nptsall, ypav/nptsall, yy/nptsall, yyp/nptsall, ypyp/nptsall

zav, Enav = zav/nptsall, Enav/nptsall

epsH = ((xx-xav**2)*(xpxp-xpav**2) - (xxp-xav*xpav)**2)**.5

betHL, alfHL = (xx-xav**2)/epsH, -(xxp-xav*xpav)/epsH

epsV = ((yy-yav**2)*(ypyp-ypav**2) - (yyp-yav*ypav)**2)**.5

betVL, alfVL = (yy-yav**2)/epsV, -(yyp-yav*ypav)/epsV



print( f" Horizontal eps ={1.e6*epsH:7.5f} um, beta ={betHL:7.5f} m and alf ={alfHL:7.5f}" )

print( f"   based on var(x) ={1.e6*(xx-xav**2):8.5f} mm^2, var(x') ={1.e6*(xpxp-xpav**2):8.5f} mrad^2", end="" )

print( f" and cov(x,x') ={1.e6*(xxp-xav*xpav):8.5f} mm mrad" )

print( f" Vertical   eps ={1.e6*epsV:7.5f} um, beta ={betVL:7.5f} m and alf ={alfVL:7.5f}" )

print( f"   based on var(y) ={1.e6*(yy-yav**2):8.5f} mm^2, var(y') ={1.e6*(ypyp-ypav**2):8.5f} mrad^2", end="" )

print( f" and cov(y,y') ={1.e6*(yyp-yav*ypav):8.5f} mm mrad" )



# %% 

def dividedistribution( coords, nparts, nslices, Eref, R56, Vdeb, Phasdeb):

    if nparts*nslices > len(coords):

        sys.exit( '=====>>> nor enough macro-particles in distribution <<<======')

    coordsloc = coords.copy()

    coordslist = []

    for islice in range(nslices):

        coords = []

        for ipart in range (nparts):

            x, xp, y, yp, zeta, En = coordsloc.pop( np.random.randint(0, len(coordsloc)) )

            delt = En/Eref - 1

            zeta = zeta + R56*delt

            delt = delt + Vdeb*np.sin( (zeta*3.000/.2997925)*2*np.pi + Phasdeb ) # for 3GHz RF 

            coords.append( [x, xp, y, yp, zeta, delt] )

        coordslist.append( coords )

    return coordslist



# for test no debunching (R56 = Vdeb = 0) and keep energy (Eref = 1)

# coordslist = dividedistribution( coords, 3007, 8, Eref, 1., .2, np.pi/2)

coordslist = dividedistribution( coords, 3007, 8, 1, 0, 0, np.pi/2)



coordsmod = coords.copy()

coordsmod[17] = coordsmod[3721] # just one particle overwritten by another one

print( f' ===> {len(coords):6d}, {sum([len(item) for item in coordslist]):6d} ', end='')

print( f' and {len(coordslist)*len(coordslist[0]):6d} and {len(coordsmod):6d}' )



print( sum([part[0] for part in coords])/len(coords) )

print( sum([ sum(part[0] for part in item )/len(item) for item in coordslist])/len(coordslist) )

print( sum([part[0] for part in coordsmod])/len(coordsmod) )

print( sum([part[1] for part in coords])/len(coords) )

print( sum([ sum(part[1] for part in item )/len(item) for item in coordslist])/len(coordslist) )

print( sum([part[1] for part in coordsmod])/len(coordsmod) )

print( sum([part[-1] for part in coords])/len(coords) )

print( sum([ sum(part[-1] for part in item )/len(item) for item in coordslist])/len(coordslist) )

print( sum([part[-1] for part in coordsmod])/len(coordsmod) )



# %% define a routine 

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



# %% plot of particle distribution in all three planes without debunching for all part's

coordsdeb = dividedistribution( coords, len(coords), 1, Eref, .0, .0, 0.)[0]



fig = plt.figure( figsize = (21, 7) )

ax = addplots( 0./3., [1000.*item[0] for item in coordsdeb], [1000.*item[1] for item in coordsdeb],

              [-10., 10., "x [mm]"], [-6., 6., "x' [mrad]"], fig )

ax.plot( [(epsH*betHL)**.5*np.cos(mu) for mu in np.linspace(0, 2*np.pi, 51)], 

         [(epsH/betHL)**.5*(np.sin(mu) - alfHL*np.cos(mu)) for mu in np.linspace(0, 2*np.pi, 51)], c='C1'  )

ay = addplots( 1./3., [1000.*item[2] for item in coordsdeb], [1000.*item[3] for item in coordsdeb],

              [-10., 10., "y [mm]"], [-6., 6., "y' [mrad]"], fig)

ay.plot( [(epsV*betVL)**.5*np.cos(mu) for mu in np.linspace(0, 2*np.pi, 51)], 

         [(epsV/betVL)**.5*(np.sin(mu) - alfVL*np.cos(mu)) for mu in np.linspace(0, 2*np.pi, 51)], c='C1'  )

addplots( 2./3., [1000.*(item[4] - zav) for item in coordsdeb], [1000*item[5] for item in coordsdeb],

         [-100., 150., "zeta [mm]"], [-200., 100., "delta [1e-3]"], fig)



#fig.savefig('/Users/ccarli/Documents/FCC/PolarizerRing/FromPosLinac/PhSpaces.pdf')

fig.savefig('./PhSpaces.pdf')



# %% plot of long. phase space initial after bunch lengtening and RF

R56, Vdeb, Phasdeb, Nparts = .32, -.05, .27, 5000

coordsdeb1 = dividedistribution( coords, Nparts, 1, Eref, 0., .0,    0.)[0]

coordsdeb2 = dividedistribution( coords, Nparts, 1, Eref, R56, 0.,   0.)[0]

coordsdeb3 = dividedistribution( coords, Nparts, 1, Eref, R56, Vdeb, Phasdeb)[0]



fig = plt.figure( figsize = (21, 7) )

addplots( 0./3., [1000.*(item[4] - zav) for item in coordsdeb1], [1000*item[5] for item in coordsdeb1],

         [-100., 150., "zeta [mm]"], [-200., 100., "delta [1e-3]"], fig )

a2 = addplots( 1./3., [1000.*(item[4] - zav) for item in coordsdeb2], [1000*item[5] for item in coordsdeb2],

         [-100., 150., "zeta [mm]"], [-200., 100., "delta [1e-3]"], fig)

a2.plot( [zeta for zeta in range(-100, 150)], 

        [1000.*Vdeb*np.sin((zeta*3.000/.2997925)*2*np.pi + Phasdeb) for zeta in range(-100, 150)], c='C1' )

a3 = addplots( 2./3., [1000.*(item[4] - zav) for item in coordsdeb3], [1000*item[5] for item in coordsdeb3],

         [-100., 150., "zeta [mm]"], [-200., 100., "delta [1e-3]"], fig)

a3.plot( [zeta for zeta in range(-100, 150)], 

        [1000.*Vdeb*np.sin((zeta*3.000/.2997925)*2*np.pi + Phasdeb) for zeta in range(-100, 150)], c='C1' )
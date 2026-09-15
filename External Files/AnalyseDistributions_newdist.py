#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis of positron macro-particle distributions send by Iryna:
 - first file sent in spring 2026 contains macro-particles at the end of the 
   positron linac without energy compression

"""

import numpy              as np
import scipy              as sp
import matplotlib.pyplot  as plt
import matplotlib.colors  as mcolors
from   functools          import partial
import sys

Eref = 2860.

# %% Finaly quite some routines!!

# import data from file: each list element describes one particle [x(mm), x'(mrad), , ,z(mm),delta(1.e-3)]
def getdata( filenam ):
    fct = 1.
    f = open(filenam, 'r')
    lines = f.readlines()
    f.close()
    coords = [ (lambda x: [fct*float(x[0]), fct*float(x[1]), fct*float(x[2]), fct*float(x[3]),
                           fct*float(x[4]), 1000.*(float(x[5])/Eref - 1)])(line.split(' ')) 
               for line in lines[1:] ]
    return coords

def partssep( coords, deltmin, deltmax ):
    coordsbulk, coordstail = [], []
    for part in coords:
        if deltmin < part[-1] < deltmax:
            coordsbulk.append( part )
        else:
            coordstail.append( part )
    return coordsbulk, coordstail


# estimating average and variance for a set of one-dimensional variable
#   - using first (sufficiently fine) binning followed by Gaussian fitting
def getrms( xis, maxnmax=50, label='' ):
#   xiav  = sum(xis)/len(xis)   # first guess for average position and variance
#   xivar = sum([(x-xiav)**2 for x in xis])/len(xis)
    minxi, maxxi = min( xis ) - .001*(max(xis)-min( xis )), max(xis) + .001*(max(xis)-min( xis ))  # start to look into binning
    nbins = 2**(int(np.floor(np.log2(len(xis)/maxnmax)))) # any advantage with nbin being power of 2??
    nmax = 2*maxnmax

    while nmax > maxnmax:
        nbins *= 2
        dats = np.histogram( xis, nbins, range=[minxi, maxxi] )
        nmax = max( dats[0] )
    
    xishist = [(xi1 + xi2)/2 for xi1, xi2 in zip(dats[1][:-1], dats[1][1:]) ]
    def func(x, amp, xav, xvar):
       return amp*np.exp(-(x - xav)**2/(2*xvar) )    
    popt, pcov = sp.optimize.curve_fit( func, xishist, dats[0] )
    return popt[1], popt[2]   # returns estimate for average and variance


# Emittance and twiss parameters based on correlations and moments only
# Worked initially, but failed with distribution after energy compressor due to few
#    particles far away from the bulk of the beam contributing significantly to moments
def analysedata( coords ):  
    nptsall = len(coords)
    deltav  = sum([delta for _, _, _, _, _, delta in coords])/nptsall
    zav     = sum([z  for _, _, _, _, z, _ in coords])/nptsall
    xav     = sum([x  for x, _, _, _, _, _ in coords])/nptsall
    xpav    = sum([xp for _, xp, _, _, _, _ in coords])/nptsall
    yav     = sum([y  for _, _, y, _, _, _ in coords])/nptsall
    ypav    = sum([yp for _, _, _, yp, _, _ in coords])/nptsall
    varE = sum([(En - Eav)**2 for _, _, _, _, _, En in coords])/nptsall
    Dtx  = sum([(En - Eav)*(x - xav) for x, _, _, _, _, En in coords])/(nptsall*varE)
    Dptx = sum([(En - Eav)*(xp - xpav) for _, xp, _, _, _, En in coords])/(nptsall*varE)
    xxbet   = sum((x - xav - Dtx*(En - Eav))**2 for x, xp, _, _, _, En in coords)/nptsall
    xpxpbet  = sum((xp - xpav - Dptx*(En - Eav))**2 for x, xp, _, _, _, En in coords)/nptsall
    xxpbet = sum((x - xav - Dtx*(En - Eav))*(xp - xpav - Dptx*(En - Eav)) for x, xp, _, _, _, En in coords)/nptsall
    epsx = (xxbet*xpxpbet - xxpbet**2)**.5
    Dty  = sum([(En - Eav)*(y - yav) for _, _, y, _, _, En in coords])/(nptsall*varE)
    Dpty = sum([(En - Eav)*(yp - ypav) for _, _, _, yp, _, En in coords])/(nptsall*varE)
    yybet   = sum((y - yav - Dty*(En - Eav))**2 for _, _, y, yp, _, En in coords)/nptsall
    ypypbet  = sum((yp - ypav - Dpty*(En - Eav))**2 for _, _, y, yp, _, En in coords)/nptsall
    yypbet = sum((y - yav - Dty*(En - Eav))*(yp - ypav - Dpty*(En - Eav)) for _, _, y, yp, _, En in coords)/nptsall
    epsy = (yybet*ypypbet - yypbet**2)**.5
#   print( f' Average energy {Eav:8.2f} MeV and variance {varE:8.2f} MeV^2' )
#   print( f' Geometric rms emittances {1.e6*epsx:7.4f} um and {1.e6*epsy:7.4f} um' )
    return epsx, xxbet/epsx, -xxpbet/epsx, xav, xpav, Dtx, Dptx, epsy, yybet/epsy, -yypbet/epsy, yav, ypav, Dty, Dpty, Eav, varE, zav 

# Emittance and twiss parameters based on correlations and moments only
# Worked initially, but failed with distribution after energy compressor due to few
#    particles far away from the bulk of the beam contributing significantly to moments
def analysedataProj( coords, iters=3, maxnmax=50 ):  
    nptsall = len(coords)   # start with estimating dispersions etc!!
    zav     = sum([z  for _, _, _, _, z, _  in coords])/nptsall # needed for plotting
    deltav  = sum([delt for _, _, _, _, _, delt in coords])/nptsall
    vardelt = sum([(delt - deltav)**2 for _, _, _, _, _, delt in coords])/nptsall
    xav     = sum([x  for x, _, _, _, _, _ in coords])/nptsall
    xpav    = sum([xp for _, xp, _, _, _, _ in coords])/nptsall
    Dtx     = sum([(delt - deltav)*(x - xav) for x, _, _, _, _, delt in coords])/(nptsall*vardelt)
    Dptx    = sum([(delt - deltav)*(xp - xpav) for _, xp, _, _, _, delt in coords])/(nptsall*vardelt)
    xxbet   = sum((x - xav - Dtx*(delt - deltav))**2 for x, xp, _, _, _, delt in coords)/nptsall
    xpxpbet = sum((xp - xpav - Dptx*(delt - deltav))**2 for x, xp, _, _, _, delt in coords)/nptsall
    xxpbet  = sum((x - xav - Dtx*(delt - deltav))*(xp - xpav - Dptx*(delt - deltav)) for x, xp, _, _, _, delt in coords)/nptsall
    epsx = (xxbet*xpxpbet - xxpbet**2)**.5
    yav     = sum([y  for _, _, y, _, _, _ in coords])/nptsall
    ypav    = sum([yp for _, _, _, yp, _, _ in coords])/nptsall
    Dty     = sum([(delt - deltav)*(y - yav) for _, _, y, _, _, delt in coords])/(nptsall*vardelt)
    Dpty    = sum([(delt - deltav)*(yp - ypav) for _, _, _, yp, _, delt in coords])/(nptsall*vardelt)
    yybet   = sum((y - yav - Dty*(delt - deltav))**2 for _, _, y, yp, _, delt in coords)/nptsall
    ypypbet  = sum((yp - ypav - Dpty*(delt - deltav))**2 for _, _, y, yp, _, delt in coords)/nptsall
    yypbet = sum((y - yav - Dty*(delt - deltav))*(yp - ypav - Dpty*(delt - deltav)) for _, _, y, yp, _, delt in coords)/nptsall
    epsy = (yybet*ypypbet - yypbet**2)**.5

#   horizontal 2 dimensional betatron phase space 
    coords2 = [ [x - Dtx*(delt-deltav), xp - Dptx*(delt-deltav)] for x, xp, _, _, _, delt in coords]
    betxref, alfxref = xxbet/epsx, -xxpbet/epsx  # very first
    for ind in range(iters):
        coordsnor = [[x/betxref**.5, betxref**.5*xp + alfxref*x/betxref**.5] for x, xp in coords2]
        xiav,  xivar  = getrms( [xi for xi, _ in coordsnor], maxnmax, 'Horizontal xi for ind = ' + str(ind) )
        xi1av, xi1var = getrms( [(-xi + 3.**.5*xip)/2 for xi, xip in coordsnor], maxnmax, f'Horizontal xi1 for ind={ind:3d}' )
        xi2av, xi2var = getrms( [(+xi + 3.**.5*xip)/2 for xi, xip in coordsnor], maxnmax, f'Horizontal xi2 for ind={ind:3d}' )
        print( f' iter ={ind:3d} for horizontal reconstruction:', end='' )
        print( f' xivar ={xivar:7.4f}, xi1var ={xi1var:7.4f}, xi2var ={xi2var:7.4f}' )
        
        betnoreps = xivar; alfnoreps = (xi1var - xi2var)/3.**.5; gamnoreps = (2*(xi1var + xi2var) - xivar)/3.
        epsx  = (betnoreps*gamnoreps - alfnoreps*alfnoreps)**.5
        betn, alfn = betnoreps/epsx, alfnoreps/epsx
        xav   = xiav*betxref**.5
        xpav  = ((xi1av + xi2av)/3.**.5 - alfxref*xiav)/betxref**.5
#       alfxref = betn*alfxref + betxref*alfn
        alfxref = betn*alfxref + alfn
        betxref = betn*betxref
        
#   vertical 2 dimensional betatron phase space
    coords2 = [ [y - Dty*(delt-deltav), yp - Dpty*(delt-deltav)] for _, _, y, yp, _, delt in coords]
    betyref, alfyref = yybet/epsy, -yypbet/epsy  # very first
    for ind in range(iters):
        coordsnor = [[y/betyref**.5, betyref**.5*yp + alfyref*y/betyref**.5] for y, yp in coords2]
        xiav,  xivar  = getrms( [xi for xi, _ in coordsnor], maxnmax )
        xi1av, xi1var = getrms( [(-xi + 3.**.5*xip)/2 for xi, xip in coordsnor], maxnmax )
        xi2av, xi2var = getrms( [(+xi + 3.**.5*xip)/2 for xi, xip in coordsnor], maxnmax )
        print( f' iter ={ind:3d} for vertical reconstruction:', end='' )        
        print( f' xivar ={xivar:7.4f}, xi1var ={xi1var:7.4f}, xi2var ={xi2var:7.4f}' )
        
        betnoreps = xivar; alfnoreps = (xi1var - xi2var)/3.**.5; gamnoreps = (2*(xi1var + xi2var) - xivar)/3.
        epsy  = (betnoreps*gamnoreps - alfnoreps*alfnoreps)**.5
        betn, alfn = betnoreps/epsy, alfnoreps/epsy
        yav   = xiav*betyref**.5
        ypav  = ((xi1av + xi2av)/3.**.5 - alfyref*xiav)/betyref**.5
        alfyref = betn*alfyref + alfn
        betyref = betn*betyref

    return epsx, betxref, alfxref, xav, xpav, Dtx, Dptx, epsy, betyref, alfyref, yav, ypav, Dty, Dpty, deltav, vardelt, zav


#def plotdistribution(coords):
def densityplots( poss, xbulk, ybulk, xtail, ytail, xlims, ylims, fig=None ):
#   posdata contains 
    if fig==None: fig = plt.figure(  )
    figsx, figsy = fig.get_size_inches()
    amain = fig.add_axes( [poss[0][0]/figsx, poss[1][0]/figsy, (poss[0][1]-poss[0][0])/figsx, (poss[1][1]-poss[1][0])/figsy] )
    amain.set_xlim( xlims[0], xlims[1] ); amain.set_ylim( ylims[0], ylims[1] )
    amain.set_xlabel( xlims[2] ); amain.set_ylabel( ylims[2] )
    
    def my_kde_bandwidth(obj, fac=1./5):   # works, but would be nice to understand concepts!!!
        """We use Scott's Rule, multiplied by a constant factor."""
        return np.power(obj.n, -1./(obj.d+4)) * fac
#   z = gaussian_kde( [xdata, ydata], bw_method=partial(my_kde_bandwidth, fac=0.02) )
    z = sp.stats.gaussian_kde( [xbulk, ybulk] )
    zev = z.evaluate( [xbulk, ybulk] )
    amain.scatter( xbulk, ybulk, c=zev, s=.3, edgecolors='none')
    amain.scatter( xtail, ytail, c='orange', s=.3, edgecolors='none' )
#   amain.hist2d(xdata, ydata, bins=100, norm=mcolors.PowerNorm(0.8)) 
#   amain.scatter( xdata, ydata, c='C0', s=.1, edgecolors='none' ) 
  
    ahor  = fig.add_axes( [poss[0][0]/figsx, poss[1][2]/figsy, (poss[0][1]-poss[0][0])/figsx, (poss[1][3]-poss[1][2])/figsy], sharex=amain)
    ahor.tick_params(axis="x", labelbottom=False)
    ahor.hist( [xbulk, xtail], np.linspace( xlims[0], xlims[1], 100 ), histtype='bar', stacked=True, color=['C0', 'orange'] )
    avert = fig.add_axes( [poss[0][2]/figsx, poss[1][0]/figsy, (poss[0][3]-poss[0][2])/figsx, (poss[1][1]-poss[1][0])/figsy], sharey=amain)
    avert.tick_params(axis="y", labelleft=False)
    avert.hist([ybulk, ytail], np.linspace( ylims[0], ylims[1], 100 ), orientation='horizontal', histtype='bar', stacked=True, color=['C0', 'orange']  )
    
    return amain, ahor, avert

# %% Analysis carried for four different cases (removing longitudinal tails or not)
#    and fitting of projections or not) always using data from spring and recent ones 
#    energy compression
#    Scatter plots without removing (very small) dispersive contribution
# ==> get some text output and plots with most relevant infos

for case in [['Analysis based only on moments of distribution and all macro-particles',
         -1000., 1000., -1000., 1000., 0, 50, 'MomsAll'],
        ['Analysis based only on moments of part of the distribution',
         -30., 30., -64., 80., 0, 50, 'MomsBulk'],
        ['Analysis based only on fits to projections and all macro-particles',
         -1000., 1000., -1000., 1000., 5, 50, 'FitsAll'],
        ['Analysis based on fits to projections for bulk of macro-particles',
         -30., 30., -64., 80., 5, 50, 'FitsBulk']]:

   fig = plt.figure( figsize=(19.3, 13.5) )
   fig.suptitle( case[0], fontsize = 20 )

   cosbulk, costail = partssep( getdata('Documents/FCC/PolarizerRing/FromPosLinac/beam_ECS_04092026.dat'), case[1], case[2] )
   epsx, betx, alfx, xav, xpav, Dtx, Dptx, epsy, bety, alfy, yav, ypav, Dty, Dpty, deltav, vardelt, zav = analysedataProj( cosbulk, iters=case[5], maxnmax=case[6] )
   print( f'\n\n==> With energy compression, fraction of particles within bulk:\n  len(cosbulk)/(len(cosbulk)+len(costail)) ={len(cosbulk):6d}/({len(cosbulk):6d} +{len(costail):6d}) ={len(cosbulk)/(len(cosbulk)+len(costail)):7.4f}' )
   print( f'  average rel. momentum offset {deltav:8.2f}e-3 and variance {vardelt:8.2f}e-6' )
   print( f'  geometric rms emittances {epsx:7.4f} um and {epsy:7.4f} um' )
   print( f'  ( Dx^2/betx + (Dpx*alfx/betx^.5 + betx^.5*Dpx)^2) vardelt ={(Dtx**2/betx + (Dptx*alfx/betx**.5 + betx**.5*Dptx)**2)*vardelt:7.4f} um and', end='')
   print( f'  ( Dy^2/bety + (Dpy*alfy/bety^.5 + bety^.5*Dpy)^2) vardelt ={(Dty**2/bety + (Dpty*alfy/bety**.5 + bety**.5*Dpty)**2)*vardelt:7.4f} um' )
   ax, _, _ = densityplots( [[ 0.5,  4.5,  4.9,  5.9], [ .5,  4.5,  4.9,  5.9]], [el[0] for el in cosbulk], [el[1] for el in cosbulk], [el[0] for el in costail], [el[1] for el in costail], [-15, 15, 'x (mm)'], [-4, 4, "x' (mrad)"], fig )
   ax.plot( [(epsx*betx)**.5*np.cos(mu) for mu in np.arange(0., 2.*np.pi+.0001, .05*np.pi)], [(epsx/betx)**.5*(np.sin(mu) - alfx*np.cos(mu)) for mu in np.arange(0., 2.*np.pi+.0001, .05*np.pi)] )
   ax.text( .1, .9, r'$\varepsilon_x$ =' + f'{epsx:7.3f} ' + r'$\mu$m    $\beta_x$ =' + f'{betx:7.3f} m    ' + r'$\alpha_x$ =' + f'{alfx:7.3f}', transform=ax.transAxes)
   ax, _, _ = densityplots( [[ 7.2, 11.2, 11.6, 12.6], [ .5,  4.5,  4.9,  5.9]], [el[2] for el in cosbulk], [el[3] for el in cosbulk], [el[2] for el in costail], [el[3] for el in costail], [-15, 15, 'y (mm)'], [-4, 4, "y' (mrad)"], fig )
   ax.text( .1, .9, r'$\varepsilon_y$ =' + f'{epsy:7.3f} ' + r'$\mu$m    $\beta_y$ =' + f'{bety:7.3f} m    ' + r'$\alpha_y$ =' + f'{alfy:7.3f}', transform=ax.transAxes)
   ax.plot( [(epsy*bety)**.5*np.cos(mu) for mu in np.arange(0., 2.*np.pi+.0001, .05*np.pi)], [(epsy/bety)**.5*(np.sin(mu) - alfy*np.cos(mu)) for mu in np.arange(0., 2.*np.pi+.0001, .05*np.pi)] )
   densityplots( [[13.9, 17.9, 18.3, 19.3], [ .5,  4.5,  4.9,  5.9]], [(el[4]-zav) for el in cosbulk], [el[5] for el in cosbulk], [(el[4]-zav) for el in costail], [el[5] for el in costail], [-100., 150., "zeta [mm]"], [-200., 100., "delta [1e-3]"], fig )

#  densityplots( [[13.9, 17.9, 18.3, 19.3], [ .5,  4.5,  4.9,  5.9]], [(el[4]-zav) for el in cosbulk], [el[5] for el in cosbulk], [(el[4]-zav) for el in costail], [el[5] for el in costail], [-100., 250., "zeta [mm]"], [-350., 100., "delta [1e-3]"] )
#  fig, ax = plt.subplots( figsize=(6., 6.) )
#  ax.scatter( [el[4]-zav for el in cosbulk+costail ], [el[5] for el in cosbulk+costail ], s=.3, edgecolors = 'none')

   cosbulk, costail = partssep( getdata('Documents/FCC/PolarizerRing/FromPosLinac/Beam_3GHzOption_2.86GeV_20260421.dat'), case[3], case[4] )
   epsx, betx, alfx, xav, xpav, Dtx, Dptx, epsy, bety, alfy, yav, ypav, Dty, Dpty, deltav, vardelt, zav = analysedataProj( cosbulk, iters=case[5], maxnmax=case[6] )
   print( f'\n==> Without energy compression, fraction of particles within bulk\n  len(cosbulk)/(len(cosbulk)+len(costail) ={len(cosbulk):6d}/({len(cosbulk):6d}+{len(costail):6d}) ={len(cosbulk)/(len(cosbulk)+len(costail)):7.4f}' )
   print( f'  average rel. momentum offset {deltav:8.2f}e-3 and variance {vardelt:8.2f}e-6' )
   print( f'  geometric rms emittances {epsx:7.4f} um and {epsy:7.4f} um' )
   ax, _, _ = densityplots( [[ 0.5,  4.5,  4.9,  5.9], [ 7.2, 11.2, 11.6, 12.6]], [el[0] for el in cosbulk], [el[1] for el in cosbulk], [el[0] for el in costail], [el[1] for el in costail], [-15, 15, 'x (mm)'], [-4, 4, "x' (mrad)"], fig )
   ax.plot( [(epsx*betx)**.5*np.cos(mu) for mu in np.arange(0., 2.*np.pi+.0001, .05*np.pi)], [(epsx/betx)**.5*(np.sin(mu) - alfx*np.cos(mu)) for mu in np.arange(0., 2.*np.pi+.0001, .05*np.pi)] )
   ax.text( .1, .9, r'$\varepsilon_x$ =' + f'{epsx:7.3f} ' + r'$\mu$m    $\beta_x$ =' + f'{betx:7.3f} m    ' + r'$\alpha_x$ =' + f'{alfx:7.3f}', transform=ax.transAxes)
   ax, _, _ = densityplots( [[ 7.2, 11.2, 11.6, 12.6], [ 7.2, 11.2, 11.6, 12.6]], [el[2] for el in cosbulk], [el[3] for el in cosbulk], [el[2] for el in costail], [el[3] for el in costail], [-15, 15, 'y (mm)'], [-4, 4, "y' (mrad)"], fig )
   ax.plot( [(epsy*bety)**.5*np.cos(mu) for mu in np.arange(0., 2.*np.pi+.0001, .05*np.pi)], [(epsy/bety)**.5*(np.sin(mu) - alfy*np.cos(mu)) for mu in np.arange(0., 2.*np.pi+.0001, .05*np.pi)] )
   ax.text( .1, .9, r'$\varepsilon_y$ =' + f'{epsy:7.3f} ' + r'$\mu$m    $\beta_y$ =' + f'{bety:7.3f} m    ' + r'$\alpha_y$ =' + f'{alfy:7.3f}', transform=ax.transAxes)
   densityplots( [[13.9, 17.9, 18.3, 19.3], [ 7.2, 11.2, 11.6, 12.6]], [(el[4]-zav) for el in cosbulk], [el[5] for el in cosbulk], [(el[4]-zav) for el in costail], [el[5] for el in costail], [-100., 150., "zeta [mm]"], [-200., 100., "delta [1e-3]"], fig )

   fig.savefig( 'Documents/FCC/PolarizerRing/FromPosLinac/' + case[7] + '.png' )
# %%

import xtrack as xt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import os
#Chromaticity correction

def ChromCorrect(ring, ksf, ksd, MakePlot=False):
        
    opt_chrom = ring.match(
        solve=False,
        method='4d',
        vary=xt.VaryList([ksf, ksd], step=1e-3),
        targets=xt.TargetSet(dqx=0, dqy=0, tol=1e-3))
    opt_chrom.target_status()
    opt_chrom.run_jacobian(n_steps=100)
    opt_chrom.target_status()
    # Print the final matched strengths

    print(f"Matched kSF: {pdr.vars['kSF']._get_value():.6f}")
    print(f"Matched kSD: {pdr.vars['kSD']._get_value():.6f}")   


#Suitability of working point

def plot_working_point(qx, qy, max_order=4):
    fig, ax = plt.subplots(figsize=(8, 8))
    drawn_lines = set()

    for order in range(1, max_order + 1):
        linewidth = 2.0 / order
        alpha = 1.0 / order
        
        for m in range(order + 1):
            n = order - m
            for p in range(-max_order, max_order * 2):
                
                common = math.gcd(m, n, p)
                if common == 0: continue 
                
                line_id = (m // common, n // common, p // common)
                
                if line_id not in drawn_lines:
                    if n != 0:
                        x = np.array([0, 1])
                        y = (p - m * x) / n
                        if np.any((y >= 0) & (y <= 1)) or np.any((y <= 0) & (y >= 1)):
                            ax.plot(x, y, 'k-', lw=linewidth, alpha=alpha)
                    elif m != 0:
                        x_pos = p / m
                        if 0 <= x_pos <= 1:
                            ax.axvline(x_pos, color='k', lw=linewidth, alpha=alpha)
                    
                    drawn_lines.add(line_id)

    # Plot the specific working point
    ax.plot(qx % 1, qy % 1, 'ro', ms=10, label=f'WP ({qx:.4f}, {qy:.4f})')
    
    # Formatting
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f'Tune Diagram - Order 1 to {max_order}')
    ax.set_xlabel('$Q_x$ fraction')
    ax.set_ylabel('$Q_y$ fraction')
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    ax.legend()
    
    plt.show()

def addSketchBL( acc_tw, acc, lims, limy1, limy2, scK1 ):
    fig, ax = plt.subplots( 3, 1, figsize=(12, 8), height_ratios=(1.8, 4, 3) )
    axl, axp, axt = ax
    axp.sharex( axl )
    fig.subplots_adjust( hspace=0.0, top=1.0, bottom=0.0, left=.1 )

    axp.set_xlabel( 'Position s [m]' )
    axp.set_ylim( limy1 )
    axp.set_ylabel( r'$\beta_x$, $\beta_y$ [m]' )
      
    indm = int( np.array([(np.sign(lims[-1]-x) + 1)/2 for x in acc_tw.s ]).sum() )
    axp.plot( acc_tw.s[:indm], acc_tw.betx[:indm], color='black', linestyle=(0, (10, 0)) )
    axp.plot( acc_tw.s[:indm], acc_tw.bety[:indm], color='black', linestyle=(0, (6, 4)) )
    axp2 = axp.twinx()
    axp2.set_ylim( limy2 )
    axp2.set_ylabel( r'$D_x$, $D_y$ [m]' )
    axp2.plot( acc_tw.s[:indm], acc_tw.dx[:indm], color='black', linestyle=(0, (5, 3, 1, 3)) )
    axp2.plot( acc_tw.s[:indm], acc_tw.dy[:indm] )
    axp.set_xlim( lims )
    
    axl.set_ylim( -.30, 1.00 )
    axl.axis( 'off' )
    tab_pan = acc.get_table(attr=True).to_pandas()
    for ind in range( len(tab_pan.T.columns) -1 ):
        if tab_pan['element_type'][ind].find('Drift') >= 0:
           axl.plot( [tab_pan['s'][ind], tab_pan['s'][ind+1]], [0, 0], color='black' )
        if tab_pan['element_type'][ind].find('Bend') >= 0:
           axl.add_patch( patches.Rectangle( (tab_pan['s'][ind], -0.08), 
               tab_pan['s'][ind+1] - tab_pan['s'][ind], 0.16, fill=True, color='tab:blue' ) ) 
        if tab_pan['element_type'][ind].find('Quad') >= 0:
           kstr = scK1*tab_pan['k1l'][ind]/tab_pan['length'][ind]
           axl.add_patch( patches.Rectangle( (tab_pan['s'][ind], min(0.0, kstr)),
               tab_pan['s'][ind+1] - tab_pan['s'][ind], abs(kstr), fill=True, color='tab:orange' ) )    
        if tab_pan['element_type'][ind].find('Sext')>=0:
           kstr = scK1*tab_pan['k2l'][ind]/tab_pan['length'][ind]
           axl.add_patch( patches.Rectangle( (tab_pan['s'][ind], min(0.0, kstr)),
               tab_pan['s'][ind+1] - tab_pan['s'][ind], abs(kstr), fill=True, color='tab:green' ) )      
    axp.plot( [acc.get_length(), acc.get_length()], limy1, color='black', 
              linestyle=(0, (8, 8)), linewidth=.5 )

    axt.axis( 'off' )
    axt.set_xlim( -.3, 3.8 ); axt.set_ylim( 0, 1 )
    return axt

# Routine generating graphical and text output describing the ring    
def SpuckParsAus( acc_tw, acc, lims, limy1, limy2, scK1, grname='NoGraph' ):
    axt = addSketchBL( acc_tw, acc, lims, limy1, limy2, scK1 )
    axt.text( 0.0, .7, f'C ={3*acc_tw.circumference:8.4f} m', horizontalalignment='left' )
    axt.text( 1.0, .7, f'T ={3e6*acc_tw.T_rev0:8.4f} us', horizontalalignment='left')
    axt.text( 2.0, .7, r'($Q_x$, $Q_y$)' + f' = ({3*acc_tw.qx:9.5f}, {3*acc_tw.qy:9.5f})', 
        horizontalalignment='left')
    pos=0
    for item in pdr.vars.keys()[2:]:
       print( "'" + item + f"': {pdr[item]:8.4f}," )
       axt.text( pos%4 , .6 - .1*int( pos/4 ),
          "'" + item + f"': {pdr[item]:8.4f},", horizontalalignment='left' )
       pos += 1
    current_dir = os.getcwd()

    if grname != 'NoGraph': 
        # Check for the file in the current directory
        if grname in os.listdir(current_dir):
            print(f' Error: file {grname} exists <<<<<<<<<<<================' )
        else:
            # Use os.path.join to handle slashes correctly
            save_path = os.path.join(current_dir, grname)
            plt.savefig(save_path)
            print(f'Successfully saved to: {save_path}')
    tab_pan = acc.get_table(attr=True).to_pandas()
    print('  Name        Type      L(m)     sin(m)   sout(m)  driftl  driftr    k1(1/m^2)')
    for ind in range( len(tab_pan.T.columns) - 1 ):
       eltyp = tab_pan['element_type'][ind]
       if eltyp.find('Drift') < 0:
          elnam = tab_pan['name'][ind]
          if ind < 1: sin = '  0.0000'
          else: sin = f"{tab_pan['length'][ind-1]:8.4f}"         
          k1expr = pdr.element_refs[elnam].k1._expr
          if k1expr == None: k1expr = ' None'
          else: k1expr = str(k1expr)[5:-1]
          print(  "  " + elnam.ljust( 12 ) + eltyp.ljust(12)[:8] + f" {tab_pan['length'][ind]:7.4f} " + 
               f"{tab_pan['s'][ind]:8.4f} {tab_pan['s'][ind+1]:8.4f} " + sin +
               f"{tab_pan['length'][ind+1]:8.4f} " + k1expr.ljust(9) + 
               f"= {tab_pan['k1l'][ind]/max(tab_pan['length'][ind],1e-6):7.4f}  " )

#Best sextupole positions
def SextupolePosition(ring):
   tw = ring.twiss4d()

   sf_candidates = []
   sd_candidates = []
   
   sf_top = []
   sd_top = []

   for name in tw.name:
         row = tw.rows[name]
         dx = abs(row.dx[0])
         betx = row.betx[0]
         bety = row.bety[0]

         if dx > 0.3:
            if betx > bety:
               sf_candidates.append((name, dx, betx))
            else:
               sd_candidates.append((name, dx, bety))
   
   sf_candidates = sorted(sf_candidates, key=lambda x: x[1]*x[2], reverse=True)
   sd_candidates = sorted(sd_candidates, key=lambda x: x[1]*x[2], reverse=True)

   if sf_candidates:
      sf_top = sf_candidates[:10]
      for c in sf_top:
         print("SF_positions", c)

   if sd_candidates:
      sd_top = sd_candidates[:10]
      for c in sd_top:
         print("SD_positions", c)

   return sf_top, sd_top

#Placing these sextupoles
'''

for name _,_, in sf_candidates[:4]:
    ring.insert(pdr.new(f'SF_auto_{name},'SFarc'), at=name)

for name _,_, in sd_candidates[:4]:
    ring.insert(pdr.new(f'SD_auto_{name}, 'SDarc'), at=name)

    
'''
def matchingWP( qx, qy,cell_arc_opt,cell_arc,arc1R, MakePlot=False ):
    cell_arc_opt.run_jacobian(10)
    cell_arc_tw = cell_arc.twiss( method='4d' )
    arc1R_opttune = arc1R.match( method='4d', solve=False, verbose=False,
                 betx=cell_arc_tw.betx[0], alfx=cell_arc_tw.alfx[0],
                 bety=cell_arc_tw.bety[0], alfy=cell_arc_tw.alfy[0],
                 dx=cell_arc_tw.dx[0],     dpx=cell_arc_tw.dpx[0],
            vary=[ xt.VaryList(['kQFarcM', 'kQDarcM'], step=1e-4),
                   xt.VaryList(['kQFDS', 'kQDDS'], step=1e-4),
                   xt.VaryList(['kQFDoub', 'kQDDoub'], step=1e-4),
                   xt.VaryList(['kQFtr', 'kQDtr'], step=1e-4), ],
            targets=[ xt.TargetSet(dx=0, dpx=0, at=xt.END, tol=1.0e-9),
                      xt.TargetSet(mux=qx/6, muy=qy/6, at=xt.END, 
                                   tol=1.0e-9, weight=.1, tag='phase'),
                      xt.TargetSet(alfx=0, alfy=0, at=xt.END, tol=1.0e-9),
                      xt.TargetSet(alfx=0, alfy=0, at='CtrS1_xR1', 
                                   tol=1.0e-9, weight=10.) ])
    arc1R_opttune.run_jacobian(50)
    if MakePlot:
       arc1R.twiss( method='4d',
                    betx=cell_arc_tw.betx[0], alfx=cell_arc_tw.alfx[0],
                    bety=cell_arc_tw.bety[0], alfy=cell_arc_tw.alfy[0],
                    dx=cell_arc_tw.dx[0],     dpx=cell_arc_tw.dpx[0],).plot() 
        
# Matching sequence for sextant with beta at center of straights fixed
def matchingBeta( betxS, betyS,cell_arc_opt, cell_tr_opt,cell_arc,cell_tr,arc1R,MakePlot=False ): #, **kwargs ):
    cell_arc_opt.run_jacobian(10)
    cell_arc_tw = cell_arc.twiss( method='4d' )
    cell_tr_opt.targets[0].value, cell_tr_opt.targets[1].value = betxS, betyS
    cell_tr_opt.run_jacobian(10)
    cell_tr_tw = cell_tr.twiss( method='4d' )
    arc1R_opt = arc1R.match( method='4d', solve=True,
                 betx=cell_arc_tw.betx[0], alfx=cell_arc_tw.alfx[0],
                 bety=cell_arc_tw.bety[0], alfy=cell_arc_tw.alfy[0],
                 dx=cell_arc_tw.dx[0],     dpx=cell_arc_tw.dpx[0],
            vary=[ xt.VaryList(['kQFarcM', 'kQDarcM'], step=1e-4),
                   xt.VaryList(['kQFDS', 'kQDDS'], step=1e-4),
                   xt.VaryList(['kQFDoub', 'kQDDoub'], step=1e-4) ],
            targets=[ xt.TargetSet(dx=0, dpx=0, at=xt.END, tol=1.0e-9),
                      xt.TargetSet(alfx=0, alfy=0, at=xt.END, tol=1.0e-9),
                      xt.TargetSet(betx=cell_tr_tw.betx[0], bety=cell_tr_tw.bety[0],
                                   at=xt.END, tol=1.0e-9) ])
    arc1R_opt.run_jacobian(10)
    if MakePlot:
       arc1R.twiss( method='4d',
                    betx=cell_arc_tw.betx[0], alfx=cell_arc_tw.alfx[0],
                    bety=cell_arc_tw.bety[0], alfy=cell_arc_tw.alfy[0],
                    dx=cell_arc_tw.dx[0],     dpx=cell_arc_tw.dpx[0],).plot() 
       

def addSketchBL( acc_tw, acc, lims, limy1, limy2, scK1 ):
    fig, ax = plt.subplots( 3, 1, figsize=(12, 8), height_ratios=(1.8, 4, 3) )
    axl, axp, axt = ax
    axp.sharex( axl )
    fig.subplots_adjust( hspace=0.0, top=1.0, bottom=0.0, left=.1 )

    axp.set_xlabel( 'Position s [m]' )
    axp.set_ylim( limy1 )
    axp.set_ylabel( r'$\beta_x$, $\beta_y$ [m]' )
      
    indm = int( np.array([(np.sign(lims[-1]-x) + 1)/2 for x in acc_tw.s ]).sum() )
    axp.plot( acc_tw.s[:indm], acc_tw.betx[:indm], color='black', linestyle=(0, (10, 0)) )
    axp.plot( acc_tw.s[:indm], acc_tw.bety[:indm], color='black', linestyle=(0, (6, 4)) )
    axp2 = axp.twinx()
    axp2.set_ylim( limy2 )
    axp2.set_ylabel( r'$D_x$, $D_y$ [m]' )
    axp2.plot( acc_tw.s[:indm], acc_tw.dx[:indm], color='black', linestyle=(0, (5, 3, 1, 3)) )
    axp2.plot( acc_tw.s[:indm], acc_tw.dy[:indm] )
    axp.set_xlim( lims )
    
    axl.set_ylim( -.30, 1.50 )
    axl.axis( 'off' )
    tab_pan = acc.get_table(attr=True).to_pandas()
    for ind in range( len(tab_pan.T.columns) -1 ):
        if tab_pan['element_type'][ind].find('Drift') >= 0:
           axl.plot( [tab_pan['s'][ind], tab_pan['s'][ind+1]], [0, 0], color='black' )
        if tab_pan['element_type'][ind].find('Bend') >= 0:
           axl.add_patch( patches.Rectangle( (tab_pan['s'][ind], -0.08), 
               tab_pan['s'][ind+1] - tab_pan['s'][ind], 0.16, fill=True, color='tab:blue' ) ) 
        if tab_pan['element_type'][ind].find('Quad') >= 0:
           kstr = scK1*tab_pan['k1l'][ind]/tab_pan['length'][ind]
           axl.add_patch( patches.Rectangle( (tab_pan['s'][ind], min(0.0, kstr)),
               tab_pan['s'][ind+1] - tab_pan['s'][ind], abs(kstr), fill=True, color='tab:orange' ) )    
        if tab_pan['element_type'][ind].find('Sext')>=0:
           kstr = (scK1 * tab_pan['k2l'][ind] / tab_pan['length'][ind]) * 0.05
    
           axl.add_patch( patches.Rectangle( 
            (tab_pan['s'][ind], min(0.0, kstr)),
            tab_pan['s'][ind+1] - tab_pan['s'][ind], 
            abs(kstr), 
            fill=True, 
            color='tab:green' ) )      
    axp.plot( [acc.get_length(), acc.get_length()], limy1, color='black', 
              linestyle=(0, (8, 8)), linewidth=.5 )

    axt.axis( 'off' )
    axt.set_xlim( -.3, 3.8 ); axt.set_ylim( 0, 1.5 )
    return axt

# Routine generating graphical and text output describing the ring    
def SpuckParsAus( acc_tw, acc, lims, limy1, limy2, scK1, grname='NoGraph' ):
    axt = addSketchBL( acc_tw, acc, lims, limy1, limy2, scK1 )
    axt.text( 0.0, .7, f'C ={3*acc_tw.circumference:8.4f} m', horizontalalignment='left' )
    axt.text( 1.0, .7, f'T ={3e6*acc_tw.T_rev0:8.4f} us', horizontalalignment='left')
    axt.text( 2.0, .7, r'($Q_x$, $Q_y$)' + f' = ({3*acc_tw.qx:9.5f}, {3*acc_tw.qy:9.5f})', 
        horizontalalignment='left')
    pos=0
    for item in pdr.vars.keys()[2:]:
       print( "'" + item + f"': {pdr[item]:8.4f}," )
       axt.text( pos%4 , .6 - .1*int( pos/4 ),
          "'" + item + f"': {pdr[item]:8.4f},", horizontalalignment='left' )
       pos += 1
    file=os.getcwd()
    if grname != 'NoGraph': 
       if grname in file:
          print( ' Error: file ' + grname + ' exists <<<<<<<<<<<================' )
       else:
          plt.savefig( file + grname )
    tab_pan = acc.get_table(attr=True).to_pandas()
    print('  Name        Type      L(m)     sin(m)   sout(m)  driftl  driftr    k1(1/m^2)')
    for ind in range( len(tab_pan.T.columns) - 1 ):
       eltyp = tab_pan['element_type'][ind]
       if eltyp.find('Drift') < 0:
          elnam = tab_pan['name'][ind]
          if ind < 1: sin = '  0.0000'
          else: sin = f"{tab_pan['length'][ind-1]:8.4f}"         
          k1expr = pdr.element_refs[elnam].k1._expr
          if k1expr == None: k1expr = ' None'
          else: k1expr = str(k1expr)[5:-1]
          print(  "  " + elnam.ljust( 12 ) + eltyp.ljust(12)[:8] + f" {tab_pan['length'][ind]:7.4f} " + 
               f"{tab_pan['s'][ind]:8.4f} {tab_pan['s'][ind+1]:8.4f} " + sin +
               f"{tab_pan['length'][ind+1]:8.4f} " + k1expr.ljust(9) + 
               f"= {tab_pan['k1l'][ind]/max(tab_pan['length'][ind],1e-6):7.4f}  " )
          


import xtrack as xt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np

pdr=xt.Environment()

def addSketchBL(acc_tw, acc, lims, limy1, limy2, scK1):
    # Added a 4th row for Chromatic Functions (axw)
    fig, ax = plt.subplots(4, 1, figsize=(12, 12), 
                           height_ratios=(1, 3, 2, 2))
    axl, axp, axw, axt = ax  # Lattice, Optics, W-functions, Text Table
    
    # Link the x-axes
    axp.sharex(axl)
    axw.sharex(axl)
    fig.subplots_adjust(hspace=0.1, top=0.95, bottom=0.05, left=.1)

    #Subplot 2: Beta and Dispersion
    axp.set_ylabel(r'$\beta_x, \beta_y$ [m]')
    indm = np.argmin(np.abs(acc_tw.s - lims[-1])) + 1
    
    axp.plot(acc_tw.s[:indm], acc_tw.betx[:indm], color='red', label=r'$\beta_x$')
    axp.plot(acc_tw.s[:indm], acc_tw.bety[:indm], color='blue', label=r'$\beta_y$')
    axp.set_ylim(limy1)
    
    axp2 = axp.twinx()
    axp2.plot(acc_tw.s[:indm], acc_tw.dx[:indm], color='black', linestyle='--', label='$D_x$')
    axp2.set_ylabel(r'$D_x$ [m]')
    axp2.set_ylim(limy2)
    axp.set_xlim(lims)

    # Chromatic W Functions and second order dispersion
    axw.plot(acc_tw.s[:indm], acc_tw.wx_chrom[:indm], label='$W_x$', color='red')
    axw.plot(acc_tw.s[:indm], acc_tw.wy_chrom[:indm], label='$W_y$', color='blue')
    axw2 = axw.twinx()
    axw2.plot(acc_tw.s[:indm], acc_tw.ddx[:indm], color='black', linestyle='--', label="$D'_x$")
    

    '''tt_sliced = acc.get_table(attr=True)
    tbends = tt_sliced.rows[tt_sliced.element_type == 'Bend']
    
    # Apply shading ONLY to the Chromatic W-functions axis
    for nn in tbends.name:
        axw.axvspan(
            tbends['s', nn], 
            tbends['s', nn] + tbends['length', nn],
            color='tab:blue', 
            alpha=0.15,       # Subtle enough to see the Wx/Wy lines
            linewidth=0
        )'''
    
    axw.set_ylabel('Chromatic $W$')
    axw.set_xlabel('Position s [m]')
    axw.legend(loc='upper right', fontsize=8)
    axw2.legend(loc='upper right', fontsize=8)
    axw2.set_ylabel(r"$D'_x$ [m]")
    
    #Lattice Sketch
    axl.set_ylim(-0.5, 1.5)
    axl.axis('off')
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
        if tab_pan['element_type'][ind].find('Sext') >= 0:
           max_k2l = max([abs(val) for ind, val in enumerate(tab_pan['k2l']) 
               if 'Sext' in tab_pan['element_type'][ind]])
           kstr = scK1*tab_pan['k2l'][ind]/tab_pan['length'][ind]/max_k2l *3
           axl.add_patch( patches.Rectangle( (tab_pan['s'][ind], min(0.0, kstr)) ,
               tab_pan['s'][ind+1] - tab_pan['s'][ind], abs(kstr), fill=True, color='tab:green' ) )   
        if tab_pan['element_type'][ind].find('Multipole') >= 0:
           axl.add_patch( patches.Rectangle( (tab_pan['s'][ind], -0.08), 
               tab_pan['s'][ind+1] - tab_pan['s'][ind], 0.16, fill=True, color='tab:pink' ) )   
    axp.plot( [acc.get_length(), acc.get_length()], limy1, color='black', 
              linestyle=(0, (8, 8)), linewidth=.5 )

    # Text Area
    axt.axis('off')
    axt.set_xlim(-0.3, 4.0)
    axt.set_ylim(0, 1.5)
    
    return axt

# Routine generating graphical and text output describing the ring    
def SpuckParsAus( acc_tw, acc, lims, limy1, limy2, scK1,pdr, grname='NoGraph' ):
    axt = addSketchBL( acc_tw, acc, lims, limy1, limy2, scK1 )
    axt.text( 0.0, .7, f'C ={3*acc_tw.circumference:8.4f} m', horizontalalignment='left' )
    axt.text( 1.0, .7, f'T ={3e6*acc_tw.T_rev0:8.4f} us', horizontalalignment='left')
    axt.text( 2.0, .7, r'($Q_x$, $Q_y$)' + f' = ({3*acc_tw.qx:9.5f}, {3*acc_tw.qy:9.5f})',horizontalalignment='left')

    
    pos=0
    for item in pdr.vars.keys()[2:]:
         val = pdr[item]
         if abs(val) > 1e-12: 
            print("'" + item + f"': {val:8.4f},")
            axt.text(pos%4 , .6 - .1*int(pos/4),
                  "'" + item + f"': {val:8.4f},",
                  horizontalalignment='left')
            pos += 1
    file=os.getcwd()
    if grname != 'NoGraph': 
       if grname in file:
          print( ' Error: file ' + grname + ' exists <<<<<<<<<<<================' )
       else:
          plt.savefig( f"{file}/{grname}" )
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
          
    return axt




def misalignments(line, seed=None):
   sigma=0.3e-3

   rng=np.random.default_rng(seed)

   actual_seed = rng.bit_generator.seed_seq.entropy
   print(f"Applying misalignments with seed: {actual_seed}")

   tab=line.get_table()
   #Quad and sextupole misalignments
   quads = list(tab.rows[tab.element_type == 'Quadrupole'].name)
   sexts = list(tab.rows[tab.element_type == 'Sextupole'].name)
   bends = list(tab.rows[tab.element_type == 'Bend'].name)

   for name in quads + sexts + bends:
        # Access the element via the reference manager
        ref = line.element_refs[name]
        
        # Apply translations
        ref.shift_x = rng.normal(0, sigma)
        ref.shift_y = rng.normal(0, sigma)
        ref.shift_s = rng.normal(0, sigma)
        
        # Apply rotations (in radians)
        ref.rot_s_rad = rng.normal(0, sigma)
        ref.rot_x_rad = rng.normal(0, sigma)
        ref.rot_y_rad = rng.normal(0, sigma)

        relative_error = 1 + rng.normal(0, 1e-3)
        ref.knl *= relative_error
        ref.ksl *= relative_error

   return line


def survey_plot(ring):
    fig, ax = plt.subplots(figsize=(12, 12))
    sv = ring.survey()
    sv.plot(ax=ax) 
    df = sv.to_pandas()

    bpm_offset = 3.0  # Meters to push BPMs out
    k_height = 2.0    # Transverse thickness of the kicker box 

    for name in df['name']:
        # Get the survey row for this element
        row = df[df['name'] == name].iloc[0]
        
        # Logic for Kickers (Mx and My)
        if name.startswith(('Mx', 'My')):
        
            length = ring.element_dict[name].length
            if length == 0: length = 0.5 # Minimum visible length if it's a marker
            
            color = 'hotpink' if name.startswith('Mx') else 'cyan'
            label = 'H-Kicker' if name.startswith('Mx') else 'V-Kicker'
            
        
            angle_deg = np.degrees(row['theta'])
            
        
            rect = patches.Rectangle(
                (row['Z'] - length/2, row['X'] - k_height/2), 
                length, k_height, 
                angle=angle_deg, rotation_point='center',
                color=color, zorder=10, label=label
            )
            ax.add_patch(rect)

        elif name.startswith(('BPMx', 'BPMy')):

            direction = row['theta'] + np.pi/2
            
            off_z = row['Z'] + bpm_offset * np.cos(direction)
            off_x = row['X'] + bpm_offset * np.sin(direction)
            
            if name.startswith('BPMx'):
                bpm_color = 'hotpink'
                bpm_label = 'BPM (Horizontal)'
            elif name.startswith('BPMy'):
                bpm_color = 'cyan'
                bpm_label = 'BPM (Vertical)'
            else:
                bpm_color = 'black'
                bpm_label = 'BPM (Other)'

            ax.scatter(off_z, off_x, color=bpm_color, s=20, 
                       zorder=11, label=bpm_label)

    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

    ax.set_aspect('equal')
    plt.show()


import pandas as pd
def survey_plot_with_strength(ring, scale_height=5.0):
    # 1. Get the survey data
    sv = ring.survey()
    df = sv.to_pandas()

    # 2. Extract strengths from the ring environment
    def get_element_strength(name):
        el = ring.element_dict.get(name)
        if el is None or isinstance(el, xt.Drift): 
            return None
        
        s = 0.0
        # 1. Check Multipoles (Dipoles, Quads, Sextupoles, Kickers)
        if hasattr(el, 'knl'):
            # Sum up all normal components (k0l, k1l, k2l...)
            # and skew components (ksl)
            s += np.sum(el.knl) + np.sum(el.ksl)
        
        # 2. Check Thick Magnets (if knl is empty)
        if s == 0:
            if hasattr(el, 'k1'): s += el.k1 * el.length
            if hasattr(el, 'k2'): s += el.k2 * el.length
            # Dipole bending strength for thick elements
            if hasattr(el, 'k0'): s += el.k0 * el.length 
            # Kickers often use 'hkick' or 'vkick'
            if hasattr(el, 'hkick'): s += el.hkick
            if hasattr(el, 'vkick'): s += el.vkick

        return s

    # Append the strength column to our dataframe
    df['strength'] = df['name'].apply(get_element_strength)
    
    # 3. Setup Plot
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.plot(df['Z'], df['X'], color='gray', lw=1, alpha=0.3) # Baseline
    
    bpm_offset = 3.0  
    k_height = 2.0    
    max_s = df['strength'].abs().max() if df['strength'].abs().max() != 0 else 1.0

    # 4. Iterate through the updated DataFrame
    for _, row in df.iterrows():
        name = row['name']
        z0, x0 = row['Z'], row['X']
        theta = row['theta']
        el_len = row['length']
        strength = row['strength']

        # --- STRENGTH BLOCKS ---
        if strength != 0:
            norm_val = strength / max_s
            draw_len = el_len if el_len > 0 else 0.5
            
            perp_angle = theta + np.pi/2
            h = norm_val * scale_height
            
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            cos_p, sin_p = np.cos(perp_angle), np.sin(perp_angle)
            
            # Corner coordinates using your ColView geometry
            c1 = [z0 - draw_len*cos_t,             x0 - draw_len*sin_t]
            c2 = [z0,                              x0]
            c3 = [z0 + h*cos_p,                    x0 + h*sin_p]
            c4 = [z0 - draw_len*cos_t + h*cos_p,   x0 - draw_len*sin_t + h*sin_p]
            
            color = 'red' if strength > 0 else 'blue'
            ax.add_patch(patches.Polygon([c1, c2, c3, c4], color=color, alpha=0.6, zorder=5))

        # --- KICKERS ---
        elif name.startswith(('Mx', 'My')):
            color = 'hotpink' if name.startswith('Mx') else 'cyan'
            # Centered rectangle for kickers
            rect = patches.Rectangle(
                (z0 - el_len, x0 - k_height/2), 
                el_len if el_len > 0 else 0.5, k_height, 
                angle=np.degrees(theta), rotation_point='center',
                color=color, zorder=10
            )
            ax.add_patch(rect)

        # --- BPMs ---
        elif name.startswith(('BPMx', 'BPMy')):
            direction = theta + np.pi/2
            ax.scatter(z0 + bpm_offset * np.cos(direction), 
                       x0 + bpm_offset * np.sin(direction), 
                       color='hotpink' if 'BPMx' in name else 'cyan', 
                       s=20, zorder=11)

    # 5. UI Elements (+/- guide and labels)
    # Use a point near the 5% mark of the ring to show the polarity guide
    guide = df.iloc[len(df)//20]
    g_theta = guide['theta'] + np.pi/2
    for sign, color, side in [("+", "red", 1.5), ("-", "blue", -1.5)]:
        ax.text(guide['Z'] + (scale_height*side)*np.cos(g_theta), 
                guide['X'] + (scale_height*side)*np.sin(g_theta), 
                sign, fontsize=30, color=color, weight='bold', ha='center')

    ax.set_aspect('equal')
    ax.set_title("Ring Survey: Position + Strength Envelope")
    plt.xlabel("Z [m]")
    plt.ylabel("X [m]")
    plt.show()
    
    return df # Returning the dataframe so you can inspect the 'strength' column

def geometric_acceptance_plot(
    x_norm,
    y_norm,
    num_r_steps,
    num_theta_steps,
    x_aperture,
    y_aperture
):


    x2d = x_norm.reshape(num_r_steps, num_theta_steps)
    y2d = y_norm.reshape(num_r_steps, num_theta_steps)


    accepted = (
        (x2d / x_aperture)**2 +
        (y2d / y_aperture)**2
    ) <= 1

    accepted = accepted.astype(float)

    x_GA = np.full(num_theta_steps, np.nan)
    y_GA = np.full(num_theta_steps, np.nan)

    for jj in range(num_theta_steps):

        for ii in range(num_r_steps):

            if not accepted[ii, jj]:

                x_GA[jj] = x2d[ii, jj]
                y_GA[jj] = y2d[ii, jj]
                break


    GA_radius = np.sqrt(x_GA**2 + y_GA**2)

    min_GA = np.nanmin(np.round(GA_radius, 1))

    where_min_GA = np.where(
        np.round(GA_radius, 1) == min_GA
    )[0]

    fig, ax = plt.subplots(figsize=(8, 6))

    mesh = ax.pcolormesh(
        x2d,
        y2d,
        accepted,
        shading='gouraud'
    )

    ax.plot(
        x_GA,
        y_GA,
        color='red',
        linewidth=2,
        label='Geometric Acceptance'
    )

    ax.plot(
        x_GA[where_min_GA],
        y_GA[where_min_GA],
        'o',
        color='red',
        markersize=7,
        label=f'GA$_{{min}}$ = {min_GA:.1f}$\\sigma$'
    )

    ax.set_xlabel(
        r'$\hat{x}\ [\sqrt{\varepsilon_x}]$'
    )

    ax.set_ylabel(
        r'$\hat{y}\ [\sqrt{\varepsilon_y}]$'
    )

    cb = plt.colorbar(mesh)
    cb.set_label('Accepted')

    ax.legend(fontsize='small')

    plt.tight_layout()

    return x_GA, y_GA
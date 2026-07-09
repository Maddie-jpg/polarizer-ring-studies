import numpy as np
import matplotlib.pyplot as plt
import xtrack as xt

#
#   A suite of tools for spin manipulations, in particular rotations around any axis
#   This file was updated for more accurate number for the me and ae
#

electron_anom_moment_a = 0.00115965218091
electron_mass_mev = 0.5109989461

# a few lines not to miss when setting up
def setupLine4Pol(inline, ref_ein_gev):
    inline.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV,
                                       gamma0=ref_ein_gev * 1E9 / xt.ELECTRON_MASS_EV)
    inline.particle_ref.anomalous_magnetic_moment=electron_anom_moment_a
    return

# return the spin tune for the energy in GeV
def spinTuneForE(en_gev):
    return en_gev/(electron_mass_mev/(1000*electron_anom_moment_a))

# return the energy in GeV for the spin tune
def EforSpinTune(nu_s):
    return nu_s*(electron_mass_mev/(1000*electron_anom_moment_a))

# return the gamma factor for the selected spin tune
def gammaForSpinTune(nu_s):
    return nu_s*electron_anom_moment_a

# return the gamma factor for the selected spin tune
def gammaForE(en_gev):
    return en_gev*1000/electron_mass_mev

#  calculate the deflection angle of the spin for a solenoid integrated field BsL (in Tm) and energy in GeV (energyGev)
# angle [rad] = e*(1+a)* BL / (m * c * beta * gamma) e* a * BL / P
#
def spinAngleBsol(BsL, energy_gev):
# old, not correct? pi instaed of 3? 
#    angle = BsL * np.pi * (1.00116) / (10.479 * energyGeV)
    angle = BsL * (1.00116) * 0.3 / energy_gev
    return angle

#  calculate the deflection angle of the spin for a transverse integrated field BtL (in Tm) and energy in GeV (energyGev)
# angle [rad] = e* a * BL / (m * c * beta) = e* a * BL * gamma / P
#
def spinAngleBtransverse(BtL, energy_gev):
    gammae = energy_gev * 1000 / electron_mass_mev
    angle = BtL * electron_anom_moment_a * 0.3 * gammae / energy_gev
    return angle
   
def rotationX(invec, theta):
    outvec = [0,0,0]
    outvec[0] = invec[0]
    outvec[1] = np.cos(theta)*invec[1] - np.sin(theta)*invec[2]
    outvec[2] = np.sin(theta)*invec[1] + np.cos(theta)*invec[2]
    return outvec

def rotationY(invec, theta):
    outvec = [0,0,0]
    outvec[1] = invec[1]
    outvec[0] = np.cos(theta)*invec[0] + np.sin(theta)*invec[2]
    outvec[2] = -np.sin(theta)*invec[0] + np.cos(theta)*invec[2]
    return outvec

def rotationS(invec, theta):
    outvec = [0,0,0]
    outvec[2] = invec[2]
    outvec[0] = np.cos(theta)*invec[0] - np.sin(theta)*invec[1]
    outvec[1] = np.sin(theta)*invec[0] + np.cos(theta)*invec[1]
    return outvec

#
# Rotation matrices around the 3 axis, returned as np array
#
def rotationMatrixX(theta):
    rmat = [[1,0,0],[0, np.cos(theta), - np.sin(theta)],[0, np.sin(theta), np.cos(theta)]]
    return np.array(rmat)
   
def rotationMatrixY(theta):
    rmat = [[np.cos(theta), 0, np.sin(theta)],[0,1,0],[ -np.sin(theta), 0, np.cos(theta)]]
    return np.array(rmat)
   
def rotationMatrixS(theta):
    rmat = [[np.cos(theta), -np.sin(theta), 0],[ np.sin(theta), np.cos(theta), 0],[0,0,1]]
    return np.array(rmat)

def identityMatrix():
    rmat = [[0 for x in range(3)] for y in range(3)]
    rmat[0][0] = 1
    rmat[1][1] = 1
    rmat[2][2] = 1
    return np.array(rmat)
    
def angleToYaxis(invec):
    vecnorm = 0
    for u in invec:
        vecnorm = vecnorm + u*u
    norm = invec[0]*invec[0]
    norm = norm + invec[2]*invec[2]
    norm = np.sqrt(norm)
    norm = norm / vecnorm
    return norm

#
# The angle of the spin vector in the x-y plane, 0 = along x axis
#
def spinAnglePlaneXS(invec):
    norm = np.sqrt(invec[0]*invec[0] + invec[2]*invec[2])
    if (invec[2] >= 0):
        angxs = np.arccos(invec[0]/norm)
    else:
        angxs = np.pi*2 - np.arccos(invec[0]/norm)    
    return angxs[0]

#
#  From Handbook of acc engineering (A Chao)
#  The energy is in GeV, the bending radius rho in m and the circumference C in m
#  This formula is an approximation
#
def polRiseTimeInSec(enGeV, rhom, Cm):
    tau = 99.0/(2*np.pi)
    tau = tau*(Cm*rhom*rhom)/(enGeV)**5
    return tau

#
# Print Q, Q', emittances, Qs, polarization etc
#
#
def printRaditionPol(twin):
    print(f"Qx / Qy   = {twin.qx:.4f} / {twin.qy:.4f}")
    print(f"Qx' / Qy' = {twin.dqx:.4f} / {twin.dqy:.4f}")
    print(f"Alpha_c   = {twin.momentum_compaction_factor:.4e}")
    print(f'Revolution period = {twin.t_rev0*1E6:.3f} [microsec]')
    print(f"P0c       = {twin.p0c*1E-9:.5f} [GeV]")
    print(f"Ref. Spin tune  = {spinTuneForE(twin.p0c*1E-9):.6f}")
    print(f"Frac. Spin tune = {twin.spin_tune_fractional:.6f}")
    print(f'Equ. pol  = {twin.spin_polarization_eq*100:.3f} %')
    print(f'Eloss/turn      = {twin.energy_loss/1e6:.3f} [MeV]')    
    print(f"Qs              = {twin.qs:.5f}")
    print(f"Sigma_z         = {np.sqrt(twin.eq_gemitt_zeta * twin.bets0):.4f} [m]")
    print(f"Sigma_E/E       = {1000*np.sqrt(twin.eq_gemitt_zeta / twin.bets0):.4f} [permill]")
# spin_mod_index = nus* sigmaE/E / Qs
    spinModIndex = spinTuneForE(twin.p0c*1E-9)*(np.sqrt(twin.eq_gemitt_zeta / twin.bets0))/twin.qs
    print(f"Spin mod index  = {spinModIndex:.4f}")
    print(f"Emitx / Emity   = {twin.rad_int_eq_gemitt_x*1E9:.4f} / {twin.rad_int_eq_gemitt_y*1E9:.4f} [nm]")
    print(f"Damping part No = {twin.partition_numbers[0]:.4f} / {twin.partition_numbers[1]:.4f} / {twin.partition_numbers[2]:.4f}")
#
# Print Q, Q', emittances, Qs, polarization etc
#
#
def printPol(twin):
    print(f"Qx / Qy   = {twin.qx:.4f} / {twin.qy:.4f}")
    print(f"Qx' / Qy' = {twin.dqx:.4f} / {twin.dqy:.4f}")
    print(f"Alpha_c   = {twin.momentum_compaction_factor:.4e}")
    print(f'Revolution period = {twin.t_rev0*1E6:.3f} [microsec]')
    print(f"P0c       = {twin.p0c*1E-9:.5f} [GeV]")
    print(f"Ref Spin tune   = {spinTuneForE(twin.p0c*1E-9):.6f}")
    print(f"Frac. Spin tune = {twin.spin_tune_fractional:.6f}")
    print(f'Equ. pol  = {twin.spin_polarization_eq*100:.3f} %')
    print(f'Tau pol   = {twin.spin_t_pol_buildup_s/3600:.3f} [h]')
#
# Print Q, Q', emittances, Qs, polarization etc as a header for a file
#
def writePolInfoToFile(fout, twin):
    fout.write(f"#QX , {twin.qx:.4f}\n")
    fout.write(f"#QY , {twin.qy:.4f}\n")
    fout.write(f"#QPX , {twin.dqx:.4f}\n")
    fout.write(f"#QPY , {twin.dqy:.4f}\n")
    fout.write(f"#ALPHAC , {twin.momentum_compaction_factor:.4e}\n")
    fout.write(f'#TREV , {twin.t_rev0*1E6:.3f} , MICROSEC\n')
    fout.write(f"#P0C , {twin.p0c*1E-9:.5f} , GEV\n")
    fout.write(f"#REFNUS , {spinTuneForE(twin.p0c*1E-9):.6f}\n")
    fout.write(f"#FRACNUS , {twin.spin_tune_fractional:.6f}\n")
    fout.write(f'#POL , {twin.spin_polarization_eq*100:.3f} , \%\n')
    fout.write(f'#TAUPOL , {twin.spin_t_pol_buildup_s/3600:.3f} , H')

#
#  Tracks the spin through a section of the machine from mustart to muend (V phase advance)
#  spinV is the initial spin vector.
#  For checkAngle = True a message is printed for non-dipole deflections > 1E-8 rad
#
def spinAnglePerSectionHist(twisstb, mustart, muend, nuSpin, spinV, debugOn = False, checkAngle = True):
    typear = twisstb.element_type
    k0lar = twisstb.k0l
    k1lar = twisstb.k1l
    kslar = twisstb.ks
    k2lar = twisstb.k2l
    vkickar = twisstb.vkick
    hkickar = twisstb.hkick
    yar = twisstb.y
    xar = twisstb.x
    lenar = twisstb.length
    muyar = twisstb.muy
    sar = twisstb.s
    namear = twisstb.name
    kk = 0

    spinIn = spinV

    angleMessCut = 5E-7

    history = []
    history.append(spinV)
    
    for eln in typear:
        isRotation = False
        if (muyar[kk] > mustart and muyar[kk] < muend):
            if eln.find('Bend') >= 0:
# must change sign for the bends !
                bangle = -k0lar[kk]*nuSpin
                if (debugOn == True):
                    print("Bend k0l " , bangle)
                spinOut = rotationY(spinIn, bangle)
                isRotation = True
                
            if eln.find('Quadrupole') == 0:
                qangley = k1lar[kk]*yar[kk]*(nuSpin+1)
                qanglex = -k1lar[kk]*xar[kk]*(nuSpin+1)
                if (debugOn == True or (checkAngle == True and (np.abs(qangley) > angleMessCut or np.abs(qanglex) > angleMessCut))):
                    print("Quad k1l " , k1lar[kk], " - angle y/x ", qangley, " / ", qanglex)
                spinTmp = rotationX(spinIn, qangley)
                spinOut = rotationY(spinTmp, qanglex)
                isRotation = True
                
            if eln.find('Sextupole') == 0:
                sanglex = -k2lar[kk]*0.5*(xar[kk]*xar[kk]-yar[kk]*yar[kk])*(nuSpin+1)
                sangley = k2lar[kk]*xar[kk]*yar[kk]*(nuSpin+1)
                if (debugOn == True or (checkAngle == True and (np.abs(sanglex) > angleMessCut or np.abs(sangley) > angleMessCut) ) ):
                   print("Sextupole k2l " , k2lar[kk], " - angle (x/y) ", sanglex , " / ", sangley)
                spinTmp = rotationX(spinIn, sangley)
                spinOut = rotationY(spinTmp, sanglex)
                isRotation = True
                
            if eln.find('Solenoid') == 0 or eln.find('UniformSolenoid') == 0:
                sangle = -kslar[kk]*lenar[kk]
                if (debugOn == True or checkAngle == True):
                    print("Solenoid angle " , sangle, " k1s ", kslar[kk])
                spinOut = rotationS(spinIn, sangle)
                isRotation = True
                   
            if eln.find('Multipole') == 0 or eln.find('Kicker') == 0 or eln.find('Mag') >=0:
                kangley = vkickar[kk]*(nuSpin+1)
                kanglex = hkickar[kk]*(nuSpin+1)
                if (debugOn == True or (checkAngle == True and (np.abs(kangley) > angleMessCut or np.abs(kanglex) > angleMessCut))):
                    print("Multipole hkick / vkick " , kanglex, " / ", kangley)
                spinTmp = rotationX(spinIn, kangley)
                spinOut = rotationY(spinTmp, kanglex)
                isRotation = True

        if (isRotation):
            spinIn = spinOut
            history.append(spinOut)
            
        kk = kk + 1
        
    return spinIn, history

#
#  Tracks the spin through a section of the machine from mustart to muend (V phase advance)
#  spinV is the initial spin vector.
#  This version also handles the case of UniformSolenoids that are sliced
#  The first argument is the line (tofind a parent element)
#  For checkAngle = True a message is printed for non-dipole deflections > 1E-8 rad
#
def spinAnglePerSectionHistNew(inline, twisstb, mustart, muend, nuSpin, spinV, debugOn = False, checkAngle = True):
    typear = twisstb.element_type
    k0lar = twisstb.k0l
    k1lar = twisstb.k1l
    kslar = twisstb.ks
    k2lar = twisstb.k2l
    vkickar = twisstb.vkick
    hkickar = twisstb.hkick
    yar = twisstb.y
    xar = twisstb.x
    lenar = twisstb.length
    muyar = twisstb.muy
    sar = twisstb.s
    namear = twisstb.name
    kk = 0

    spinIn = spinV

    angleMessCut = 5E-7

    history = []
    history.append(spinV)
    
    for eln in typear:
        isRotation = False
        if (muyar[kk] > mustart and muyar[kk] < muend):
            if eln.find('Bend') >= 0:
# must change sign for the bends !
                bangle = -k0lar[kk]*nuSpin
                if (debugOn == True):
                    print("Bend k0l " , bangle)
                spinOut = rotationY(spinIn, bangle)
                isRotation = True
                
            if eln.find('Quadrupole') == 0:
                qangley = k1lar[kk]*yar[kk]*(nuSpin+1)
                qanglex = -k1lar[kk]*xar[kk]*(nuSpin+1)
                if (debugOn == True or (checkAngle == True and (np.abs(qangley) > angleMessCut or np.abs(qanglex) > angleMessCut))):
                    print("Quad k1l " , k1lar[kk], " - angle y/x ", qangley, " / ", qanglex)
                spinTmp = rotationX(spinIn, qangley)
                spinOut = rotationY(spinTmp, qanglex)
                isRotation = True
                
            if eln.find('Sextupole') == 0:
                sanglex = -k2lar[kk]*0.5*(xar[kk]*xar[kk]-yar[kk]*yar[kk])*(nuSpin+1)
                sangley = k2lar[kk]*xar[kk]*yar[kk]*(nuSpin+1)
                if (debugOn == True or (checkAngle == True and (np.abs(sanglex) > angleMessCut or np.abs(sangley) > angleMessCut) ) ):
                   print("Sextupole k2l " , k2lar[kk], " - angle (x/y) ", sanglex , " / ", sangley)
                spinTmp = rotationX(spinIn, sangley)
                spinOut = rotationY(spinTmp, sanglex)
                isRotation = True
                
            if eln.find('Solenoid') == 0:
                sangle = -kslar[kk]*lenar[kk]
                if (debugOn == True or checkAngle == True):
                    print("Solenoid angle " , sangle, " k1s ", kslar[kk])
                spinOut = rotationS(spinIn, sangle)
                isRotation = True
                
            if eln.find('UniformSolenoid') == 0:
                sangle = kslar[kk]*lenar[kk]
                if (debugOn == True or checkAngle == True):
                    print("Solenoid angle " , sangle, " k1s ", kslar[kk])
                spinOut = rotationS(spinIn, sangle)
                isRotation = True
       
            if eln.find('ThickSliceUniformSolenoid') == 0:
                solElName = namear[kk]
                solEl = inline[solElName]
                solParent = inline[solEl.parent_name]
                sangle = solParent.ks*solEl.weight*solParent.length*(1.00116)
                if (debugOn == True or checkAngle == True):
                    print("Solenoid angle " , sangle, " k1s ", kslar[kk])
                spinOut = rotationS(spinIn, sangle)
                isRotation = True
            
            if eln.find('Multipole') == 0 or eln.find('Kicker') == 0 or eln.find('Mag') >=0:
                kangley = vkickar[kk]*(nuSpin+1)
                kanglex = hkickar[kk]*(nuSpin+1)
                if (debugOn == True or (checkAngle == True and (np.abs(kangley) > angleMessCut or np.abs(kanglex) > angleMessCut))):
                    print("Multipole hkick / vkick " , kanglex, " / ", kangley)
                spinTmp = rotationX(spinIn, kangley)
                spinOut = rotationY(spinTmp, kanglex)
                isRotation = True

        if (isRotation):
            spinIn = spinOut
            history.append(spinOut)
            
        kk = kk + 1
        
    return spinIn, history

#
#  Build the one-turn spin matrix based on the input value of the spin tune
#
#  This version is fixed for the new solenoid modelling where the strength is in the parent
#
#  inline : Xsuite line
#  twisstb : is the  twiss
#  nuSpin : is the reference spin value  
#  For checkAngle = True a message is printed for non-dipole deflections > 1E-8 rad
#  Returns the total H bending angle and the one-turn matrix
#
def oneTurnSpinRotationMatrix(inline, twisstb, nuSpin, debugOn = False, checkAngle = True):
    typear = twisstb.element_type
    k0lar = twisstb.k0l
    k1lar = twisstb.k1l
    kslar = twisstb.ks
    k2lar = twisstb.k2l
    vkickar = twisstb.vkick
    hkickar = twisstb.hkick
    yar = twisstb.y
    xar = twisstb.x
    lenar = twisstb.length
    muyar = twisstb.muy
    namear = twisstb.name
    kk = 0

    totalBendAng = 0

    angleMessCut = 5E-7

# we start with the unit matrix
    
    turnMatrix = np.array([[1,0,0], [0,1,0], [0,0,1]])
     
    for eln in typear:
        isRotation = False
        rMat = np.array([[1,0,0], [0,1,0], [0,0,1]])
        
        if eln.find('Bend') >= 0:
# must change sign for the bends !
            bangle = -k0lar[kk]*nuSpin
            totalBendAng = totalBendAng + k0lar[kk]
            if (debugOn == True):
                print("Bend k0l " , bangle)
            rMat = rotationMatrixY(bangle)
            
            isRotation = True
            
        if eln.find('Quadrupole') == 0:
            qangley = k1lar[kk]*yar[kk]*(nuSpin+1)
            qanglex = -k1lar[kk]*xar[kk]*(nuSpin+1)
            if (debugOn == True or (checkAngle == True and (np.abs(qangley) > angleMessCut or np.abs(qanglex) > angleMessCut))):
                print("Quad k1l " , k1lar[kk], " - angle y/x ", qangley, " / ", qanglex)
                
            rMaty = rotationMatrixX(qangley)
            rMatx = rotationMatrixY(qanglex)
            rMat = np.matmul(rMaty, rMatx)
         
            isRotation = True
            
        if eln.find('Sextupole') == 0:
            sanglex = -k2lar[kk]*0.5*(xar[kk]*xar[kk]-yar[kk]*yar[kk])*(nuSpin+1)
            sangley = k2lar[kk]*xar[kk]*yar[kk]*(nuSpin+1)
            if (debugOn == True or (checkAngle == True and (np.abs(sanglex) > angleMessCut or np.abs(sangley) > angleMessCut) ) ):
                print("Sextupole k2l " , k2lar[kk], " - angle (x/y) ", sanglex , " / ", sangley)
            rMatx = rotationMatrixX(sangley)
            rMaty = rotationMatrixY(sanglex)
            rMat = np.matmul(rMaty, rMatx)
            
            isRotation = True
            
        if eln.find('Solenoid') == 0 or eln.find('UniformSolenoid') == 0:
            sangle = -kslar[kk]*lenar[kk]*(1.00116)
            if (debugOn == True or checkAngle == True):
                print("Solenoid angle " , sangle, " k1s ", kslar[kk])
            rMat = rotationMatrixS(sangle)

            isRotation = True
            
        if eln.find('ThickSliceUniformSolenoid') == 0:
            solElName = namear[kk]
            solEl = inline[solElName]
            solParent = inline[solEl.parent_name]
            sangle = solParent.ks*solEl.weight*solParent.length*(1.00116)
            if (debugOn == True or checkAngle == True):
                print("Solenoid angle " , sangle, " k1s ", kslar[kk])
            rMat = rotationMatrixS(sangle)

            isRotation = True
            
        if eln.find('Multipole') == 0 or eln.find('Kicker') >= 0 or eln.find('Mag') >=0 :
            kangley = vkickar[kk]*(nuSpin+1)
            kanglex = hkickar[kk]*(nuSpin+1)
            if (debugOn == True or (checkAngle == True and (np.abs(kangley) > angleMessCut or np.abs(kanglex) > angleMessCut))):
                print("Multipole hkick / vkick " , kanglex, " / ", kangley)
                
            rMaty = rotationMatrixX(kangley)
            rMatx = rotationMatrixY(kanglex)
            rMat = np.matmul(rMaty, rMatx)            
            
            isRotation = True
            
        if (isRotation == True):
            newMatrix = np.matmul(rMat, turnMatrix)
            turnMatrix = newMatrix
        
        kk = kk + 1
        
    return totalBendAng, turnMatrix

#
#   Given the 1turn spin matrix, extract the spin tune from the trace
#
def spinTuneFromTrace(spinMatrix):
    
    trace1t = np.trace(spinMatrix)
    nus = np.arccos((trace1t-1)/2)/(2*np.pi)

    return nus

# plot the history of the spin vector in X-S plane. 
# the title is used to open a file for a picture
#
# useLine: if True plots symbols and lines between points, else only points
#
def plotSpinHistory(spinHis, xylim, title, useline=True):
    plt.close('all')

    figspin = plt.figure(1, figsize=(12., 12.))
    spinpl = plt.subplot(1,1,1)

    spinpl.tick_params(axis='both', direction = 'in')
    histColMap = [plt.cm.plasma(i) for i in np.linspace(0,1,len(spinHis)+1)]        
    plt.gca().set_prop_cycle(color=histColMap)
    
    radTomrad = 1000
    
#    for spinVal in spinHis:
#        spinpl.plot(spinVal[2]*radTomrad, spinVal[0]*radTomrad, 'o');

    plotStyle = 'o-'
    if (useline == False):
        plotStyle = 'o'
    
    for i in range(0, len(spinHis)-1, 1):
        xv = [spinHis[i][2]*radTomrad, spinHis[i+1][2]*radTomrad]
        yv = [spinHis[i][0]*radTomrad, spinHis[i+1][0]*radTomrad]
        spinpl.plot(xv, yv, 'o-');
    
    spinpl.set_ylabel(r'$n_{x}$ [mrad]')
    spinpl.set_xlabel(r'$n_{s}$ [mrad]')

    spinpl.set_xlim(-xylim, xylim)
    spinpl.set_ylim(-xylim, xylim)
    spinpl.grid(True)

    figspin.subplots_adjust(left=.15, right=.92, hspace=.27)
    plt.show() 
    figspin.savefig(title + ".png", transparent=False, facecolor='white', bbox_inches='tight')
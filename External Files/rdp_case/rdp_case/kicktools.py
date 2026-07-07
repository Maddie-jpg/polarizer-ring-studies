import numpy as np

from datetime import date
from datetime import datetime

#
#  A library with tools for kicks
#

def check_xsuite_var(inline, var_name):
    exists_var = True
    try:
        inline.vars[var_name]
    except:
        print('Variable does not exist : ' + var_name)
        exists_var = False

    return exists_var

#
#   Sets the expression for a list of elements in table var_table
#   Sets the expression as the value + the strength new_str
#
def insert_expr(inline, var_table, new_str):
    
    for knext in var_table.name:
        testStr = str(inline[knext]) + ' + ' + new_str
        inline.vars[knext] = testStr

    return

#
# Returns a string with the date-time in a format to use for example in a file name
#
def getDateTimeString():
    formatD = date.today().isoformat()
    timenow = datetime.today()
    yFm = formatD + '_' + str(timenow.hour) + '-' + str(timenow.minute) + '-' + str(timenow.second)
    return yFm

# calculate the kick in radian from the integrated B field in Tm and the energy in GeV
#  bltm : integrated B field in Tm
#  energyGeVl : the beam energy in GeV
#
def fromBinTm2KickV2(bltm, energyGeV):
    kick = 47.7E-3*bltm*2*np.pi/(energyGeV)
    return kick

# the classical way
#  bltm : integrated B field in Tm
#  energyGeVl : the beam energy in GeV
def fromBinTm2Kick(bltm, energyGeV):
    kick = bltm*0.3/(energyGeV)
    return kick

#
# Returns the B-rho value (= P/q) in Tm for the input momentum given in GeV/c
#
def bRhoFromPinGeV(pGeV):
    brho = 3.3356*pGeV
    return brho
#
#  Returns the total RF voltage in MV of the line
#
def getTotalRfVoltageMV(inline):
    tab = inline.get_table()
    tt_rfc = tab.rows[tab.element_type == 'Cavity']

    vrfTot = 0
    for cavn in tt_rfc.name:
        vrfTot = vrfTot + inline[cavn].voltage
    return vrfTot*1E-6

#
#  returns the s-weighted mean energy of the machine
#
def dpAverageFromTwiss(twin):
    dpar = twin.delta
    sar = twin.s
    dpmax = 1000*np.max(dpar)
    dpmin = 1000*np.min(dpar)
    dpav = 1000*np.average(dpar)
# integrate over s coordinate
    prevs = 0
    stot = 0
    dptot  = 0
    for nexts,nextdp in zip(sar, dpar):
        stot = nexts
        dptot = dptot + (nexts-prevs)*nextdp
        prevs = nexts

    dpmean = dptot/stot
    print(f'dp/p mean/min/max [permill] = {dpmean:.6f}/{dpmin:.6f}/{dpmax:.6f}')
    return dpmean

#
# Lists the elements of inline from element fromEl to element toEl
# prints name and s position
#
def listElementsFromTo(inline, fromEl, toEl):

    testTabLine = inline.get_table()
    testSubTabLine = testTabLine.rows[fromEl:toEl]
    
    kkk = 0
    elCnt = 0
    previousS = testSubTabLine['s'][0]
    for el in testSubTabLine['name']:
        elCnt = elCnt + 1
        stemp = testSubTabLine['s'][kkk]
        sdelta = stemp - previousS
        print(f'{elCnt:6d} {el:20s} @ {stemp:12.6f} m [Delta = {sdelta:10.6f}]')

        previousS = testSubTabLine['s'][kkk]
        
        kkk = kkk+1

    return
#
# For an input radiation/6d twiss, prints the main variables (emittances, damping times ...)
#
def printRadiationVars(twin):
    eloss = twin['energy_loss']/1E6
# we must use the geometric emittance 
    emitx = twin['eq_gemitt_x']*1E12
    emity = twin['eq_gemitt_y']*1E12
    print('Energy loss / turn [MeV] = ', f'{eloss:7.2f}' )
    print('Hor emittance [pm] = ' , f'{emitx:7.3f}')
    print('Ver emittance [pm] = ' , f'{emity:7.3f}')
    print('Emittance ratio = ' , f'{emity/emitx:7.4f}')    
    print('Long emittance [] = ' , twin['eq_nemitt_zeta'])
    print('Damping time X [ms] = ' , f'{1000/twin.damping_constants_s[0]:.2f}')
    print('Damping time Y [ms] = ' , f'{1000/twin.damping_constants_s[1]:.2f}')
    print('Damping time Z [ms] = ' , f'{1000/twin.damping_constants_s[2]:.2f}')
    print('Damping part no X = ' , f'{twin.partition_numbers[0]:.4f}')
    print('Damping part no Y = ' , f'{twin.partition_numbers[1]:.4f}')
    print('Damping part no Z = ' , f'{twin.partition_numbers[2]:.4f}')
    qsTune = twin['qs']
    print('Qs = ' , f'{twin.qs:7.4f}')
    
    return

def printCoupling(twin):
    print(f'C- = {twin.c_minus:.5f} : Re = {twin.c_minus_im_0:.5f} , Im = {twin.c_minus_re_0:.5f}' )
    return 
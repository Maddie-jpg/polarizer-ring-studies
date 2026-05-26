import xtrack as xt
import xpart as xp
import numpy as np
#######################
# whole magnet clic pdr
#######################

### we generally want whole element in the lattice to keep element shifts and fringe fields geometrically consistent with the physical ring
### we will match shallow copies of those elements symmetrically, starting and ending at symmetry point (ie: imposing symmetry condition α=0 explicitly) to ge the right knobs for our ring design


#########################################
# beam definition
#########################################
particle = "positron"               
energy0_gev = 2.86                                      
npart_per_bunch = 4.5e9            
#kbunch = 312       # for 2 ghz
kbunch = 156 # we do 1 ghz 
#n_trains = 1 # for 2 ghz
n_trains = 2 # we do 1 ghz
#h = 2596 # for 2 ghz
h = 1298 # we do 1 ghz
ex_inj = 1.250699e-6                   
ey_inj = 1.250699e-3                
et = 7.5e-5                        
sigt_m = 0.009
sige = 0.015

particle_ref = xp.Particles(
    mass0=xp.ELECTRON_MASS_EV,        
    q0=+1,                            
    energy0=energy0_gev * 1e9          
)


#########################################
# xsuite env
#########################################   
env = xt.get_environment()            
env.particle_ref=particle_ref
env.vars.default_to_zero = False              ###

#########################################
# rigidity
#########################################
env.vars.update({  
    "nd": 38,           # number of dipoles [-] ### set following the resonance free lattice condition for avoiding nonlin resonances
    'ntme': 32,         # number of tme cells
    "nfodow": 18,       # number of fodow cells
    "en": 2.86,         # energy [gev]
    "bb": 1.2,          # arc bending field [t]
    "bw": 1.9,          # wiggler field [t]
})

env['trep'] = 20 # repetition rate [ms] 

#########################################
# tme parameters
#########################################

# tme drift lengths
env.vars.update({
    "l1a": 0.9,
    "l2a": 0.6,
    "l3a": 0.5 * 2,
    "lls": 0.3,  # length of all sexts
#    "lls2": 0.15, # not used
})

#########################################
# tme elements
#########################################

# ---dipole--- 
#env['nd'] = 38
#env["ang"] = 2*np.pi / (env["nd"] * 2)            # bending angle of half dipole [rad] 
env["ang"] = 2*np.pi / (env["nd"])                 # bending angle of whole dipole [rad] ### theta_whole = 2*theta_half =>
env["r"]   = (1/0.2998) * (env["en"] / env["bb"])  # bending radius [m] ### rho_whole = rho_half
env["ld"]  = env["ang"] * env["r"]                 # dipole length [m] =1.3144662215258722 ### => ldip_whole = 2*l_dip_half  ##### deferred expression
#env["ld"] = "ang * r"
#line.set('ld', "ang * r")        ##### how to set deferred expression?

env.new(
    "dip", xt.Bend,
    length=env["ld"], angle=env["ang"],    ###
#    edge_entry_angle=env["ang"]/2, edge_exit_angle=0,    ###
#    ,force=True
)

# ---arc quads---
env.vars.update({  
    "kqfa":  2.492391737,     ### knobs note: already close to their matched values
    "kqda": -2.069416029,
    "lq": 0.28,
})

env.new("qfa", xt.Quadrupole, length=env["lq"],   k1="kqfa") #) #, force=True) 
env.new("qda", xt.Quadrupole, length=env["lq"],   k1="kqda") #) #, force=True)

# ---arc sexts---
env.vars.update({
    "ksx1": 0.0,   ### knobs note: 1 sy1 and 2 sx1 in the tme, while qfa, qda approc same strengths: sy1 needs to "work more" for chroma correction
    "ksy1": 0.0,
})

env.new("sx1", xt.Sextupole,  length=env["lls"], k2="ksx1") #) #, force=True)
env.new("sy1", xt.Sextupole,  length=env["lls"], k2="ksy1") #) #, force=True)

#########################################
# tme cell
#########################################

##### define the edge to edge drift lengths #####
env.vars.update({
    "dtme_1": (env["l1a"] - env["lls"]) / 2,
    "dtme_2": (env["l1a"] - env["lls"] - env["lq"]) / 2,
    "dtme_3": env["l2a"] - env["lq"],
    "dtme_4": (env["l3a"] - env["lq"] - env["lls"]) / 2,
})

# drift elements
env.new("d_tme_1", xt.Drift, length=env["dtme_1"]) #) #, force=True) # dip2->sx1
env.new("d_tme_2", xt.Drift, length=env["dtme_2"]) #) #, force=True) # sx1->qfa
env.new("d_tme_3", xt.Drift, length=env["dtme_3"]) #) #, force=True) # qfa->qda
env.new("d_tme_4", xt.Drift, length=env["dtme_4"]) #) #, force=True) # qda->sy1

# ------tme build------
env.new_line(
    name="tme",
    components=[
    'dip', ###
    env.new('d_tme_1.1', 'd_tme_1'), # ('name d_tme copy', 'brings the knobs')
    env.new('sx1.1', 'sx1'),
    env.new('d_tme_2.1', 'd_tme_2'),
    env.new('qfa.1', 'qfa'),
    env.new('d_tme_3.1', 'd_tme_3'),
    env.new('qda.1', 'qda'),
    env.new('d_tme_4.1', 'd_tme_4'),
    env.new('sy1.1', 'sy1'), ### sym point
    env.new('d_tme_4.2', 'd_tme_4'),
    env.new('qda.2', 'qda'),
    env.new('d_tme_3.2', 'd_tme_3'),
    env.new('qfa.2', 'qfa'),
    env.new('d_tme_2.2', 'd_tme_2'),
    env.new('sx1.2', 'sx1'),
    env.new('d_tme_1.2', 'd_tme_1'),\
],
)

#########################################
# wiggler params
#########################################

# wiggler lengths and bending angle
env['nwigg'] = 36
env["wigglerperiod"] = 0.3                       # [m]
env["polelength"] = env["wigglerperiod"] / 4.0   # [m]
env["wigangle"] = 1.0027 * env["polelength"] * env["bw"] / (3.356 * env["en"])  # [rad]   

#########################################
# wiggler poles (hard-edge model)
#########################################

# wiggler poles
wigpolepos = env.new(
    "wigpolepos", xt.RBend,
    length_straight=env["polelength"], angle=env["wigangle"],
    edge_entry_angle=0.0, edge_exit_angle=0.0, rbend_model='straight-body',               
    #    force=True,
)
wigpoleneg = env.new(
    "wigpoleneg", xt.RBend,
    length_straight=env["polelength"], angle=-env["wigangle"],
    edge_entry_angle=0.0, edge_exit_angle=0.0, rbend_model='straight-body',
    #    force=True,
)

# and half poles
wighalfpolepos = env.new(
    "wighalfpolepos", xt.RBend,
    length_straight=env["polelength"]/2.0, angle=env["wigangle"]/2.0,
    edge_entry_angle=0.0, edge_exit_angle=0.0, rbend_model='straight-body',
    #    force=True,
)

wighalfpoleneg = env.new(
    "wighalfpoleneg", xt.RBend,
    length_straight=env["polelength"]/2.0, angle=-env["wigangle"]/2.0,
    edge_entry_angle=0.0, edge_exit_angle=0.0, rbend_model='straight-body',
    #    force=True,
)

# drift between poles  
wigdrift = env.new("wigdrift", xt.Drift, length=env["polelength"]) #) #, force=True)   # wigdrift = polelength

#########################################
# wiggler build
#########################################

wig_components = [wighalfpolepos, wigdrift]  # start with one halfpolepos and drift 

for i in range(19):           # -> then alternating between 19 poles (10 poleneg and 9 polepos) including drifts inbetween
    wig_components.append(wigpoleneg if (i % 2 == 0) else wigpolepos)
    wig_components.append(wigdrift)

wig_components.append(wighalfpolepos)  # -> finish by adding one halfpolepos

env.new_line(name="wiggler", components=wig_components)
env["wiggler"].replace_all_repeated_elements() ###       

#########################################
# f0d0w elements
#########################################

# f0d0w lengths
env["lq1w"] = 0.1 * 2      # [m] ### whole quad length l_whole = 2*l_half
env["lq2w"] = 0.1 * 2      # [m]
env["lwd1"] = 0.5          # [m] drift lengths
env["lwd2"] = 0.5          # [m]

env.vars.update({               
    "kqfw":  1.55261226,          # straight f quad strength [1/m^2] ### k_whole = k_half
    "kqdw": -1.301643467,
})

env.new("qfw", xt.Quadrupole, length=env["lq1w"], k1="kqfw") #) #, force=True)     
env.new("qdw", xt.Quadrupole, length=env["lq2w"], k1="kqdw") #) #, force=True)

env.new("wd1",  xt.Drift, length=env["lwd1"]) #) #, force=True)
env.new("wd2",  xt.Drift, length=env["lwd2"]) #) #, force=True)

# -----fodow build-----
env.new_line(
    name="fodow",
    components=['qfw', ### 
                env.new('wd2.1', 'wd2'),
                env["wiggler"].clone(suffix=".1"),
                env.new('wd1.1', 'wd1'),
                'qdw', ###
                env.new('wd1.2', 'wd1'),
                env["wiggler"].clone(suffix=".2"),
                env.new('wd2.2', 'wd2')],
)

#########################################
# dispsup-beta_matching:
# ds-bm sections
#########################################

# ds & mds quads
env["lqds"]   = 0.35

env.vars.update({
    "kqds11": -0.951299623,     ### ds knobs: kqds11-17         
    "kqds12":  3.215394621,
    "kqds13": -2.322720333,
    "kqds14": -1.22300104,
    "kqds15":  2.078486212,
    "kqds16": -0.1274254316,
    "kqds17": -1.306390283,     
    "kqds18":  0.8174937228,    ### mds knobs: kqds18,19 + kqds21-17
    "kqds19": -0.7202944284,
    "kqds21":  2.19351565,
    "kqds22": -3.193124933,
    "kqds23":  2.952413639,
    "kqds24": -1.518275912,
    "kqds25":  0.5014376022,
    "kqds26":  1.323682568,
    "kqds27": -1.411675205,
}) 

# ds quads
env.new("qds11",  xt.Quadrupole, length=env["lqds"], k1="kqds11") #, force=True)
env.new("qds12",  xt.Quadrupole, length=env["lqds"], k1="kqds12") #, force=True)
env.new("qds13",  xt.Quadrupole, length=env["lqds"], k1="kqds13") #, force=True)
env.new("qds14",  xt.Quadrupole, length=env["lqds"], k1="kqds14") #, force=True)
env.new("qds15",  xt.Quadrupole, length=env["lqds"], k1="kqds15") #, force=True)
env.new("qds16",  xt.Quadrupole, length=env["lqds"], k1="kqds16") #, force=True)
env.new("qds17",  xt.Quadrupole, length=env["lqds"], k1="kqds17") #, force=True)
env.new("qds18",  xt.Quadrupole, length=env["lqds"], k1="kqds18") #, force=True)
env.new("qds19",  xt.Quadrupole, length=env["lqds"], k1="kqds19") #, force=True)

# mds quads
env.new("qds21", xt.Quadrupole, length=env["lqds"], k1="kqds21") #, force=True)
env.new("qds22", xt.Quadrupole, length=env["lqds"], k1="kqds22") #, force=True)
env.new("qds23", xt.Quadrupole, length=env["lqds"], k1="kqds23") #, force=True)
env.new("qds24", xt.Quadrupole, length=env["lqds"], k1="kqds24") #, force=True)
env.new("qds25", xt.Quadrupole, length=env["lqds"], k1="kqds25") #, force=True)
env.new("qds26", xt.Quadrupole, length=env["lqds"], k1="kqds26") #, force=True)
env.new("qds27", xt.Quadrupole, length=env["lqds"], k1="kqds27") #, force=True)

# ds & mds sexts (powered off)
env.vars.update({
    "ksx2": 0.0,     
    "ksy2": 0.0,
})

env.new("sx2", xt.Sextupole,  length=env["lls"], k2="ksx2") #, force=True)
env.new("sy2", xt.Sextupole,  length=env["lls"], k2="ksy2") #, force=True)

# all ds-bm regions (ds & mds) drift lengths
env.vars.update({
    "lds11": 0.6,
    "lds12": 0.6,
    "lds13": 0.5,
    "lds14": 0.5,
    "lds15": 0.5,
    "lds16": 0.5,
    "lds17": 2.0,
    "lds18": 2.0,# mds_2
    "lds19": 2.0,# mds_1
    "lds110": 3.0,# ds_10
    "lds111": 3.0,# ds_11

    "lds21": 0.6,# ds_1 # mds_9
    "lds22": 0.6,# ds_2 # mds_8
    "lds23": 0.5,# ds_3 # mds_7
    "lds24": 0.5,# ds_4 # mds_6
    "lds25": 0.4,# ds_5 # mds_5
    "lds26": 0.6,# ds_6 # mds_4
    "lds27": 0.6,# ds_7 # mds_3
    "lds28": 3.3,# ds_8  
    "lds29": 3.3 ,# ds_9 
})

# ds total length
env["lds"] = (
    env["ld"]
  + env["lds21"] + env["lqds"]
  + env["lds22"] + env["lqds"]
  + env["lds23"] + env["lqds"]
  + env["lds24"]
  + env["ld"]
  + env["lds25"] + env["lqds"]
  + env["lds26"] + env["lqds"]
  + env["lds27"] + env["lqds"]
  + env["lds28"] + env["lqds"]
  + env["lds29"] + env["lqds"]
  + env["lds110"] + env["lqds"]
  + env["lds111"]
)

# ----------ds build----------
# define end Drift
env.new("dend", xt.Drift, length=env["lds111"]) #, force=True)

# ds_1
env.new_line(
    name="ds_1",
    length=env["lds"],
    components=[
        env.place(env.new("dip.ds_1.1", "dip"), anchor="center", at=env["ld"]/2),

        env.place(env.new("sx2.ds_1.1", "sx2"), anchor="center",
                  at=env["ld"] + (env["lds21"]-env["lls"])/2 + env["lls"]/2),

        env.place(env.new("qds11.ds_1.1", "qds11"), anchor="center",
                  at=env["ld"] + env["lds21"] + env["lqds"]/2),

        env.place(env.new("qds12.ds_1.1", "qds12"), anchor="center",
                  at=env["ld"] + env["lds21"] + env["lqds"]
                  + env["lds22"] + env["lqds"]/2),

        env.place(env.new("sy2.ds_1.1", "sy2"), anchor="center",
                  at=env["ld"] + env["lds21"] + env["lqds"]
                  + env["lds22"] + env["lqds"]
                  + (env["lds23"]-env["lls"])/2 + env["lls"]/2),

        env.place(env.new("qds13.ds_1.1", "qds13"), anchor="center",
                  at=env["ld"] + env["lds21"] + env["lqds"]
                  + env["lds22"] + env["lqds"]
                  + env["lds23"] + env["lqds"]/2),

        env.place(env.new("dip.ds_1.2", "dip"), anchor="center",
                  at=env["ld"] + env["lds21"] + env["lqds"]
                  + env["lds22"] + env["lqds"]
                  + env["lds23"] + env["lqds"]
                  + env["lds24"] + env["ld"]/2),

        env.place(env.new("qds14.ds_1.1", "qds14"), anchor="center",
                  at=env["ld"] + env["lds21"] + env["lqds"]
                  + env["lds22"] + env["lqds"]
                  + env["lds23"] + env["lqds"]
                  + env["lds24"] + env["ld"]
                  + env["lds25"] + env["lqds"]/2),

        env.place(env.new("qds15.ds_1.1", "qds15"), anchor="center",
                  at=env["ld"] + env["lds21"] + env["lqds"]
                  + env["lds22"] + env["lqds"]
                  + env["lds23"] + env["lqds"]
                  + env["lds24"] + env["ld"]
                  + env["lds25"] + env["lqds"]
                  + env["lds26"] + env["lqds"]/2),

        env.place(env.new("qds16.ds_1.1", "qds16"), anchor="center",
                  at=env["ld"] + env["lds21"] + env["lqds"]
                  + env["lds22"] + env["lqds"]
                  + env["lds23"] + env["lqds"]
                  + env["lds24"] + env["ld"]
                  + env["lds25"] + env["lqds"]
                  + env["lds26"] + env["lqds"]
                  + env["lds27"] + env["lqds"]/2),

        env.place(env.new("qds17.ds_1.1", "qds17"), anchor="center",
                  at=env["ld"] + env["lds21"] + env["lqds"]
                  + env["lds22"] + env["lqds"]
                  + env["lds23"] + env["lqds"]
                  + env["lds24"] + env["ld"]
                  + env["lds25"] + env["lqds"]
                  + env["lds26"] + env["lqds"]
                  + env["lds27"] + env["lqds"]
                  + env["lds28"] + env["lqds"]/2),

        env.place(env.new("qds18.ds_1.1", "qds18"), anchor="center",
                  at=env["ld"] + env["lds21"] + env["lqds"]
                  + env["lds22"] + env["lqds"]
                  + env["lds23"] + env["lqds"]
                  + env["lds24"] + env["ld"]
                  + env["lds25"] + env["lqds"]
                  + env["lds26"] + env["lqds"]
                  + env["lds27"] + env["lqds"]
                  + env["lds28"] + env["lqds"]
                  + env["lds29"] + env["lqds"]/2),

        env.place(env.new("qds19.ds_1.1", "qds19"), anchor="center",
                  at=env["ld"] + env["lds21"] + env["lqds"]
                  + env["lds22"] + env["lqds"]
                  + env["lds23"] + env["lqds"]
                  + env["lds24"] + env["ld"]
                  + env["lds25"] + env["lqds"]
                  + env["lds26"] + env["lqds"]
                  + env["lds27"] + env["lqds"]
                  + env["lds28"] + env["lqds"]
                  + env["lds29"] + env["lqds"]
                  + env["lds110"] + env["lqds"]/2),

        env.place(env.new("dend.ds_1.1", "dend"), anchor="center",
                  at=env["ld"] + env["lds21"] + env["lqds"]
                  + env["lds22"] + env["lqds"]
                  + env["lds23"] + env["lqds"]
                  + env["lds24"] + env["ld"]
                  + env["lds25"] + env["lqds"]
                  + env["lds26"] + env["lqds"]
                  + env["lds27"] + env["lqds"]
                  + env["lds28"] + env["lqds"]
                  + env["lds29"] + env["lqds"]
                  + env["lds110"] + env["lqds"]
                  + env["lds111"]/2),
    ],
)

# ds_2
env.new_line(
    name="ds_2",
    length=env["lds"],
    components=[
        env.place(env.new("dip.ds_2.1", "dip"), anchor="center", at=env["ld"]/2),

        env.place(env.new("sx2.ds_2.1", "sx2"), anchor="center",
                  at=env["ld"] + (env["lds21"]-env["lls"])/2 + env["lls"]/2),

        env.place(env.new("qds11.ds_2.1", "qds11"), anchor="center",
                  at=env["ld"] + env["lds21"] + env["lqds"]/2),

        env.place(env.new("qds12.ds_2.1", "qds12"), anchor="center",
                  at=env["ld"] + env["lds21"] + env["lqds"]
                  + env["lds22"] + env["lqds"]/2),

        env.place(env.new("sy2.ds_2.1", "sy2"), anchor="center",
                  at=env["ld"] + env["lds21"] + env["lqds"]
                  + env["lds22"] + env["lqds"]
                  + (env["lds23"]-env["lls"])/2 + env["lls"]/2),

        env.place(env.new("qds13.ds_2.1", "qds13"), anchor="center",
                  at=env["ld"] + env["lds21"] + env["lqds"]
                  + env["lds22"] + env["lqds"]
                  + env["lds23"] + env["lqds"]/2),

        env.place(env.new("dip.ds_2.2", "dip"), anchor="center",
                  at=env["ld"] + env["lds21"] + env["lqds"]
                  + env["lds22"] + env["lqds"]
                  + env["lds23"] + env["lqds"]
                  + env["lds24"] + env["ld"]/2),

        env.place(env.new("qds14.ds_2.1", "qds14"), anchor="center",
                  at=env["ld"] + env["lds21"] + env["lqds"]
                  + env["lds22"] + env["lqds"]
                  + env["lds23"] + env["lqds"]
                  + env["lds24"] + env["ld"]
                  + env["lds25"] + env["lqds"]/2),

        env.place(env.new("qds15.ds_2.1", "qds15"), anchor="center",
                  at=env["ld"] + env["lds21"] + env["lqds"]
                  + env["lds22"] + env["lqds"]
                  + env["lds23"] + env["lqds"]
                  + env["lds24"] + env["ld"]
                  + env["lds25"] + env["lqds"]
                  + env["lds26"] + env["lqds"]/2),

        env.place(env.new("qds16.ds_2.1", "qds16"), anchor="center",
                  at=env["ld"] + env["lds21"] + env["lqds"]
                  + env["lds22"] + env["lqds"]
                  + env["lds23"] + env["lqds"]
                  + env["lds24"] + env["ld"]
                  + env["lds25"] + env["lqds"]
                  + env["lds26"] + env["lqds"]
                  + env["lds27"] + env["lqds"]/2),

        env.place(env.new("qds17.ds_2.1", "qds17"), anchor="center",
                  at=env["ld"] + env["lds21"] + env["lqds"]
                  + env["lds22"] + env["lqds"]
                  + env["lds23"] + env["lqds"]
                  + env["lds24"] + env["ld"]
                  + env["lds25"] + env["lqds"]
                  + env["lds26"] + env["lqds"]
                  + env["lds27"] + env["lqds"]
                  + env["lds28"] + env["lqds"]/2),

        env.place(env.new("qds18.ds_2.1", "qds18"), anchor="center",
                  at=env["ld"] + env["lds21"] + env["lqds"]
                  + env["lds22"] + env["lqds"]
                  + env["lds23"] + env["lqds"]
                  + env["lds24"] + env["ld"]
                  + env["lds25"] + env["lqds"]
                  + env["lds26"] + env["lqds"]
                  + env["lds27"] + env["lqds"]
                  + env["lds28"] + env["lqds"]
                  + env["lds29"] + env["lqds"]/2),

        env.place(env.new("qds19.ds_2.1", "qds19"), anchor="center",
                  at=env["ld"] + env["lds21"] + env["lqds"]
                  + env["lds22"] + env["lqds"]
                  + env["lds23"] + env["lqds"]
                  + env["lds24"] + env["ld"]
                  + env["lds25"] + env["lqds"]
                  + env["lds26"] + env["lqds"]
                  + env["lds27"] + env["lqds"]
                  + env["lds28"] + env["lqds"]
                  + env["lds29"] + env["lqds"]
                  + env["lds110"] + env["lqds"]/2),

        env.place(env.new("dend.ds_2.1", "dend"), anchor="center",
                  at=env["ld"] + env["lds21"] + env["lqds"]
                  + env["lds22"] + env["lqds"]
                  + env["lds23"] + env["lqds"]
                  + env["lds24"] + env["ld"]
                  + env["lds25"] + env["lqds"]
                  + env["lds26"] + env["lqds"]
                  + env["lds27"] + env["lqds"]
                  + env["lds28"] + env["lqds"]
                  + env["lds29"] + env["lqds"]
                  + env["lds110"] + env["lqds"]
                  + env["lds111"]/2),
    ],
)
# rf cavity for ds
env["lrf"]  = 1e-5 # [m] ### thin cavity
env.new("rf", xt.Cavity, length=env["lrf"], voltage=10e6, frequency=0.0, lag=0, force=True, harmonic=2596/2)  # 10 mev

# mds total length
env["lds2"] = (
    env["lq1w"]
  + env["lds19"] + env["lqds"]
  + env["lds18"] + env["lqds"]
  + env["lds27"] + env["lqds"]
  + env["lds26"] + env["lqds"]
  + env["lds25"]
  + env["ld"]
  + env["lds24"] + env["lqds"]
  + env["lds23"] + env["lqds"]
  + env["lds22"] + env["lqds"]
  + env["lds21"]
)

# ----------mds build----------
# define end drift
env.new("dend2", xt.Drift, length=(env["lds21"]-env['lls'])/2) #, force=True)

# mds_1
env.new_line(
    name="mds_1",
    length=env["lds2"],
    components=[

        env.place(env.new("qfw.mds_1.1", "qfw"),
            anchor="center",
            at=env["lq1w"]/2),

        env.place(env.new("qds27.mds_1.1", "qds27"),
            anchor="center",
            at=env["lq1w"]
              + env["lds19"] + env["lqds"]/2),

        env.place(env.new("qds26.mds_1.1", "qds26"),
            anchor="center",
            at=env["lq1w"]
              + env["lds19"] + env["lqds"]
              + env["lds18"] + env["lqds"]/2),

        env.place(env.new("qds25.mds_1.1", "qds25"),
            anchor="center",
            at=env["lq1w"]
              + env["lds19"] + env["lqds"]
              + env["lds18"] + env["lqds"]
              + env["lds27"] + env["lqds"]/2),

        env.place(env.new("qds24.mds_1.1", "qds24"),
            anchor="center",
            at=env["lq1w"]
              + env["lds19"] + env["lqds"]
              + env["lds18"] + env["lqds"]
              + env["lds27"] + env["lqds"]
              + env["lds26"] + env["lqds"]/2),

        env.place(env.new("dip.mds_1.1", "dip"),
            anchor="center",
            at=env["lq1w"]
              + env["lds19"] + env["lqds"]
              + env["lds18"] + env["lqds"]
              + env["lds27"] + env["lqds"]
              + env["lds26"] + env["lqds"]
              + env["lds25"] + env["ld"]/2),

        env.place(env.new("qds23.mds_1.1", "qds23"),
            anchor="center",
            at=env["lq1w"]
              + env["lds19"] + env["lqds"]
              + env["lds18"] + env["lqds"]
              + env["lds27"] + env["lqds"]
              + env["lds26"] + env["lqds"]
              + env["lds25"] + env["ld"]
              + env["lds24"] + env["lqds"]/2),

        env.place(env.new("sy2.mds_1.1", "sy2"),
            anchor="center",
            at=env["lq1w"]
              + env["lds19"] + env["lqds"]
              + env["lds18"] + env["lqds"]
              + env["lds27"] + env["lqds"]
              + env["lds26"] + env["lqds"]
              + env["lds25"] + env["ld"]
              + env["lds24"] + env["lqds"]
              + (env["lds23"] - env["lls"])/2 + env["lls"]/2),

        env.place(env.new("qds22.mds_1.1", "qds22"),
            anchor="center",
            at=env["lq1w"]
              + env["lds19"] + env["lqds"]
              + env["lds18"] + env["lqds"]
              + env["lds27"] + env["lqds"]
              + env["lds26"] + env["lqds"]
              + env["lds25"] + env["ld"]
              + env["lds24"] + env["lqds"]
              + env["lds23"] + env["lqds"]/2),

        env.place(env.new("qds21.mds_1.1", "qds21"),
            anchor="center",
            at=env["lq1w"]
              + env["lds19"] + env["lqds"]
              + env["lds18"] + env["lqds"]
              + env["lds27"] + env["lqds"]
              + env["lds26"] + env["lqds"]
              + env["lds25"] + env["ld"]
              + env["lds24"] + env["lqds"]
              + env["lds23"] + env["lqds"]
              + env["lds22"] + env["lqds"]/2),

        env.place(env.new("sx2.mds_1.1", "sx2"),
            anchor="center",
            at=env["lq1w"]
              + env["lds19"] + env["lqds"]
              + env["lds18"] + env["lqds"]
              + env["lds27"] + env["lqds"]
              + env["lds26"] + env["lqds"]
              + env["lds25"] + env["ld"]
              + env["lds24"] + env["lqds"]
              + env["lds23"] + env["lqds"]
              + env["lds22"] + env["lqds"]
              + (env["lds21"] - env["lls"])/2 + env["lls"]/2),

        env.place(env.new("dend2.mds_1.1", "dend2"),
            anchor="start",
            at=env["lq1w"]
              + env["lds19"] + env["lqds"]
              + env["lds18"] + env["lqds"]
              + env["lds27"] + env["lqds"]
              + env["lds26"] + env["lqds"]
              + env["lds25"] + env["ld"]
              + env["lds24"] + env["lqds"]
              + env["lds23"] + env["lqds"]
              + env["lds22"] + env["lqds"]
              + (env["lds21"] + env["lls"])/2),
    ],
)

# mds_2
env.new_line(
    name="mds_2",
    length=env["lds2"],
    components=[

        env.place(env.new("qfw.mds_2.1", "qfw"),
            anchor="center",
            at=env["lq1w"]/2),

        env.place(env.new("qds27.mds_2.1", "qds27"),
            anchor="center",
            at=env["lq1w"]
              + env["lds19"] + env["lqds"]/2),

        env.place(env.new("qds26.mds_2.1", "qds26"),
            anchor="center",
            at=env["lq1w"]
              + env["lds19"] + env["lqds"]
              + env["lds18"] + env["lqds"]/2),

        env.place(env.new("qds25.mds_2.1", "qds25"),
            anchor="center",
            at=env["lq1w"]
              + env["lds19"] + env["lqds"]
              + env["lds18"] + env["lqds"]
              + env["lds27"] + env["lqds"]/2),

        env.place(env.new("qds24.mds_2.1", "qds24"),
            anchor="center",
            at=env["lq1w"]
              + env["lds19"] + env["lqds"]
              + env["lds18"] + env["lqds"]
              + env["lds27"] + env["lqds"]
              + env["lds26"] + env["lqds"]/2),

        env.place(env.new("dip.mds_2.1", "dip"),
            anchor="center",
            at=env["lq1w"]
              + env["lds19"] + env["lqds"]
              + env["lds18"] + env["lqds"]
              + env["lds27"] + env["lqds"]
              + env["lds26"] + env["lqds"]
              + env["lds25"] + env["ld"]/2),

        env.place(env.new("qds23.mds_2.1", "qds23"),
            anchor="center",
            at=env["lq1w"]
              + env["lds19"] + env["lqds"]
              + env["lds18"] + env["lqds"]
              + env["lds27"] + env["lqds"]
              + env["lds26"] + env["lqds"]
              + env["lds25"] + env["ld"]
              + env["lds24"] + env["lqds"]/2),

        env.place(env.new("sy2.mds_2.1", "sy2"),
            anchor="center",
            at=env["lq1w"]
              + env["lds19"] + env["lqds"]
              + env["lds18"] + env["lqds"]
              + env["lds27"] + env["lqds"]
              + env["lds26"] + env["lqds"]
              + env["lds25"] + env["ld"]
              + env["lds24"] + env["lqds"]
              + (env["lds23"] - env["lls"])/2 + env["lls"]/2),

        env.place(env.new("qds22.mds_2.1", "qds22"),
            anchor="center",
            at=env["lq1w"]
              + env["lds19"] + env["lqds"]
              + env["lds18"] + env["lqds"]
              + env["lds27"] + env["lqds"]
              + env["lds26"] + env["lqds"]
              + env["lds25"] + env["ld"]
              + env["lds24"] + env["lqds"]
              + env["lds23"] + env["lqds"]/2),

        env.place(env.new("qds21.mds_2.1", "qds21"),
            anchor="center",
            at=env["lq1w"]
              + env["lds19"] + env["lqds"]
              + env["lds18"] + env["lqds"]
              + env["lds27"] + env["lqds"]
              + env["lds26"] + env["lqds"]
              + env["lds25"] + env["ld"]
              + env["lds24"] + env["lqds"]
              + env["lds23"] + env["lqds"]
              + env["lds22"] + env["lqds"]/2),

        env.place(env.new("sx2.mds_2.1", "sx2"),
            anchor="center",
            at=env["lq1w"]
              + env["lds19"] + env["lqds"]
              + env["lds18"] + env["lqds"]
              + env["lds27"] + env["lqds"]
              + env["lds26"] + env["lqds"]
              + env["lds25"] + env["ld"]
              + env["lds24"] + env["lqds"]
              + env["lds23"] + env["lqds"]
              + env["lds22"] + env["lqds"]
              + (env["lds21"] - env["lls"])/2 + env["lls"]/2),

        env.place(env.new("dend2.mds_2.1", "dend2"),
            anchor="start",
            at=env["lq1w"]
              + env["lds19"] + env["lqds"]
              + env["lds18"] + env["lqds"]
              + env["lds27"] + env["lqds"]
              + env["lds26"] + env["lqds"]
              + env["lds25"] + env["ld"]
              + env["lds24"] + env["lqds"]
              + env["lds23"] + env["lqds"]
              + env["lds22"] + env["lqds"]
              + (env["lds21"] + env["lls"])/2),
    ],
)

 # ARCS
cells = []
for ii in range(env["ntme"]):     # order TME cells inside ring
    new_cell = env['tme'].clone(suffix=f'tme_{ii+1}')
    cells.append(new_cell)

env.new_line(
    name="ring",
    components=cells)

ring = env['ring'].copy(shallow=True)

tt = ring.get_table()     # create copy of ring to put markers in

tt_bend = tt.rows.match(element_type='Bend')

insertions = []        # insert markers at dip centers for matching
for nn in tt_bend.name:
    nn_marker = nn + '.center'
    env.new(nn_marker, xt.Marker)
    insertions.append(env.place(nn_marker, at=0, from_=nn)) #, from_anchor='center'))

#symmetric tme cell
ring.insert(insertions)
env["tme_cell_s"] = ring.select('dip.tme_8.center', 'dip.tme_9.center') # match this: "tme_cell_s"

env["arc1"] = ring.select(    # divide ring in arcs (shallow copies with whole elements)
    start="dip.tme_1_entry",  # these lines have the same knobs as ring: check with: env.info("kqfa", limit=None)
    end="dip.tme_17_entry",
)

env["arc2"] = ring.select(
    start="dip.tme_17_entry",
    end='_end_point',
)

# STRAIGHTS
cells = []
for ii in range(env["nfodow"]):
    new_cell = env['fodow'].clone(suffix=f'fodow_{ii+1}')
    cells.append(new_cell)

env.new_line(
    name="straight",
    components=cells)

straight = env['straight'].copy(shallow=True)

tt = straight.get_table()

tt_quad = tt.rows.match(element_type='Quadrupole')

insertions = []
for nn in tt_quad.name:
    nn_marker = nn + '.center'
    env.new(nn_marker, xt.Marker)
    insertions.append(env.place(nn_marker, at=0, from_=nn)) #, from_anchor='center'))

# symmetric fodow cell
straight.insert(insertions)
env["fodow_cell_s"] = straight.select('qfw.fodow_3.center', 'qfw.fodow_4.center') # match this symmetric cell

env["ss1"] = straight.select(
    start="qfw.fodow_1_entry",
    end="qfw.fodow_10_entry",
)

env["ss2"] = straight.select(
    start="qfw.fodow_10_entry",
    end='_end_point',
)

# DS #
#ss2 = env['ss2'].copy(shallow=True)
#ds_1 = env['ds_1'].copy(shallow=True)
#ds_2 = env['ds_2'].copy(shallow=True)
#arc1 = env['arc1'].copy(shallow=True)

con_ds = env["ds_1"].copy(shallow=True) + env["ss2"].copy(shallow=True)

env.new("dip.ds_1.1.center", xt.Marker)
con_ds.insert(
    env.place("dip.ds_1.1.center", at=0, from_="dip.ds_1.1")
)

env["ds_match"] = con_ds.select(
    start="dip.ds_1.1.center",
    end="qfw.fodow_10.center",
) # match ds starting from middle of arc's dip and ending in middle of straight's quad

# MDS #

con_mds = env["mds_1"].copy(shallow=True) + env["arc1"].copy(shallow=True)

env.new("qfw.mds_1.1.center", xt.Marker)
con_mds.insert(
    env.place("qfw.mds_1.1.center", at=0, from_="qfw.mds_1.1")
)

env["mds_match"] = con_mds.select(
    start="qfw.mds_1.1.center",
    end="dip.tme_1.center",
) # match mds starting from middle of straight's quad and ending in middle of arc's dip

#########################################
# pdr build
#########################################

env["pdr"] = (  # section_1,2 have same knobs
    env["ss1"]
    + env["mds_1"]
    + env["arc1"]
    + env["ds_1"]
    + env["ss2"]
    + env["mds_2"]
    + env["arc2"]
    + env["ds_2"]
)

env.vars.default_to_zero = False
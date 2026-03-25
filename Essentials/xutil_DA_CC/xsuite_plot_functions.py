#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 12:03:48 2021

@author: kyriacos
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from matplotlib.patches import Ellipse
import matplotlib.colors
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
import numpy as np
import scipy as sp
import scipy.constants as sp_co
import scipy.optimize as opt
from scipy.special import gamma
from scipy.optimize import curve_fit
from scipy.integrate import dblquad
from scipy.stats import multivariate_normal
#import machine_beam_parameters as my_mbp
from cpymad.madx import Madx
from pymadng import MAD
import xtrack as xt
import xpart as xp
import xobjects as xo
import xplt
import json
import pandas as pd
#import resonance_plot_v1 as rp

#import xsuite_functions as my_xf



def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def categorical_cmap(nc, nsc, cmap="tab20", continuous=False):
    if nc > plt.get_cmap(cmap).N:
        raise ValueError("Too many categories for colormap.")
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0,1,nc))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
    cols = np.zeros((nc*nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = matplotlib.colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv,nsc).reshape(nsc,3)
        arhsv[:,1] = np.linspace(chsv[1],0.25,nsc)
        arhsv[:,2] = np.linspace(chsv[2],1,nsc)
        rgb = matplotlib.colors.hsv_to_rgb(arhsv)
        cols[i*nsc:(i+1)*nsc,:] = rgb       
    cmap = matplotlib.colors.ListedColormap(cols)
    return cmap


def sketch_lattice (survey, line, labels=None):
    
    xplt.FloorPlot(survey, line, labels=labels) 
    plt.legend(fontsize='small', loc='upper left')
        
    return


def beta_dispertion_vs_s (twiss, ip_names = None):

    if ip_names == None:
        ip_names = twiss.rows['ip.*'].name

    plt.figure()
    ax = plt.subplot(2, 1, 1)
    plt.plot(twiss.s, np.sqrt(twiss.bety), '-', color='y', label='y')
    plt.plot(twiss.s, np.sqrt(twiss.betx), '-', color='crimson', label='x')
    for ii in ip_names:
        plt.axvline(x = twiss.rows[ii].s, linestyle = '--', color = 'black')
        plt.text(twiss.rows[ii].s, 0.8*ax.get_ylim()[1], ii, horizontalalignment='center',
        verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=1))
    plt.ylabel(r'$\sqrt{\beta_{x,y}}$ [$\sqrt{m}$]')
    plt.legend(fontsize='small')
    plt.grid()
    #plt.tight_layout()

    plt.subplot(2, 1, 2, sharex=ax)
    plt.plot(twiss.s, my_mbp.q_norm(twiss.dy, twiss.bety), '-', color='y', label='y')
    plt.plot(twiss.s, my_mbp.q_norm(twiss.dx, twiss.betx), '-', color='crimson', label='x')
    for ii in ip_names:
        plt.axvline(x = twiss.rows[ii].s, linestyle = '--', color = 'black')
    plt.ylabel(r'$\hat{D}_{x,y}$ [$\sqrt{m}$]')
    plt.xlabel('s [m]')
    plt.legend(fontsize='small', loc='best')
    plt.grid()
    plt.tight_layout()

    return


def beta_phase_advance_vs_s (twiss, ip_names = None):

    if ip_names == None:
        ip_names = twiss.rows['ip.*'].name

    plt.figure()
    ax = plt.subplot(2, 1, 1)
    plt.plot(twiss.s, np.sqrt(twiss.bety), '-', color='y', label='y')
    plt.plot(twiss.s, np.sqrt(twiss.betx), '-', color='crimson', label='x')
    for ii in ip_names:
        plt.axvline(x = twiss.rows[ii].s, linestyle = '--', color = 'black')
        plt.text(twiss.rows[ii].s, 0.8*ax.get_ylim()[1], ii, horizontalalignment='center',
        verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=1))
    plt.ylabel(r'$\sqrt{\beta_{x,y}}$ [$\sqrt{m}$]')
    plt.legend(fontsize='small')
    plt.grid()
    #plt.tight_layout()

    plt.subplot(2, 1, 2, sharex=ax)
    plt.plot(twiss.s, twiss.muy, '-', color='y', label='y')
    plt.plot(twiss.s, twiss.mux, '-', color='crimson', label='x')
    for ii in ip_names:
        plt.axvline(x = twiss.rows[ii].s, linestyle = '--', color = 'black')
    plt.ylabel(r'$\mu_{x,y}$ [$2\pi$]')
    plt.xlabel('s [m]')
    plt.legend(fontsize='small', loc='best')
    plt.grid()
    plt.tight_layout()

    return


def beta_chromatic_w_vs_s (twiss, ip_names = None):

    if ip_names == None:
        ip_names = twiss.rows['ip.*'].name

    plt.figure()
    ax = plt.subplot(2, 1, 1)
    plt.plot(twiss.s, np.sqrt(twiss.bety), '-', color='y', label='y')
    plt.plot(twiss.s, np.sqrt(twiss.betx), '-', color='crimson', label='x')
    for ii in ip_names:
        plt.axvline(x = twiss.rows[ii].s, linestyle = '--', color = 'black')
        plt.text(twiss.rows[ii].s, 0.8*ax.get_ylim()[1], ii, horizontalalignment='center',
        verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=1))
    plt.ylabel(r'$\sqrt{\beta_{x,y}}$ [$\sqrt{m}$]')
    plt.legend(fontsize='small')
    plt.grid()
    #plt.tight_layout()

    plt.subplot(2, 1, 2, sharex=ax)
    plt.plot(twiss.s, twiss.wy_chrom, '-', color='y', label='y')
    plt.plot(twiss.s, twiss.wx_chrom, '-', color='crimson', label='x')
    for ii in ip_names:
        plt.axvline(x = twiss.rows[ii].s, linestyle = '--', color = 'black')
    plt.ylabel(r'$W_{x,y}$')
    plt.xlabel('s [m]')
    plt.legend(fontsize='small', loc='best')
    plt.grid()
    plt.tight_layout()

    return


def chromatic_B_A_vs_s (twiss, ip_names = None):

    if ip_names == None:
        ip_names = twiss.rows['ip.*'].name

    plt.figure()
    ax = plt.subplot(2, 1, 1)
    plt.plot(twiss.s, twiss.by_chrom, '-', color='y', label='y')
    plt.plot(twiss.s, twiss.bx_chrom, '-', color='crimson', label='x')
    for ii in ip_names:
        plt.axvline(x = twiss.rows[ii].s, linestyle = '--', color = 'black')
        plt.text(twiss.rows[ii].s, 0.8*ax.get_ylim()[1], ii, horizontalalignment='center',
        verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=1))
    plt.ylabel(r'$B_{x,y}$')
    plt.legend(fontsize='small')
    plt.grid()
    #plt.tight_layout()

    plt.subplot(2, 1, 2, sharex=ax)
    plt.plot(twiss.s, twiss.ay_chrom, '-', color='y', label='y')
    plt.plot(twiss.s, twiss.ax_chrom, '-', color='crimson', label='x')
    for ii in ip_names:
        plt.axvline(x = twiss.rows[ii].s, linestyle = '--', color = 'black')
    plt.ylabel(r'$A_{x,y}$')
    plt.xlabel('s [m]')
    plt.legend(fontsize='small', loc='best')
    plt.grid()
    plt.tight_layout()

    return


def closed_orbit_phase_advance_vs_s (twiss, ip_names = None):

    if ip_names == None:
        ip_names = twiss.rows['ip.*'].name

    plt.figure()
    ax = plt.subplot(2, 1, 1)
    plt.plot(twiss.s, twiss.y, '-', color='y', label='y')
    plt.plot(twiss.s, twiss.x, '-', color='crimson', label='x')
    for ii in ip_names:
        plt.axvline(x = twiss.rows[ii].s, linestyle = '--', color = 'black')
        plt.text(twiss.rows[ii].s, 0.8*ax.get_ylim()[1], ii, horizontalalignment='center',
        verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=1))
    plt.ylabel(r'$x_{co},y_{co}$ [m]')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.legend(fontsize='small')
    plt.grid()
    #plt.tight_layout()

    plt.subplot(2, 1, 2, sharex=ax)
    plt.plot(twiss.s, twiss.muy, '-', color='y', label='y')
    plt.plot(twiss.s, twiss.mux, '-', color='crimson', label='x')
    for ii in ip_names:
        plt.axvline(x = twiss.rows[ii].s, linestyle = '--', color = 'black')
    plt.ylabel(r'$\mu_{x,y}$ [$2\pi$]')
    plt.xlabel('s [m]')
    plt.legend(fontsize='small', loc='best')
    plt.grid()
    plt.tight_layout()

    return


def closed_orbit_dispertion_vs_s (twiss, ip_names = None):

    if ip_names == None:
        ip_names = twiss.rows['ip.*'].name

    plt.figure()
    ax = plt.subplot(2, 1, 1)
    plt.plot(twiss.s, twiss.y, '-', color='y', label='y')
    plt.plot(twiss.s, twiss.x, '-', color='crimson', label='x')
    for ii in ip_names:
        plt.axvline(x = twiss.rows[ii].s, linestyle = '--', color = 'black')
        plt.text(twiss.rows[ii].s, 0.8*ax.get_ylim()[1], ii, horizontalalignment='center',
        verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=1))
    plt.ylabel(r'$x_{co},y_{co}$ [m]')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.legend(fontsize='small')
    plt.grid()
    #plt.tight_layout()

    plt.subplot(2, 1, 2, sharex=ax)
    plt.plot(twiss.s, twiss.dy, '-', color='y', label='y')
    plt.plot(twiss.s, twiss.dx, '-', color='crimson', label='x')
    for ii in ip_names:
        plt.axvline(x = twiss.rows[ii].s, linestyle = '--', color = 'black')
    plt.ylabel(r'$D_{x,y}$ [m]')
    plt.xlabel('s [m]')
    plt.legend(fontsize='small', loc='best')
    plt.grid()
    plt.tight_layout()

    return


def beta_beating_vs_s(reference_twiss, studied_twiss):

    ## plot beta-beating
    fig = plt.subplots()
    beta_beat_x = (studied_twiss.betx-reference_twiss.betx)/reference_twiss.betx
    beta_beat_y = (studied_twiss.bety-reference_twiss.bety)/reference_twiss.bety
    if np.max(np.abs([np.max(beta_beat_x),np.min(beta_beat_x)])) > np.max(np.abs([np.max(beta_beat_y),np.min(beta_beat_y)])):
        plt.plot(reference_twiss.s, beta_beat_x, '-', color='crimson', label='x', alpha=0.7)
        plt.plot(reference_twiss.s, beta_beat_y, '-', color='y', label='y', alpha=0.7)
    else:
        plt.plot(reference_twiss.s, beta_beat_y, '-', color='y', label='y', alpha=0.7)
        plt.plot(reference_twiss.s, beta_beat_x, '-', color='crimson', label='x', alpha=0.7)
    plt.xlabel('s [m]')
    plt.ylabel(r'$\dfrac{\beta_{x_{er},y_{er}}-\beta_{x,y}}{\beta_{x,y}}$')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.legend(fontsize='small', loc='best')
    plt.grid()
    plt.tight_layout()

    return


def normalized_closed_orbit_diviation_vs_s(reference_twiss, studied_twiss, norm_emitt_x, norm_emitt_y, bunch_length):

    ## normalized closed orbit diviation
    beam_sizes_reference = reference_twiss.get_beam_covariance(nemitt_x=norm_emitt_x, nemitt_y=norm_emitt_y,
                                    gemitt_zeta=bunch_length**2/reference_twiss.bets0)
    beam_sizes_studied = studied_twiss.get_beam_covariance(nemitt_x=norm_emitt_x, nemitt_y=norm_emitt_y,
                                    gemitt_zeta=bunch_length**2/studied_twiss.bets0)
    
    fig = plt.subplots()
    norm_orbit_div_x = studied_twiss.x/beam_sizes_studied.sigma_x - reference_twiss.x/beam_sizes_reference.sigma_x
    norm_orbit_div_y = studied_twiss.y/beam_sizes_studied.sigma_y - reference_twiss.y/beam_sizes_reference.sigma_y
    if np.max(np.abs([np.max(norm_orbit_div_x),np.min(norm_orbit_div_x)])) > np.max(np.abs([np.max(norm_orbit_div_y),np.min(norm_orbit_div_y)])):
        plt.plot(reference_twiss.s, norm_orbit_div_x, '-', color='crimson', label=r'x', alpha=0.7)
        plt.plot(reference_twiss.s, norm_orbit_div_y, '-', color='y', label='y', alpha=0.7)
    else:
        plt.plot(reference_twiss.s, norm_orbit_div_y, '-', color='y', label='y', alpha=0.7)
        plt.plot(reference_twiss.s, norm_orbit_div_x, '-', color='crimson', label=r'x', alpha=0.7)
    plt.xlabel('s [m]')
    plt.ylabel(r'$x_{er}-x;y_{er}-y$ [$\sigma_{x,y}$]')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.legend(fontsize='small', loc='best')
    plt.grid()
    plt.tight_layout()

    return


def closed_orbit_diviation_vs_s(reference_twiss, studied_twiss, spatial_shifts=None):

    ## closed orbit diviation
    orbit_div_x = studied_twiss.x-reference_twiss.x
    orbit_div_y = studied_twiss.y-reference_twiss.y
    if isinstance(spatial_shifts, pd.DataFrame): 
        fig,ax = plt.subplots()
        axb = ax.twinx()
        if np.max(np.abs([np.max(orbit_div_x),np.min(orbit_div_x)])) > np.max(np.abs([np.max(orbit_div_y),np.min(orbit_div_y)])):
            p1=axb.plot(spatial_shifts.s, spatial_shifts.shift_x, '.', color='mediumpurple', label=r'$d_x$', alpha=0.7)
            p2=axb.plot(spatial_shifts.s, spatial_shifts.shift_y, '.', color='seagreen', label=r'$d_y$', alpha=0.7)
            p3=ax.plot(reference_twiss.s, orbit_div_x, '-', color='crimson', label=r'x', alpha=0.7)
            p4=ax.plot(reference_twiss.s, orbit_div_y, '-', color='y', label='y', alpha=0.7)
        else:
            p1=axb.plot(spatial_shifts.s, spatial_shifts.shift_y, '.', color='seagreen', label=r'$d_y$', alpha=0.7)
            p2=axb.plot(spatial_shifts.s, spatial_shifts.shift_x, '.', color='mediumpurple', label=r'$d_x$', alpha=0.7)
            p3=ax.plot(reference_twiss.s, orbit_div_y, '-', color='y', label='y', alpha=0.7)
            p4=ax.plot(reference_twiss.s, orbit_div_x, '-', color='crimson', label=r'x', alpha=0.7)
        ax.set_xlabel('s [m]')
        ax.set_ylabel(r'$x_{er}-x~;~y_{er}-y$ [m]')
        axb.set_ylabel(r'$d_x~;~d_y$ [m]')
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        axb.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.set_zorder(axb.get_zorder()+1)
        ax.patch.set_visible(False)
        # added these three lines
        lns = p1+p2+p3+p4
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, fontsize='small', loc='best')
        ax.grid()
        plt.tight_layout()
    else:
        fig = plt.subplots()
        if np.max(np.abs([np.max(orbit_div_x),np.min(orbit_div_x)])) > np.max(np.abs([np.max(orbit_div_y),np.min(orbit_div_y)])):
            plt.plot(reference_twiss.s, orbit_div_x, '-', color='crimson', label=r'x', alpha=0.7)
            plt.plot(reference_twiss.s, orbit_div_y, '-', color='y', label='y', alpha=0.7)
        else:
            plt.plot(reference_twiss.s, orbit_div_y, '-', color='y', label='y', alpha=0.7)
            plt.plot(reference_twiss.s, orbit_div_x, '-', color='crimson', label=r'x', alpha=0.7)
        plt.xlabel('s [m]')
        plt.ylabel(r'$x_{er}-x~;~y_{er}-y$ [m]')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.legend(fontsize='small', loc='best')
        plt.grid()
        plt.tight_layout()

    return


def closed_orbit_diviation_vs_s_v2(reference_twiss, studied_twiss, spatial_shifts=None):

    ## closed orbit diviation
    if isinstance(spatial_shifts, pd.DataFrame):
        plt.figure()
        ax = plt.subplot(2, 1, 1)
        orbit_div_x = studied_twiss.x-reference_twiss.x
        orbit_div_y = studied_twiss.y-reference_twiss.y
        if np.max(np.abs([np.max(orbit_div_x),np.min(orbit_div_x)])) > np.max(np.abs([np.max(orbit_div_y),np.min(orbit_div_y)])):
            plt.plot(reference_twiss.s, orbit_div_x, '-', color='crimson', label=r'x', alpha=0.7)
            plt.plot(reference_twiss.s, orbit_div_y, '-', color='y', label='y', alpha=0.7)
        else:
            plt.plot(reference_twiss.s, orbit_div_y, '-', color='y', label='y', alpha=0.7)
            plt.plot(reference_twiss.s, orbit_div_x, '-', color='crimson', label=r'x', alpha=0.7)
        plt.ylabel(r'$x_{er}-x~;~y_{er}-y$ [m]')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.legend(fontsize='small', loc='best')
        plt.grid()

        plt.subplot(2, 1, 2, sharex=ax)
        plt.plot(spatial_shifts.s, spatial_shifts.shift_y, '.', color='y', label=r'y', alpha=0.5)
        plt.plot(spatial_shifts.s, spatial_shifts.shift_x, '.', color='crimson', label=r'x', alpha=0.5)
        plt.xlabel('s [m]')
        plt.ylabel(r'$d_x~;~d_y$ [m]')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.legend(fontsize='small', loc='best')
        plt.grid()

    else:
        fig = plt.subplots()
        orbit_div_x = studied_twiss.x-reference_twiss.x
        orbit_div_y = studied_twiss.y-reference_twiss.y
        if np.max(np.abs([np.max(orbit_div_x),np.min(orbit_div_x)])) > np.max(np.abs([np.max(orbit_div_y),np.min(orbit_div_y)])):
            plt.plot(reference_twiss.s, orbit_div_x, '-', color='crimson', label=r'x', alpha=0.7)
            plt.plot(reference_twiss.s, orbit_div_y, '-', color='y', label='y', alpha=0.7)
        else:
            plt.plot(reference_twiss.s, orbit_div_y, '-', color='y', label='y', alpha=0.7)
            plt.plot(reference_twiss.s, orbit_div_x, '-', color='crimson', label=r'x', alpha=0.7)
        plt.xlabel('s [m]')
        plt.ylabel(r'$x_{er}-x~;~y_{er}-y$ [m]')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.legend(fontsize='small', loc='best')
        plt.grid()
        plt.tight_layout()

    return


def normalized_dispertion_diviation_vs_s(reference_twiss, studied_twiss):

    ## plot normalized dispertion diviation
    fig = plt.subplots()
    norm_disp_div_x = my_mbp.q_norm(studied_twiss.dx, studied_twiss.betx)-my_mbp.q_norm(reference_twiss.dx, reference_twiss.betx)
    norm_disp_div_y = my_mbp.q_norm(studied_twiss.dy, studied_twiss.bety)-my_mbp.q_norm(reference_twiss.dy, reference_twiss.bety)
    if np.max(np.abs([np.max(norm_disp_div_x),np.min(norm_disp_div_x)])) > np.max(np.abs([np.max(norm_disp_div_y),np.min(norm_disp_div_y)])):
        plt.plot(reference_twiss.s, norm_disp_div_x, '-', color='crimson', label='x', alpha=0.7)
        plt.plot(reference_twiss.s, norm_disp_div_y, '-', color='y', label='y', alpha=0.7)
    else:
        plt.plot(reference_twiss.s, norm_disp_div_y, '-', color='y', label='y', alpha=0.7)
        plt.plot(reference_twiss.s, norm_disp_div_x, '-', color='crimson', label='x', alpha=0.7)
    plt.ylabel(r'$\hat{D}_{x_{er},y_{er}}-\hat{D}_{x,y}$ [$\sqrt{m}$]')
    plt.xlabel('s [m]')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.legend(fontsize='small', loc='best')
    plt.grid()
    plt.tight_layout()

    return


def dispertion_diviation_vs_s(reference_twiss, studied_twiss):

    ## plot dispertion diviation
    fig = plt.subplots()
    disp_div_x = studied_twiss.dx-reference_twiss.dx
    disp_div_y = studied_twiss.dy-reference_twiss.dy
    if np.max(np.abs([np.max(disp_div_x),np.min(disp_div_x)])) > np.max(np.abs([np.max(disp_div_y),np.min(disp_div_y)])):
        plt.plot(reference_twiss.s, disp_div_x, '-', color='crimson', label='x', alpha=0.7)
        plt.plot(reference_twiss.s, disp_div_y, '-', color='y', label='y', alpha=0.7)
    else:
        plt.plot(reference_twiss.s, disp_div_y, '-', color='y', label='y', alpha=0.7)
        plt.plot(reference_twiss.s, disp_div_x, '-', color='crimson', label='x', alpha=0.7)
    plt.ylabel(r'$D_{x_{er},y_{er}}-D_{x,y}$ [$m$]')
    plt.xlabel('s [m]')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.legend(fontsize='small', loc='best')
    plt.grid()
    plt.tight_layout()

    return


def tune_vs_delta (tune_delta_data, sigma_delta=None):

    # Assuming 'x' is your common x-axis column in df
    x = tune_delta_data['delta0']
    y1 = tune_delta_data['qx']
    y2 = tune_delta_data['qy']

    fig, ax1 = plt.subplots()

    # Plot df.Qx on the left y-axis
    lin1, = ax1.plot(x, y1, 'o-', color='crimson', alpha=0.7, label='Qx')
    ax1.set_xlabel(r'$\delta$ [%]')
    ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax1.set_ylabel('Qx') #, color='crimson'
    ax1.tick_params(axis='y') #, labelcolor='crimson'
    ax1.grid(color='gray', linestyle='--', linewidth=0.5)

    # Create a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    lin2, = ax2.plot(x, y2, 'o-', color='y', alpha=0.7, label='Qy')
    ax2.set_ylabel('Qy') #, color='y'
    ax2.tick_params(axis='y') #, labelcolor='y'
    ax2.grid(color='gray', linestyle=':', linewidth=0.5)

    # Add legends
    lins = [lin1, lin2]
    labels = [lin.get_label() for lin in lins]
    ax1.legend(lins, labels, loc="best")  # Combine both lines in a legend on the left

    # Add a secondary x-axis below the original, using rescaled x-values
    if sigma_delta is not None:
        ax3 = ax1.twiny()  # Create a new x-axis
        ax3.spines['bottom'].set_position(('outward', 40))  # Offset it below the main x-axis
        ax3.set_frame_on(True)
        ax3.patch.set_visible(False)
        ax3.xaxis.set_ticks_position('bottom')
        ax3.xaxis.set_label_position('bottom')
        ax3.spines['bottom'].set_visible(True)

        # Synchronize ticks with rescaled values on secondary axis
        ax3.set_xlim(ax1.get_xlim())  # Align the limits of the secondary x-axis with the main x-axis
        ax3.xaxis.set_major_locator(ax1.xaxis.get_major_locator())  # Use the same locator for consistent tick positions
        ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda val, pos: f'{val / sigma_delta:.1f}'))  # Rescale labels

        ax3.set_xlabel(r'$\delta$ [$\sigma$]')

    # Optionally, add legends or customize the plot
    fig.tight_layout()
    plt.show()

    return


def DA_vs_turns(particles, num_r_steps, num_theta_steps, x_norm, y_norm, delta_initial, delta_plots=False):

    if isinstance(particles, dict):
        max_turns = np.shape(particles['x'])[1]-1 # minus 1 for the initial condition
        part_at_turn = np.nanmax(particles['at_turn'],axis=1)
    else:
        max_turns = np.max(particles.filter(particles.at_element==0).at_turn) # normally I should pass the maximum number (n_turn) of asked turns
        part_at_turn = particles.at_turn

    if delta_plots and np.size(delta_initial) > 1:

        for ii in np.unique(delta_initial):
            delta_index = np.where(delta_initial==ii)[0]
            
            x_norm_1d = x_norm[delta_index]
            y_norm_1d = y_norm[delta_index]
            part_at_turn_1d = part_at_turn[delta_index]
            x_norm_2d = x_norm_1d.reshape(num_r_steps, num_theta_steps)
            y_norm_2d = y_norm_1d.reshape(num_r_steps, num_theta_steps)
            part_at_turn_2d = part_at_turn_1d.reshape(num_r_steps, num_theta_steps)
                        
            x_DA = np.full(num_theta_steps, np.nan)
            y_DA = np.full(num_theta_steps, np.nan)
            for jj in range(num_theta_steps):
                for ii in range(num_r_steps):
                    if part_at_turn_2d[ii,jj] != max_turns:
                        x_DA[jj] = x_norm_2d[ii,jj]
                        y_DA[jj] = y_norm_2d[ii,jj]
                        break

            min_DA = np.nanmin(np.round(np.sqrt(x_DA**2+y_DA**2),1)) 
            where_min_DA = np.where(np.round(np.sqrt(x_DA**2+y_DA**2),1) == min_DA)[0]
            
            # Plot DA using scatter and pcolormesh
            fig = plt.subplots()
            plt.scatter(x_norm_1d, y_norm_1d, c=part_at_turn_1d)
            plt.plot(x_DA, y_DA, '-', color='r', label='DA for $\delta$=%.1E'%(ii))
            plt.plot(x_DA[where_min_DA], y_DA[where_min_DA], 'o', color='r', label='DA$_{min}$=%.1f$\sigma$'%(min_DA))
            plt.xlabel(r'$\hat{x}$ [$\sqrt{\varepsilon_x}$]')
            plt.ylabel(r'$\hat{y}$ [$\sqrt{\varepsilon_y}$]')
            cb = plt.colorbar()
            cb.set_label('Lost at turn')
            plt.legend(fontsize='small', loc='best')

            fig = plt.subplots()
            plt.pcolormesh(x_norm_2d, y_norm_2d, part_at_turn_2d, shading='gouraud')
            plt.plot(x_DA, y_DA, '-', color='r', label='DA for $\delta$=%.1E'%(ii))
            plt.plot(x_DA[where_min_DA], y_DA[where_min_DA], 'o', color='r', label='DA$_{min}$=%.1f$\sigma$'%(min_DA))    
            plt.xlabel(r'$\hat{x}$ [$\sqrt{\varepsilon_x}$]')
            plt.ylabel(r'$\hat{y}$ [$\sqrt{\varepsilon_y}$]')
            ax = plt.colorbar()
            ax.set_label('Lost at turn')
            plt.legend(fontsize='small', loc='best')
    
    else:

        if not delta_plots and np.size(delta_initial) > 1:
            closest_to_zero_delta = delta_initial[(np.abs(delta_initial - 0)).argmin()]
            delta_index = np.where(delta_initial==closest_to_zero_delta)[0]
            x_norm_1d = x_norm[delta_index]
            y_norm_1d = y_norm[delta_index]
            part_at_turn_1d = part_at_turn[delta_index]
        else:
            x_norm_1d = x_norm
            y_norm_1d = y_norm      
            part_at_turn_1d = part_at_turn

        x_norm_2d = x_norm_1d.reshape(num_r_steps, num_theta_steps)
        y_norm_2d = y_norm_1d.reshape(num_r_steps, num_theta_steps)
        part_at_turn_2d = part_at_turn_1d.reshape(num_r_steps, num_theta_steps)
        x_DA = np.full(num_theta_steps, np.nan)
        y_DA = np.full(num_theta_steps, np.nan)
        for jj in range(num_theta_steps):
            for ii in range(num_r_steps):
                if part_at_turn_2d[ii,jj] != max_turns:
                    x_DA[jj] = x_norm_2d[ii,jj]
                    y_DA[jj] = y_norm_2d[ii,jj]
                    break

        min_DA = np.nanmin(np.round(np.sqrt(x_DA**2+y_DA**2),1)) 
        where_min_DA = np.where(np.round(np.sqrt(x_DA**2+y_DA**2),1) == min_DA)[0]
        
        # Plot DA using scatter and pcolormesh
        fig = plt.subplots()
        plt.scatter(x_norm_1d, y_norm_1d, c=part_at_turn_1d)
        plt.plot(x_DA, y_DA, '-', color='r', label='DA')
        plt.plot(x_DA[where_min_DA], y_DA[where_min_DA], 'o', color='r', label='DA$_{min}$=%.1f$\sigma$'%(min_DA))
        plt.xlabel(r'$\hat{x}$ [$\sqrt{\varepsilon_x}$]')
        plt.ylabel(r'$\hat{y}$ [$\sqrt{\varepsilon_y}$]')
        cb = plt.colorbar()
        cb.set_label('Lost at turn')
        plt.legend(fontsize='small', loc='best')

        fig = plt.subplots()
        plt.pcolormesh(x_norm_2d, y_norm_2d, part_at_turn_2d, shading='gouraud')
        plt.plot(x_DA, y_DA, '-', color='r', label='DA')
        plt.plot(x_DA[where_min_DA], y_DA[where_min_DA], 'o', color='r', label='DA$_{min}$=%.1f$\sigma$'%(min_DA))    
        plt.xlabel(r'$\hat{x}$ [$\sqrt{\varepsilon_x}$]')
        plt.ylabel(r'$\hat{y}$ [$\sqrt{\varepsilon_y}$]')
        ax = plt.colorbar()
        ax.set_label('Lost at turn')
        plt.legend(fontsize='small', loc='best')

    return (x_DA, y_DA, where_min_DA)


def MA_vs_turns(particles, num_r_steps, num_delta_steps, x_norm, y_norm, delta_initial):

    if isinstance(particles, dict):
        max_turns = np.shape(particles['x'])[1]-1 # minus 1 for the initial condition
        part_at_turn = np.nanmax(particles['at_turn'],axis=1)
    else:
        max_turns = np.max(particles.filter(particles.at_element==0).at_turn) # normally I should pass the maximum number (n_turn) of asked turns
        part_at_turn = particles.at_turn

    theta = np.max(np.unique(np.arctan2(y_norm,x_norm)))

    x_norm_2d = x_norm.reshape(num_delta_steps, num_r_steps)
    y_norm_2d = y_norm.reshape(num_delta_steps, num_r_steps)
    delta_norm_2d = delta_initial.reshape(num_delta_steps, num_r_steps)
    part_at_turn = part_at_turn
    part_at_turn_2d = part_at_turn.reshape(num_delta_steps, num_r_steps)
    x_MA = np.full(num_delta_steps, np.nan)
    y_MA = np.full(num_delta_steps, np.nan)
    delta_MA = np.full(num_delta_steps, np.nan)
    for jj in range(num_delta_steps):
        for ii in range(num_r_steps):
            if part_at_turn_2d[jj,ii] != max_turns:
                x_MA[jj] = x_norm_2d[jj,ii]
                y_MA[jj] = y_norm_2d[jj,ii]
                delta_MA[jj] = delta_norm_2d[jj,ii]
                break

    min_MA = np.nanmin(np.round(np.sqrt(x_MA**2+y_MA**2),1)) 
    where_min_MA = np.where(np.round(np.sqrt(x_MA**2+y_MA**2),1) == min_MA)[0]
    
    # Plot MA using scatter and pcolormesh
    fig = plt.subplots()
    plt.scatter(delta_initial*100, x_norm, c=part_at_turn)
    plt.plot(delta_MA*100, x_MA, '-', color='r', label='MA')
    plt.plot(delta_MA[where_min_MA]*100, x_MA[where_min_MA], 'o', color='r', label='MA$_{min}$=%.1f$\sigma$'%(min_MA))
    plt.xlabel(r'$\delta$ [%]')
    plt.ylabel(r'$\hat{x}$ [$\sqrt{\varepsilon_x}$], $\hat{y}$ [Tan(%.1f)$\sqrt{\varepsilon_y}$]'%(theta*180/np.pi))
    cb = plt.colorbar()
    cb.set_label('Lost at turn')
    plt.legend(fontsize='small', loc='best')

    fig = plt.subplots()
    plt.pcolormesh(delta_norm_2d*100, x_norm_2d, part_at_turn_2d, shading='gouraud')
    plt.plot(delta_MA*100, x_MA, '-', color='r', label='MA')
    plt.plot(delta_MA[where_min_MA]*100, x_MA[where_min_MA], 'o', color='r', label='MA$_{min}$=%.1f$\sigma$'%(min_MA))    
    plt.xlabel(r'$\delta$ [%]')
    plt.ylabel(r'$\hat{x}$ [$\sqrt{\varepsilon_x}$], $\hat{y}$ [Tan(%.1f)$\sqrt{\varepsilon_y}$]'%(theta*180/np.pi))
    ax = plt.colorbar()
    ax.set_label('Lost at turn')
    plt.legend(fontsize='small', loc='best')

    return (x_MA, delta_MA, where_min_MA)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def initial_conditions_2d_grid (x_axis, y_axis, x_axis_label, y_axis_label, q_normalized=True, delta_initial=None, theta=None):

    particle_id_cmap = truncate_colormap(plt.get_cmap('nipy_spectral'), 0.05, 0.95)
    
    if 'delta' in x_axis_label:

        # Plot initial conditions
        fig = plt.subplots()
        plt.scatter(x_axis*100, y_axis, c=np.arange(0,len(x_axis)), cmap=particle_id_cmap)
        plt.xlabel(r'$\delta$ [%]')
        if theta is not None:
            if q_normalized:
                plt.ylabel(r'$\hat{x}$ [$\sqrt{\varepsilon_x}$], $\hat{y}$ [Tan(%.1f)$\sqrt{\varepsilon_y}$]'%(theta*180/np.pi))
            else:
                plt.ylabel(r'x [m], y [Tan(%.1f) m]'%(theta*180/np.pi))
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        else:
            if q_normalized:
                plt.ylabel(r'$\hat{x}$ [$\sqrt{\varepsilon_x}$]')
            else:
                plt.ylabel(r'x [m]')
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        cb = plt.colorbar()
        cb.set_label('Particle ID')

    else:
        if delta_initial is not None and np.size(delta_initial) > 1:
            for ii in np.unique(delta_initial):
                delta_index = np.where(delta_initial==ii)[0]
                
                x_norm_1d = x_axis[delta_index]
                y_norm_1d = y_axis[delta_index]
                
                # Plot initial conditions
                fig = plt.subplots()
                plt.scatter(x_norm_1d, y_norm_1d, c=np.arange(0,len(x_norm_1d)), cmap=particle_id_cmap)
                if q_normalized:
                    plt.xlabel(fr'$\hat{{{x_axis_label}}}$ [$\sqrt{{\varepsilon_{x_axis_label}}}$]')
                    plt.ylabel(fr'$\hat{{{y_axis_label}}}$ [$\sqrt{{\varepsilon_{y_axis_label}}}$]')
                else:
                    plt.xlabel(fr'{x_axis_label} [m]')
                    plt.ylabel(fr'{y_axis_label} [m]')
                    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                plt.plot(np.nan, np.nan, '>', color='k', label='$\delta$=%.1E'%(ii))
                plt.legend(fontsize='small', loc='best')
                cb = plt.colorbar()
                cb.set_label('Particle ID')
        
        else:
            fig = plt.subplots()
            plt.scatter(x_axis, y_axis, c=np.arange(0,len(x_axis)), cmap=particle_id_cmap)
            if q_normalized:
                plt.xlabel(fr'$\hat{{{x_axis_label}}}$ [$\sqrt{{\varepsilon_{x_axis_label}}}$]')
                plt.ylabel(fr'$\hat{{{y_axis_label}}}$ [$\sqrt{{\varepsilon_{y_axis_label}}}$]')
            else:
                plt.xlabel(fr'{x_axis_label} [m]')
                plt.ylabel(fr'{y_axis_label} [m]')
                plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            if delta_initial is not None:
                plt.plot(np.nan, np.nan, '>', color='k', label='$\delta$=%.1E'%(delta_initial))
                plt.legend(fontsize='small', loc='best')
            cb = plt.colorbar()
            cb.set_label('Particle ID')

    return


def phase_space_distribution (x_cord, y_cord, x_axis_label=None, y_axis_label=None, particle_id_list=None, full_init_cond=None, apply_gaus_fit=False, apply_q_gaus_fit=False):

    particle_id_cmap = truncate_colormap(plt.get_cmap('nipy_spectral'), 0.05, 0.95)
    # Create colors and a ScalarMappable for colorbar
    if particle_id_list is not None and full_init_cond is not None:
        colors = [particle_id_cmap(i) for i in particle_id_list/(full_init_cond-1)]
    else:
        colors = [particle_id_cmap(i) for i in np.arange(len(x_cord))/(len(x_cord)-1)]

    if x_axis_label is not None and y_axis_label is not None:
        axis_label = [x_axis_label, y_axis_label]
    else:
        axis_label = ['x', '$p_x$']

    # Emittance calculation
    emit = np.sqrt(np.mean(x_cord**2)*np.mean(y_cord**2)-np.mean(x_cord*y_cord)**2)
        
    # Sample data
    x = x_cord
    y = y_cord

    # Calculate the area under the entire 2D distribution
    def area_calculation(x_mean, x_beta, x_q, y_mean, y_beta, y_q, num_sigma=1):

        # q-Gaussian standard deviation
        def q_gaussian_std (q, beta):
            if 2<=q<3:
                return np.nan
            elif 5/3<=q<2:
                return np.inf
            elif q<5/3:
                return np.sqrt(1/(beta*(5-3*q)))

        # Define the bounds of the integration region
        def integrand(x, y):
            return 1  # We're just integrating 1 to get the area of the projection

        def bounds_y(x):
            pdf_1sigma = np.nanmin([my_xf.q_gaussian(num_sigma*q_gaussian_std (x_q, x_beta), x_mean, x_beta, x_q)*my_xf.q_gaussian(0, y_mean, y_beta, y_q),
                                    my_xf.q_gaussian(0, x_mean, x_beta, x_q)*my_xf.q_gaussian(num_sigma*q_gaussian_std (y_q, y_beta), y_mean, y_beta, y_q)])
            pdf_core_x = np.sqrt(x_beta)*np.maximum((1+(x_q-1)*x_beta*(x-x_mean)**2),0)**(1/(1-x_q))

            if x_q<1 :
                pdf_norm_x = (2*np.sqrt(np.pi)*gamma(1/(1-x_q))/((3-x_q)*np.sqrt(1-x_q)*gamma((3-x_q)/(2-2*x_q))))
            elif x_q>1:
                pdf_norm_x = (np.sqrt(np.pi)*gamma((3-x_q)/(2-2*x_q))/(np.sqrt(x_q-1)*gamma(1/(x_q-1))))

            if y_q<1 :
                pdf_norm_y = (2*np.sqrt(np.pi)*gamma(1/(1-y_q))/((3-y_q)*np.sqrt(1-y_q)*gamma((3-y_q)/(2-2*y_q))))
            elif y_q>1:
                pdf_norm_y = (np.sqrt(np.pi)*gamma((3-y_q)/(2-2*y_q))/(np.sqrt(y_q-1)*gamma(1/(y_q-1))))

            bound_cor = np.maximum((pdf_1sigma*pdf_norm_x*pdf_norm_y/(pdf_core_x*np.sqrt(y_beta)))**(1-y_q),0)
            bound = np.sqrt((bound_cor-1)/((y_q-1)*y_beta))

            return -bound+y_mean, bound+y_mean

        # Define integration bounds
        x_min, x_max = x_mean - num_sigma * q_gaussian_std (x_q, x_beta), x_mean + num_sigma * q_gaussian_std (x_q, x_beta)
        y_min, y_max = y_mean - num_sigma * q_gaussian_std (y_q, y_beta), y_mean + num_sigma * q_gaussian_std (y_q, y_beta)
        
        # Perform numerical integration
        area, _ = dblquad(integrand, x_min, x_max, lambda x:bounds_y(x)[0],  lambda x:bounds_y(x)[1])
        
        return area

    # Create the figure and gridspec layout
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(4, 4, figure=fig)
    ax_main = fig.add_subplot(gs[1:4, 0:3])
    ax_xhist = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    ax_yhist = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

    # Scatter plot      
    ax_main.scatter(x, y, color=colors, alpha=0.5)

    # Calculate and plot contour lines for multiple standard deviations
    x_mean, y_mean = np.mean(x), np.mean(y)
    x_std, y_std = np.std(x), np.std(y)

    # Mean and covariance matrix
    mean = [x_mean, y_mean]
    cov = np.cov(x, y)

    # Create a 2D Gaussian distribution grid
    xx, yy = np.meshgrid(np.linspace(x_mean - 3*x_std, x_mean + 3*x_std, 100), 
                         np.linspace(y_mean - 3*y_std, y_mean + 3*y_std, 100))
    pos = np.dstack((xx, yy))

    # 2D Gaussian distribution
    rv = multivariate_normal(mean, cov)
    zz = rv.pdf(pos)

    # Plot contour lines for different sigma levels
    sigma_levels = [1, 2, 3]  # 1-sigma, 2-sigma, 3-sigma

    # Calculate contour levels using the corresponding PDF values
    contour_levels = rv.pdf(mean) * np.exp(-np.array(sigma_levels)**2 / 2)
    contour_levels = np.sort(contour_levels) # Ensure levels are in increasing order

    # Plot contour based on the 2D Gaussian distribution
    contours = ax_main.contour(xx, yy, zz, levels=contour_levels, colors='k')

    # Add labels to the contour lines
    fmt = {contour_levels[ii]: f'{sigma_levels[::-1][ii]}$\\sigma$' for ii in range(len(sigma_levels))}
    ax_main.clabel(contours, contours.levels, inline=True, fontsize=10, fmt=fmt)

    # # Overlay Ellipses corresponding to the sigma levels
    # for sigma in sigma_levels[:1]:
    #     eigenvalues, eigenvectors = np.linalg.eigh(cov)
    #     order = eigenvalues.argsort()[::-1]
    #     eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        
    #     angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    #     width, height = 2 * sigma * np.sqrt(eigenvalues)
        
    #     ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
    #                     edgecolor='none', facecolor='lightblue', alpha=0.5)
    #     ax_main.add_patch(ellipse)

    # Calculate the area under the 1-sigma contour line
    sigma = 1
    area_1std = np.pi * x_std * y_std * sigma**2

    # Add emittance and 1sigma area calculations from data in the main plot
    if not apply_q_gaus_fit:
        ax_main.text(0.05, 0.95, f'$\\varepsilon$ (data) = $\\sqrt{{\\langle k^2\\rangle \\langle p_k^2\\rangle-\\langle kp\\rangle^2}}$ = {emit:.2E}\nArea (1$\\sigma_{{data~Gaus. fit}}$) = {area_1std/np.pi:.2E} [$\\pi$]', transform=ax_main.transAxes,
                verticalalignment='top', horizontalalignment='left', fontsize=12, bbox=dict(facecolor='white', alpha=0.6))


    # Plot histograms
    x_hist, x_bins, _ = ax_xhist.hist(x, bins=120, density=True, color = "red", ec="darkred", alpha=0.7)
    y_hist, y_bins, _ = ax_yhist.hist(y, bins=120, density=True, color = "red", ec="darkred", alpha=0.7, orientation='horizontal')

    if apply_gaus_fit or apply_q_gaus_fit:
        # Fit distribution to histograms and plot
        x_bin_centers = 0.5 * (x_bins[1:] + x_bins[:-1])
        y_bin_centers = 0.5 * (y_bins[1:] + y_bins[:-1])

    if apply_gaus_fit:
        # Fit the Gaussian to the x and y data
        initial_guess = [0, 1]  # Initial guess for [mu, sigma]
        popt_x, pcov_x = curve_fit(my_xf.gaussian, x_bin_centers, x_hist, p0=initial_guess, bounds=([-np.inf, 0], [np.inf, np.inf]))
        popt_y, pcov_y = curve_fit(my_xf.gaussian, y_bin_centers, y_hist, p0=initial_guess, bounds=([-np.inf, 0], [np.inf, np.inf]))

        # Extract parameters
        mu_x, sigma_x = popt_x
        mu_y, sigma_y = popt_y

        # Fit Gaussian to histograms and plot
        x_fit = my_xf.gaussian(x_bin_centers, mu_x, sigma_x)
        y_fit = my_xf.gaussian(y_bin_centers, mu_y, sigma_y)

        ax_xhist.plot(x_bin_centers, x_fit, '-', color = "gold")
        ax_yhist.plot(y_fit, y_bin_centers, '-', color = "gold")

        # Add sigma and mu value labels
        ax_xhist.text(0.05, 0.95, f'$\\sigma = {sigma_x:.2f}$\n$\\mu = {mu_x:.2f}$', 
            transform=ax_xhist.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=12, bbox=dict(facecolor='gold', edgecolor='gold', alpha=0.6))
        ax_yhist.text(0.4, 0.95, f'$\\sigma = {sigma_y:.2f}$\n$\\mu = {mu_y:.2f}$', 
            transform=ax_yhist.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=12, bbox=dict(facecolor='gold', edgecolor='gold', alpha=0.6))

    if apply_q_gaus_fit:
        # Fit the q-Gaussian to the x and y data
        initial_guess_q = [0, 1, 2]  # Initial guess for [mu, beta, q]
        popt_x_q, pcov_x_q = curve_fit(my_xf.q_gaussian, x_bin_centers, x_hist, p0=initial_guess_q, bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, 3]))
        popt_y_q, pcov_y_q = curve_fit(my_xf.q_gaussian, y_bin_centers, y_hist, p0=initial_guess_q, bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, 3]))


        # Extract q parameters
        mu_x_q, beta_x_q, q_x = popt_x_q
        mu_y_q, beta_y_q, q_y = popt_y_q

        # Fit q-Gaussian to histograms and plot
        x_fit_q = my_xf.q_gaussian(x_bin_centers, mu_x_q, beta_x_q, q_x)
        y_fit_q = my_xf.q_gaussian(y_bin_centers, mu_y_q, beta_y_q, q_y)

        ax_xhist.plot(x_bin_centers, x_fit_q, '-', color='mediumblue')
        ax_yhist.plot(y_fit_q, y_bin_centers, '-', color='mediumblue')

        # Add q, sigma and mu value labels
        ax_xhist.text(0.95, 0.95, f'$q = {q_x:.2f}$\n$\\beta = {beta_x_q:.2f}$\n$\\mu = {mu_x_q:.2f}$', 
            transform=ax_xhist.transAxes, verticalalignment='top', horizontalalignment='right', fontsize=12, bbox=dict(facecolor='mediumblue', edgecolor='mediumblue', alpha=0.6))
        ax_yhist.text(0.9, 0.15, f'$q = {q_y:.2f}$\n$\\beta = {beta_y_q:.2f}$\n$\\mu = {mu_y_q:.2f}$', 
            transform=ax_yhist.transAxes, verticalalignment='top', horizontalalignment='right', fontsize=12, bbox=dict(facecolor='mediumblue', edgecolor='mediumblue', alpha=0.6))

        # Calculate the total area of the distribution
        area_of_interest = area_calculation(mu_x_q, beta_x_q, q_x, mu_y_q, beta_y_q, q_y)

        # Add labels to the main plot
        ax_main.text(0.05, 0.95, f'$\\varepsilon$ (data) = $\\sqrt{{\\langle k^2\\rangle \\langle p_k^2\\rangle-\\langle kp\\rangle^2}}$ = {emit:.2E}\nArea (1$\\sigma_{{data~Gaus. fit}}$) = {area_1std/np.pi:.2E} [$\\pi$]\nArea (1$\\sigma_{{data~qGaus. fit}}$) = {area_of_interest/np.pi:.2E} [$\\pi$]', transform=ax_main.transAxes,
                verticalalignment='top', horizontalalignment='left', fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

    # Set labels
    ax_main.set_xlabel(axis_label[0])
    ax_main.set_ylabel(axis_label[1])
    ax_main.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    ax_xhist.set_ylabel('Density')
    ax_xhist.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    ax_yhist.set_xlabel('Density')
    ax_yhist.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    plt.grid()
    plt.show()

    return 


def phase_space_evolution_difference (x_ini_cord, y_ini_cord, x_fin_cord, y_fin_cord, x_axis_label=None, y_axis_label=None):
 
    # Generate sample data
    q_initial = x_ini_cord
    pq_initial = y_ini_cord
    q_final = x_fin_cord
    pq_final = y_fin_cord

    if x_axis_label is not None and y_axis_label is not None:
        axis_label = [x_axis_label, y_axis_label]
    else:
        axis_label = ['x', '$p_x$']

    # Create a figure and a GridSpec layout
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Scatter plot of initial values
    ax_scatter = fig.add_subplot(gs[1:, :-1])
    ax_scatter.scatter(q_initial, pq_initial, label='Initial', alpha=0.5, color='blue')
    ax_scatter.scatter(q_final, pq_final, label='Final', alpha=0.5, marker='x', color='orange')
    ax_scatter.set_xlabel(axis_label[0])
    ax_scatter.set_ylabel(axis_label[1])
    ax_scatter.legend()
    ax_scatter.grid()

    # Histogram for q (x-axis)
    ax_hist_q = fig.add_subplot(gs[0, :-1], sharex=ax_scatter)
    ax_hist_q.hist(q_initial, bins=30, alpha=0.5, label='Initial', color='blue')
    ax_hist_q.hist(q_final, bins=30, alpha=0.5, label='Final', color='orange')
    ax_hist_q.legend()
    ax_hist_q.set_ylabel('Density')

    # Histogram for pq (y-axis)
    ax_hist_pq = fig.add_subplot(gs[1:, -1], sharey=ax_scatter)
    ax_hist_pq.hist(pq_initial, bins=30, alpha=0.5, orientation='horizontal', label='Initial', color='blue')
    ax_hist_pq.hist(pq_final, bins=30, alpha=0.5, orientation='horizontal', label='Final', color='orange')
    ax_hist_pq.legend()
    ax_hist_pq.set_xlabel('Density')

    # Clean up overlapping labels
    plt.setp(ax_hist_q.get_xticklabels(), visible=False)
    plt.setp(ax_hist_pq.get_yticklabels(), visible=False)

    plt.show()

    return


def particle_population_vs_turns(parts_state_per_turn, T_rev0=None, bunch_population=None):
    N0 = np.sum(parts_state_per_turn[:, 0])
    N_lifetime = int(np.floor(N0/np.exp(1))) 
    N_turns = np.sum(parts_state_per_turn, axis=0)

    for ii, value in enumerate(N_turns):
        if value <= N_lifetime:
            index = ii+1
            break
        else:
            index = None
    N_fit = N_turns[:index]

    if T_rev0 is not None:
        x_data = T_rev0 * np.arange(len(N_turns))
        x_data_for_fit = T_rev0 * np.arange(len(N_fit))
        time_units = 's'
    else:
        x_data = np.arange(len(N_turns))
        x_data_for_fit = np.arange(len(N_fit))
        time_units = 'turns'

    y_data_for_fit = N_fit / N0
    y_data = 100 * N_turns / N0
    min_population = np.min(y_data)

    # Fit a curve to the data
    if N_turns[-1] <= N_lifetime:
        try:
            popt, pcov = curve_fit(lambda t,tau: np.exp(-t/tau), x_data_for_fit, y_data_for_fit, p0=(x_data_for_fit[-1]))
        except Exception as e:
            popt = [np.nan]
    else:
        popt = [np.nan]

    lifetime_fit = popt[0]
    y_fit = np.exp(-x_data/lifetime_fit) * 100

    if T_rev0 is not None:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()

        # Add some extra space for the second axis at the bottom
        fig.subplots_adjust(bottom=0.2)
    else:
        fig, ax1 = plt.subplots()

    # Plotting on the primary x-axis (time in seconds)
    ax1.plot(x_data, y_data, 'ko', label='Particle population')
    if N_turns[-1] <= N_lifetime:
        ax1.plot(x_data, y_fit, '-', color='red', label='Fit for lifetime estimation')
        ax1.text(x_data[index], y_fit[index], f'$\\tau_{{fit}}$ = {lifetime_fit:.2E} [{time_units}]', #transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='left', fontsize=12, bbox=dict(facecolor='red', edgecolor='red', alpha=0.9))
        ax1.axvline(x=x_data_for_fit[-1], color='k', linestyle='--', label=r'Time for $N\sim\frac{N_0}{e}$')
        ax1.text(x_data_for_fit[-1], 100*0.7, f'$\\tau_{{data}}$ = {x_data_for_fit[-1]:.2E} [{time_units}]', ha='left', fontsize=12, bbox=dict(facecolor='y', edgecolor='y', alpha=0.9))
        
        ax1.axhline(y=np.min(y_data), color='g', linestyle='-')
        index_min = np.where(y_data == np.min(y_data))[0][0] if np.min(y_data) in y_data else -1
        ax1.text(x_data[index_min], np.min(y_data), f'{round(np.min(y_data), 2)}[%]', ha='center', va='center',
                 fontsize=12, bbox=dict(facecolor='green', edgecolor='green', alpha=0.9))

    else:
        ax1.axhline(y=np.min(y_data), color='g', linestyle='-')
        index_min = np.where(y_data == np.min(y_data))[0][0] if np.min(y_data) in y_data else -1
        ax1.text(x_data[index_min], np.min(y_data), f'{round(np.min(y_data), 2)}[%]', ha='center', va='center',
                 fontsize=12, bbox=dict(facecolor='green', edgecolor='green', alpha=0.9))


    ax1.set_xlabel(f't [{time_units}]')
    ax1.set_ylabel(r'$\frac{N}{N_0}$ [%]')
    ax1.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax1.legend(fontsize='small', loc='best')
    ax1.grid()

    if T_rev0 is not None:
        def tick_function(X):
            V = X/T_rev0
            return ["%.0f" % z for z in V]

        new_tick_locations = x_data[::int(len(x_data)/6)]

        # Move twinned axis ticks and label from top to bottom
        ax2.xaxis.set_ticks_position("bottom")
        ax2.xaxis.set_label_position("bottom")

        # Offset the twin axis below the host
        ax2.spines["bottom"].set_position(("axes", -0.2))

        # Turn on the frame for the twin axis, but hide all but the bottom spine
        ax2.set_frame_on(True)
        ax2.patch.set_visible(False)
        for sp in ax2.spines.values():
            sp.set_visible(False)
        ax2.spines["bottom"].set_visible(True)

        # Set the tick locations and labels for the secondary axis
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(new_tick_locations)
        ax2.set_xticklabels(tick_function(new_tick_locations))
        ax2.set_xlabel('turns')

    plt.tight_layout()
    plt.show()

    if np.isnan(lifetime_fit):  # Check if the lifetime_fit is NaN 
        return min_population
    return lifetime_fit


def emittances_vs_turns (emittances_x, emittances_y, emittances_z, reference_emit_x=None, reference_emit_y=None, reference_emit_z=None, log_y_axis=False):

    x_data = np.arange(len(emittances_x))

    fig, axs = plt.subplots(3, 1, sharex=True)

    # Top plot
    axs[0].plot(x_data, emittances_x, '-', color='crimson')
    if reference_emit_x is not None:
        axs[0].axhline(y=reference_emit_x, color='k', linestyle='--', label=f'Reference value = {reference_emit_x:.2E}')
        axs[0].legend(fontsize='small')
    axs[0].set_ylabel(r'$\varepsilon_x$ [m]')
    if log_y_axis:
        axs[0].set_yscale('log')  # Set y-axis to logarithmic scale
    axs[0].grid(True, which="both", ls="--")  # Apply logarithmic grid
    # Remove x-axis ticks and labels from the top plot
    axs[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # Middle plot
    axs[1].plot(x_data, emittances_y, '-', color='y')
    if reference_emit_y is not None:
        axs[1].axhline(y=reference_emit_y, color='k', linestyle='--', label=f'Reference value = {reference_emit_y:.2E}')
        axs[1].legend(fontsize='small')
    axs[1].set_ylabel(r'$\varepsilon_y$ [m]')
    if log_y_axis:
        axs[1].set_yscale('log')  # Set y-axis to logarithmic scale
    axs[1].grid(True, which="both", ls="--")  # Apply logarithmic grid
    # Remove x-axis ticks and labels from the middle plot
    axs[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # Bottom plot
    axs[2].plot(x_data, emittances_z, '-', color='seagreen')
    if reference_emit_z is not None:
        axs[2].axhline(y=reference_emit_z, color='k', linestyle='--', label=f'Reference value = {reference_emit_z:.2E}')
        axs[2].legend(fontsize='small')
    axs[2].set_ylabel(r'$\varepsilon_z$ [m]')
    if log_y_axis:
        axs[2].set_yscale('log')  # Set y-axis to logarithmic scale
    axs[2].set_xlabel('turns')
    axs[2].grid(True, which="both", ls="--")  # Apply logarithmic grid
    # Ensure x-axis ticks and labels are shown on the bottom plot
    axs[2].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)

    # Adjust layout
    plt.tight_layout()
    
    return 


def coordinates_vs_turns (q_coordinates, pq_coordinates, turns, x_axis_label=None, y_axis_label=None, particle_id_list=None, full_init_cond=None, func_for_contour=None):

    particle_id_cmap = truncate_colormap(plt.get_cmap('nipy_spectral'), 0.05, 0.95)
    
    # Create colors and a ScalarMappable for colorbar
    if particle_id_list is not None and full_init_cond is not None:
        colors = [particle_id_cmap(i) for i in particle_id_list/(full_init_cond-1)]
    else:
        colors = [particle_id_cmap(i) for i in np.linspace(0,1,np.shape(q_coordinates)[0])]
    sm = plt.cm.ScalarMappable(cmap=particle_id_cmap, norm=plt.Normalize(vmin=0, vmax=len(colors)-1))

    if x_axis_label is not None and y_axis_label is not None:
        axis_label = [x_axis_label, y_axis_label]
    else:
        axis_label = ['x', '$p_x$']

    if func_for_contour is not None:

        v_func = np.vectorize(func_for_contour)
        xx, yy = np.meshgrid(np.linspace(-0.2, 0.38, 200),
                        np.linspace(-0.014, 0.014, 200))


    # Plot phasespace
    fig, ax = plt.subplots()
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    if func_for_contour is not None:
        ax.contour(xx, yy, v_func(xx, yy), colors='k') # levels=[0], linestyle='--', 
    for ii in range(np.shape(q_coordinates)[0]):
        ax.scatter(q_coordinates[ii,:], pq_coordinates[ii,:], color=colors[ii], s=1)
    ax.set_xlabel(axis_label[0])
    ax.set_ylabel(axis_label[1])

    # Add the colorbar
    cb = fig.colorbar(sm, ax=ax)
    # Remove the colorbar ticks and labels
    cb.set_ticks([])  # Remove ticks
    cb.set_ticklabels([])  # Remove tick labels
    cb.set_label('Particle ID')

    plt.tight_layout()


    # Plot x px vs turn
    plt.figure()
    ax1 = plt.subplot(2, 1, 1)
    ax1.ticklabel_format(style='sci', axis='both', scilimits=(0,0))

    for ii in range(np.shape(q_coordinates)[0]):
        ax1.plot(turns[ii,:], q_coordinates[ii,:], color=colors[ii], linewidth=2)
    ax1.set_ylabel(axis_label[0])

    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    for ii in range(len(pq_coordinates[:,0])):
        ax2.plot(turns[ii,:], pq_coordinates[ii,:], color=colors[ii], linewidth=2)
    ax2.set_ylabel(axis_label[1])
    ax2.set_xlabel('turn')

    plt.tight_layout()

    return


# def plot_twiss_ellips (twiss, norm_emitt_x, norm_emitt_y, lattice_position=None, Jx=None, Jy=None):
    
#     if lattice_position==None:
#         lattice_position = twiss.name[0]
    
#     twiss_at_s = twiss.row['lattice_position'] 
#     sigmas_s = twiss_at_s.get_betatron_sigmas(nemitt_x=norm_emitt_x, nemitt_y=norm_emitt_y)
     
#     if Jx==None:
#         Jx = sigmas_s['Sigma11',0]/twiss_at_s.betx
#     else:
#         Jx = norm_emitt_x/(twiss_at_s.gamma0*twiss_at_s.beta0)
#     if Jy==None:
#         Jy = sigmas_s['Sigma33',0]/twiss_at_s.bety
#     else:
#         Jy = norm_emitt_y/(twiss_at_s.gamma0*twiss_at_s.beta0)
    
#     beta_x = twiss_at_s.betx
#     beta_y = twiss_at_s.bety
#     alfa_x = twiss_at_s.alfx
#     alfa_y = twiss_at_s.alfy
#     gamma_x = (1+alfa_x**2)/beta_x
#     gamma_y = (1+alfa_y**2)/beta_y
    
# # =============================================================================
# #     # numerical method 1
# #     px_p = np.linspace(np.sqrt(2*Jx*gamma_x),-np.sqrt(2*Jx*gamma_x),201)
# #     px_m = np.linspace(-np.sqrt(2*Jx*gamma_x),np.sqrt(2*Jx*gamma_x),201)
# #     x_p = (np.sqrt(4*((alfa_x*px_p)**2-gamma_x*(beta_x*px_p**2-2*Jx)))-2*alfa_x*px_p)/(2*gamma_x)
# #     x_m = (-np.sqrt(4*((alfa_x*px_m)**2-gamma_x*(beta_x*px_m**2-2*Jx)))-2*alfa_x*px_m)/(2*gamma_x)
# #     x = np.concatenate((x_p, x_m))
# #     px = np.concatenate((px_p, px_m))
# #     
# #     py_p = np.linspace(np.sqrt(2*Jy*gamma_y),-np.sqrt(2*Jy*gamma_y),201)
# #     py_m = np.linspace(-np.sqrt(2*Jy*gamma_y),np.sqrt(2*Jy*gamma_y),201)
# #     y_p = (np.sqrt(4*((alfa_y*py_p)**2-gamma_y*(beta_y*py_p**2-2*Jy)))-2*alfa_y*py_p)/(2*gamma_y)
# #     y_m = (-np.sqrt(4*((alfa_y*py_m)**2-gamma_y*(beta_y*py_m**2-2*Jy)))-2*alfa_y*py_m)/(2*gamma_y)
# #     y = np.concatenate((y_p, y_m))
# #     py = np.concatenate((py_p, py_m))
# # =============================================================================
    
    
#     # numerical method 2 which include xcenter and ycenter
#     xcenter_x = twiss_at_s.x
#     ycenter_x = twiss_at_s.px
#     theta_x = np.arctan2(2*alfa_x,(gamma_x-beta_x))/2
#     width_x = np.sqrt(2*Jx)*np.emath.sqrt(np.cos(theta_x)**4-np.sin(theta_x)**4)/np.emath.sqrt(gamma_x*np.cos(theta_x)**2-beta_x*np.sin(theta_x)**2)
#     height_x = np.sqrt(2*Jx)*np.emath.sqrt(np.cos(theta_x)**4-np.sin(theta_x)**4)/np.emath.sqrt(beta_x*np.cos(theta_x)**2-gamma_x*np.sin(theta_x)**2)
    
#     xcenter_y = twiss_at_s.y
#     ycenter_y = twiss_at_s.py
#     theta_y = np.arctan2(2*alfa_y,(gamma_y-beta_y))/2
#     width_y = np.sqrt(2*Jy)*np.emath.sqrt(np.cos(theta_y)**4-np.sin(theta_y)**4)/np.emath.sqrt(gamma_y*np.cos(theta_y)**2-beta_y*np.sin(theta_y)**2)
#     height_y = np.sqrt(2*Jy)*np.emath.sqrt(np.cos(theta_y)**4-np.sin(theta_y)**4)/np.emath.sqrt(beta_y*np.cos(theta_y)**2-gamma_y*np.sin(theta_y)**2)

#     angles = np.deg2rad(np.arange(0.0, 360.0, 1.0))
#     x_x = width_x * np.cos(angles)
#     y_x = height_x * np.sin(angles)
#     x_y = width_y * np.cos(angles)
#     y_y = height_y * np.sin(angles)
    
#     R_x = np.array([
#         [np.cos(theta_x), -np.sin(theta_x)],
#         [np.sin(theta_x),  np.cos(theta_x)],
#         ])
#     R_y = np.array([
#         [np.cos(theta_y), -np.sin(theta_y)],
#         [np.sin(theta_y),  np.cos(theta_y)],
#         ])
    
#     x_x, y_x = np.dot(R_x, np.array([x_x, y_x]))
#     x_x += xcenter_x
#     y_x += ycenter_x
#     x_y, y_y = np.dot(R_y, np.array([x_y, y_y]))
#     x_y += xcenter_y
#     y_y += ycenter_y
    
    
    
#     fig1, ax1 = plt.subplots(constrained_layout=True, dpi=150, facecolor='w', edgecolor='k')
    
# # =============================================================================
# #     ax1.fill(x, px, alpha=0.5, facecolor='red', edgecolor='none')
# #     ax1.plot(x, px,'r-', linewidth=linewidth1, label='x plane')
# # =============================================================================
#     ax1.fill(x_x, y_x, alpha=0.5, facecolor='red', edgecolor='none')
#     ax1.plot(x_x, y_x,'r-', linewidth=linewidth1, label='x plane')
    
# # =============================================================================
# #     ax1.fill(y, py, alpha=0.5, facecolor='blue', edgecolor='none')
# #     ax1.plot(y, py,'b:', linewidth=linewidth1, label='y plane')
# # =============================================================================
#     ax1.fill(x_y, y_y, alpha=0.5, facecolor='blue', edgecolor='none')
#     ax1.plot(x_y, y_y,'b:', linewidth=linewidth1, label='y plane')
    
#     ax1.set_xlabel('x [m], y [m]', weight='bold', fontsize = fontsize1)
#     ax1.set_ylabel('$\mathbf{P_{x}}$, $\mathbf{P_{y}}$', weight='bold', fontsize = fontsize1)
    
#     ax1.tick_params(axis='x', labelsize=fontsize1)
#     ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#     ax1.xaxis.offsetText.set_fontsize(fontsize1)
#     #ax1.set_xlim([min_lim_x, max_lim_x])
    
#     ax1.tick_params(axis='y', labelsize=fontsize1)
#     ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#     ax1.yaxis.offsetText.set_fontsize(fontsize1)
#     #ax1.set_ylim([-600,700])
    
#     ax1.legend(loc='best', fontsize = fontsize1, prop=dict(weight='bold'))
#     ax1.grid()
    
#     fig1.tight_layout()
#     fig1.show()


def plot_resonance_lines (Qxmin=0,Qxmax=1,Qymin=0,Qymax=1,orders=[2],addlabels=0,fontsize1=12):
    
    n = np.empty((1,0))
    m = np.empty((1,0))
    
    for i in range(0,np.size(orders)):
        
        n1 = np.append(np.arange(orders[i], -1, -1), np.arange((orders[i]-1), 0, -1))
        n = np.append(n, n1) # Ola ta n
        m1 = np.append(np.arange(0, (orders[i]+1), 1), np.arange(-1, -orders[i], -1))
        m = np.append(m, m1) # Ola ta m
        
    # Ypologismos ton megiston kai elaxiston timon pou kinite to P gia kathe zeugos n, m
    Pextr = np.transpose([n*Qxmax+m*Qymax, n*Qxmin+m*Qymin, n*Qxmax+m*Qymin, n*Qxmin+m*Qymax])
    Pmax = np.amax(Pextr,axis=1)
    Pmin = np.amin(Pextr,axis=1)

    textlabelspos=np.zeros((1))
    num_dig = 15
    
    nmP = np.zeros((1,3))

    fontsize2=9
    
    #fig, ax = plt.subplots(constrained_layout=True, figsize=(7,5), dpi=150, facecolor='w', edgecolor='k')
    #ax.axes(xlim=(Qxmin,Qxmax), ylim=(Qymin,Qymax))
    
    for j in range(0,np.size(n)):
        
        if np.ceil(Pmin[j]) > Pmax[j]:
            
            nmP = np.vstack((nmP, np.array([[n[j], m[j], np.nan]])))
        
            continue
        
        else:
            
            #del Pvls
            Pvls = np.arange(np.ceil(Pmin[j]), np.floor(Pmax[j])+1, 1);
        
            for ii in range(0,np.size(Pvls)):
            
                if m[j]!=0:
                    
                    if m[j]%2 == 0:
                        line_type='-'
                    else:
                        line_type='--'
                        
                    print (m[j])
                    #darkslategrey
                    plt.plot( [Qxmin,Qxmax],[(Pvls[ii]-n[j]*Qxmin)/m[j],(Pvls[ii]-n[j]*Qxmax)/m[j]], linestyle=line_type, color = 'darkslategrey', linewidth = 2 , zorder=-1)
                
                    if ( addlabels == 1 and np.size(np.where(textlabelspos == np.around(((Pvls[ii]-n[j]*(Qxmin+((Qxmax-Qxmin)/5)))/m[j])*(10**num_dig))/(10**num_dig) )) == 0. ):
                        
                        #plt.text((Qxmin+((Qxmax-Qxmin)/5.0)),(Pvls[ii]-n[j]*(Qxmin+((Qxmax-Qxmin)/5.0)))/m[j],'('+str(int(n[j]))+','+str(int(m[j]))+')',fontsize=fontsize2)
                        if (Qymin<=(Pvls[ii]-n[j]*Qxmin)/m[j]):
                            
                            plt.text(Qxmin+0.005,(Pvls[ii]-n[j]*(Qxmin+0.005))/m[j],'('+str(int(n[j]))+','+str(int(m[j]))+')',fontsize=fontsize2)
                            textlabelspos = np.append(textlabelspos, np.around(((Pvls[ii]-n[j]*(Qxmin+((Qxmax-Qxmin)/5.0)))/m[j])*(10**num_dig))/(10**num_dig))
                            
                        else:
                            
                            plt.text(Qxmax-0.002,(Pvls[ii]-n[j]*(Qxmax-0.002))/m[j],'('+str(int(n[j]))+','+str(int(m[j]))+')',fontsize=fontsize2)
                            textlabelspos = np.append(textlabelspos, np.around(((Pvls[ii]-n[j]*(Qxmin+((Qxmax-Qxmin)/5.0)))/m[j])*(10**num_dig))/(10**num_dig))
                                        
                else:
                    
                    if m[j]%2 == 0:
                        line_type='-'
                    else:
                        line_type='--'
                    
                    print (m[j])
                
                    plt.plot([float(Pvls[ii])/n[j],float(Pvls[ii])/n[j]],[Qymin,Qymax], linestyle=line_type, color = 'darkslategrey', linewidth = 2 , zorder=-1)
                
                    if ( addlabels == 1 and np.size(np.where(textlabelspos == np.around((float(Pvls[ii])/n[j])*(10**num_dig))/(10**num_dig) )) == 0. ):
                        plt.text(float(Pvls[ii])/n[j],(Qymin+((Qymax-Qymin)/5.0)),'('+str(int(n[j]))+','+str(int(m[j]))+')', fontsize = fontsize2)
                        textlabelspos = np.append(textlabelspos, np.around((Pvls[ii]/n[j])*(10**num_dig))/(10**num_dig))
                
            
                nmP = np.vstack((nmP, np.array([n[j], m[j], Pvls[ii]])))
    
    plt.plot(np.nan, np.nan, '--', color = 'darkslategrey', label='Resonance lines')
    #ax.set_xlabel('$\\mathbf{Q_x}$', weight='bold', fontsize = fontsize1)
    #ax.set_ylabel('$\\mathbf{Q_y}$', weight='bold', fontsize = fontsize1)
    #ax.tick_params(axis='x', labelsize=fontsize1)
    #ax.tick_params(axis='y', labelsize=fontsize1)
    #major_ticks_x = np.arange(Qxmin, Qxmax, 4)      
    #major_ticks_y = np.arange(Qymin, Qymax, 4)                    
    #plt.set_xticks(major_ticks_x)
    #plt.set_yticks(major_ticks_y)
    #plt.grid()
    #plt.show()
    
    np.delete(nmP, 0, 0)
    
    #return (nmP)


def tune_diffusion (Qx_in, Qx_fi, Qy_in, Qy_fi, initial_conditions_x_axis=None, initial_conditions_y_axis=None, xlabel='x [m]', ylabel='y [m]', resonance_orders=None, annotate=False, delta_value= None):

    tune_diff = np.log10(np.sqrt((Qx_fi-Qx_in)**2 + (Qy_fi-Qy_in)**2))

    # Plot tune diffusion in tune space
    fig = plt.subplots()
    plt.scatter(Qx_in, Qy_in, c=tune_diff, cmap='plasma')
    if resonance_orders is not None:
        dQx=np.abs(np.max(Qx_in)-np.min(Qx_in))*0.1
        dQy=np.abs(np.max(Qy_in)-np.min(Qy_in))*0.1
        # plot_resonance_lines(Qxmin=np.min(Qx_in)-dQx, Qxmax=np.max(Qx_in)+dQx, Qymin=np.min(Qy_in)-dQy, Qymax=np.max(Qy_in)+dQy,orders=resonance_orders)
        rp.plot_res_of_orders(resonance_orders, c1='gray',lst1='-',c2='gray',lst2='--',c3='gray',annotate=annotate)
    if delta_value is not None:
        plt.plot(np.nan, np.nan, '>', color='k', label='$\delta$=%.1E'%(delta_value))
    plt.xlabel(r'$Q_x$')
    plt.ylabel(r'$Q_y$')
    cb = plt.colorbar()
    cb.set_label(r'log$_{10}$$\left[\sqrt{(Q_x^{final}-Q_x^{initial})^2 + (Q_y^{final}-Q_y^{initial})^2}\right]$')
    if delta_value is not None:
        plt.legend(fontsize='small', loc='best')
    plt.tight_layout()

    if initial_conditions_x_axis is not None and initial_conditions_y_axis is not None:
        # Plot tune diffusion in initial condition space space
        fig,ax = plt.subplots()
        # ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        sc = ax.scatter(initial_conditions_x_axis, initial_conditions_y_axis, c=tune_diff, cmap='plasma')
        if delta_value is not None:
            ax.plot(np.nan, np.nan, '>', color='k', label='$\delta$=%.1E'%(delta_value))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        cb = fig.colorbar(sc)
        cb.set_label(r'log$_{10}$$\left[\sqrt{(Q_x^{final}-Q_x^{initial})^2 + (Q_y^{final}-Q_y^{initial})^2}\right]$')
        if delta_value is not None:
            ax.legend(fontsize='small', loc='best')
        fig.tight_layout()

    return
            
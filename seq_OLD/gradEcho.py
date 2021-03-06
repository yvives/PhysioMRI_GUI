#!/usr/bin/env python3

import numpy as np
import experiment as ex
import matplotlib.pyplot as plt

import pdb
st = pdb.set_trace

def trapezoid(plateau_a, total_t, ramp_t, ramp_pts, total_t_end_to_end=True, base_a=0):
    """Helper function that just generates a Numpy array starting at time
    0 and ramping down at time total_t, containing a trapezoid going from a
    level base_a to plateau_a, with a rising ramp of duration ramp_t and
    sampling period ramp_ts."""

    # ramp_pts = int( np.ceil(ramp_t/ramp_ts) ) + 1
    rise_ramp_times = np.linspace(0, ramp_t, ramp_pts)
    rise_ramp = np.linspace(base_a, plateau_a, ramp_pts)

    # [1: ] because the first element of descent will be repeated
    descent_t = total_t - ramp_t if total_t_end_to_end else total_t
    t = np.hstack([rise_ramp_times, rise_ramp_times[:-1] + descent_t])
    a = np.hstack([rise_ramp, np.flip(rise_ramp)[1:]])
    return t, a
    
def grad_echo(self, plotSeq):

#    plotSeq=0  # 0 to run the sequence, 1 to plot the sequence
#              trs=21, plot_rx=False, init_gpa=False,
#              dbg_sc=0.5, # set to 0 to avoid 2nd RF debugging pulse, otherwise amp between 0 or 1
#              lo_freq=0.1, # MHz
#              rf_amp=1, # 1 = full-scale
#
#              slice_amp=0.4, # 1 = gradient full-scale
#              phase_amp=0.3, # 1 = gradient full-scale
#              readout_amp=0.8, # 1 = gradient full-scale
#              rf_duration=50,
#              trap_ramp_duration=50, # us, ramp-up/down time
#              trap_ramp_pts=5, # how many points to subdivide ramp into
#              phase_delay=100, # how long after RF end before starting phase ramp-up
#              phase_duration=200, # length of phase plateau
#              tr_wait=100, # delay after end of RX before start of next TR
#
#              rx_period=10/3 # us, 3.333us, 300 kHz rate


    ## All times are in the context of a single TR, starting at time 0
    tr_wait = 100
    trap_ramp_pts=5
    phase_amps = np.linspace(self.phase_amp, -self.phase_amp, self.trs)

    # Shimming
    shim_x: int = self.shim[0]
    shim_y: int = self.shim[1]
    shim_z: int = self.shim[2]

    #rf_tstart = 100 # us
    self.rf_tend = self.rf_tstart + self.rf_duration # us

    slice_tstart = self.rf_tstart - self.trap_ramp_duration
    slice_duration = (self.rf_tend - self.rf_tstart) + 2*self.trap_ramp_duration # includes rise, plateau and fall
    phase_tstart = self.rf_tend + self.phase_delay
    readout_tstart = phase_tstart
    readout_duration = self.phase_duration*2

    rx_tstart = readout_tstart + self.trap_ramp_duration # us
    rx_tend = readout_tstart + readout_duration - self.trap_ramp_duration # us

    tx_gate_pre = 2 # us, time to start the TX gate before the RF pulse begins
    tx_gate_post = 1 # us, time to keep the TX gate on after the RF pulse ends

    tr_total_time = readout_tstart + readout_duration + tr_wait + 7000 # start-finish TR time

    def grad_echo_tr(tstart, pamp):
        gvxt, gvxa = trapezoid(self.slice_amp, slice_duration, self.trap_ramp_duration, trap_ramp_pts)
        gvyt, gvya = trapezoid(pamp, self.phase_duration, self.trap_ramp_duration, trap_ramp_pts)

        gvzt1 = trapezoid(self.readout_amp, readout_duration/2, self.trap_ramp_duration, trap_ramp_pts)
        gvzt2 = trapezoid(-self.readout_amp, readout_duration/2, self.trap_ramp_duration, trap_ramp_pts)
        gvzt = np.hstack([gvzt1[0], gvzt2[0] + readout_duration/2])
        gvza = np.hstack([gvzt1[1], gvzt2[1]])

        rx_tcentre = (rx_tstart + rx_tend) / 2
        value_dict = {
            # second tx0 pulse purely for loopback debugging
            'tx0': ( np.array([self.rf_tstart, self.rf_tend,   rx_tcentre - 10, rx_tcentre + 10]) + tstart,
                     np.array([self.rf_amp,0,  self.dbg_sc*(1 + 0.5j),0]) ),
            'grad_vx': ( gvxt + tstart + slice_tstart, gvxa+shim_x ),
            'grad_vy': ( gvyt + tstart + phase_tstart, gvya+shim_y),
            'grad_vz': ( gvzt + tstart + readout_tstart, gvza+shim_z),
            'rx0_en': ( np.array([rx_tstart, rx_tend]) + tstart, np.array([1, 0]) ),
            'tx_gate': ( np.array([self.rf_tstart - tx_gate_pre, self.rf_tend + tx_gate_post]) + tstart, np.array([1, 0]) )
        }

        return value_dict

    expt = ex.Experiment(lo_freq=self.lo_freq, rx_t=self.rx_period, init_gpa=self.init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    # gpa_fhdo_offset_time in microseconds; offset between channels to
    # avoid parallel updates (default update rate is 0.2 Msps, so
    # 1/0.2 = 5us, 5 / 3.1 gives the offset between channels; extra
    # 0.1 for a safety margin)

    tr_t = 20 # start the first TR at 20us
    for pamp in phase_amps:
        expt.add_flodict( grad_echo_tr( tr_t, pamp) )
        tr_t += tr_total_time
    
    
    if plotSeq==1:
        expt.plot_sequence()
        plt.show()
        expt.__del__()
    elif plotSeq==0:
        rxd, msgs = expt.run()
        expt.__del__()
        return rxd['rx0'], msgs





#!/usr/bin/env python3
import sys
sys.path.append('../marcos_client')
import numpy as np
import experiment as ex
import matplotlib.pyplot as plt

import pdb
st = pdb.set_trace

def fid(self, plotSeq):
#             self, dbg_sc=0.5, # set to 0 to avoid 2nd RF debugging pulse, otherwise amp between 0 or 1
#             lo_freq=0.1, # MHz
#             rf_amp=1, # 1 = full-scale
#             rf_duration=50,
#             rf_tstart = 100,  # us
#             tr_wait=100, # delay after end of RX before start of next TR
#             rx_period=10/3,  # us, 3.333us, 300 kHz rate
#             readout_duration=500
#             ):

    dbg_sc=self.dbg_sc
    lo_freq=self.lo_freq
    rf_amp=self.rf_amp
    rf_duration=self.rf_duration
    rf_tstart=self.rf_tstart
    rf_wait=self.rf_wait
    rx_period=self.rx_period
    readout_duration=self.readout_duration
    
    ## All times are in the context of a single TR, starting at time 0
    init_gpa = True

#    phase_amps = np.linspace(phase_amp, -phase_amp, trs)
    rf_tend = rf_tstart + rf_duration # us

    rx_tstart = rf_tend+rf_wait # us
    rx_tend = rx_tstart + readout_duration  # us

    tx_gate_pre = 2 # us, time to start the TX gate before the RF pulse begins
    tx_gate_post = 1 # us, time to keep the TX gate on after the RF pulse ends


    def fid_tr(tstart):
        rx_tcentre = (rx_tstart + rx_tend) / 2
        value_dict = {
            # second tx0 pulse purely for loopback debugging
            'tx0': ( np.array([rf_tstart, rf_tend,   rx_tcentre - 10, rx_tcentre + 10]) + tstart,
                     np.array([rf_amp,0,  dbg_sc*(1 + 0.5j),0]) ),
            'rx0_en': ( np.array([rx_tstart, rx_tend]) + tstart, np.array([1, 0]) ),
             'tx_gate': ( np.array([rf_tstart - tx_gate_pre, rf_tend + tx_gate_post]) + tstart, np.array([1, 0]) )
        }

        return value_dict

    expt = ex.Experiment(lo_freq=lo_freq, rx_t=rx_period, init_gpa=init_gpa)
    # gpa_fhdo_offset_time in microseconds; offset between channels to
    # avoid parallel updates (default update rate is 0.2 Msps, so
    # 1/0.2 = 5us, 5 / 3.1 gives the offset between channels; extra
    # 0.1 for a safety margin)

    tr_t = 20 # start the first TR at 20us
    expt.add_flodict( fid_tr( tr_t) )

    if plotSeq==1:
        expt.plot_sequence()
        plt.show()
        expt.__del__()
    elif plotSeq==0:
        rxd, msgs = expt.run()
        expt.__del__()
        return rxd['rx0'], msgs
        

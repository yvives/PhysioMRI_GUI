import sys
sys.path.append('../marcos_client')
import numpy as np
import experiment as ex
from configs.hw_config import Gx_factor
from configs.hw_config import Gy_factor
from configs.hw_config import Gz_factor
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


def trap_cent(centre_t, plateau_a, trap_t, ramp_t, ramp_pts, base_a=0):
    """Like trapezoid, except it generates a trapezoid shape around a centre
    time, with a well-defined area given by its amplitude (plateau_a)
    times its time (trap_t), which is defined from the start of the
    ramp-up to the start of the ramp-down, or (equivalently) from the
    centre of the ramp-up to the centre of the ramp-down. All other
    parameters are as for trapezoid()."""
    t, a = trapezoid(plateau_a, trap_t, ramp_t, ramp_pts, False, base_a)
    return t + centre_t - (trap_t + ramp_t)/2, a

def spin_echo(lo_freq=3.0399, # MHz
                    rf_amp=0.6, # 1 = full-scale
                    rf_pi2_duration=1000, # us, rf pi/2 pulse length
                    rf_pi_duration=None, # us, rf pi pulse length  - if None then automatically gets set to 2 * rf_pi2_duration
                    # spin-echo properties
                    echo_duration=10000, # us, time from the centre of one echo to centre of the next
                    tr_duration=500000, 
                    BW=31, #                
                    shim_x=0, 
                    shim_y=0, 
                    shim_z=0, 
                    nScans = 1, 
                    n_rd=800, 
                    n_ph=1, 
                    n_sl=1, 
                    fov_rd=1000000, 
                    fov_ph=10000000, 
                    fov_sl=10000000, 
                    trap_ramp_duration=100, 
                    phase_grad_duration=500
                    # (must at least be longer than readout_duration + trap_ramp_duration)
                    ):
                        
    init_gpa=True                   
#    trs=self.trs
    rf_pi_duration=None

    fov_rd=fov_rd*1e-2
    fov_ph=fov_ph*1e-2
    fov_sl=fov_sl*1e-2
    
    
    """
    readout gradient: x
    phase gradient: y
    slice/partition gradient: z
    """
    init_gpa=False
                    
    if rf_pi_duration is None:
        rf_pi_duration = 2 * rf_pi2_duration

    BW=BW*1e-3
    trap_ramp_pts=np.int32(trap_ramp_duration*0.2)    # 0.2 puntos/ms
    grad_readout_delay=9   #8.83    # readout amplifier delay
    grad_phase_delay=9      #8.83
    grad_slice_delay=9        #8.83
    rx_period=1/BW
    """
    readout gradient: x
    phase gradient: y
    slice/partition gradient: z
    """

    readout_duration = n_rd/BW
    
#    echos_per_tr=1 # number of spin echoes (180 pulses followed by readouts) to do
                    
    if rf_pi_duration is None:
        rf_pi_duration = 2 * rf_pi2_duration
        
#    SweepMode=1
        
    gammaB = 42.56e6    # Hz/T
    # readout amplitude
    Grd = BW*1e6/(gammaB*fov_rd)
    # slice amplitude
    Gph = n_ph/(2*gammaB*fov_ph*phase_grad_duration*1e-6)
    # phase amplitude
    Gsl = n_sl/(2*gammaB*fov_sl*phase_grad_duration*1e-6)
    
    phase_amps = np.linspace(Gph, -Gph, n_ph)
#    phase_amps=phase_amps[getIndex(phase_amps, echos_per_tr, SweepMode)]
    slice_amps = np.linspace(Gsl, -Gsl,  n_sl)

#    slice_amps=slice_amps[getIndex(slice_amps, echos_per_tr, SweepMode)]

    # create appropriate waveforms for each echo, based on start time, echo index and TR index
    # note: echo index is 0 for the first interval (90 pulse until first 180 pulse) thereafter 1, 2 etc between each 180 pulse

    def rf_wf(tstart, echo_idx):
        pi2_phase = 1 # x
        pi_phase = 1j # y
        if echo_idx == 0:
            # do pi/2 pulse, then start first pi pulse
            return np.array([tstart + (echo_duration - rf_pi2_duration)/2, tstart + (echo_duration + rf_pi2_duration)/2,
                             tstart + echo_duration - rf_pi_duration/2]), np.array([pi2_phase*rf_amp, 0, pi_phase*rf_amp])                        
#        elif tr_idx == echos_per_tr:
#            # finish final RF pulse
#            return np.array([tstart + rf_pi_duration/2]), np.array([0])
        else:
            # finish last pi pulse, start next pi pulse
            return np.array([tstart + rf_pi_duration/2]), np.array([0])

    def tx_gate_wf(tstart, echo_idx):
        tx_gate_pre = 2 # us, time to start the TX gate before each RF pulse begins
        tx_gate_post = 1 # us, time to keep the TX gate on after an RF pulse ends

        if echo_idx == 0:
            # do pi/2 pulse, then start first pi pulse
            return np.array([tstart + (echo_duration - rf_pi2_duration)/2 - tx_gate_pre,
                             tstart + (echo_duration + rf_pi2_duration)/2 + tx_gate_post,
                             tstart + echo_duration - rf_pi_duration/2 - tx_gate_pre]), \
                             np.array([1, 0, 1])
#        elif echo_idx == echos_per_tr:
#            # finish final RF pulse
#            return np.array([tstart + rf_pi_duration/2 + tx_gate_post]), np.array([0])
        else:
            # finish last pi pulse, start next pi pulse
            return np.array([tstart + rf_pi_duration/2 + tx_gate_post]), np.array([0])

    def readout_wf(tstart, echo_idx):
        if echo_idx != 0:
            return np.array([tstart + (echo_duration - readout_duration)/2, tstart + (echo_duration + readout_duration)/2 ]), np.array([1, 0])
        else:
            return np.array([tstart]), np.array([0]) # keep on zero otherwise
            
            
    def readout_grad_wf(tstart, echo_idx):

        if echo_idx == 0:
                    #            return trap_cent(tstart + self.echo_duration*3/4, readout_amp, readout_grad_duration/2,
                    #                             trap_ramp_duration, trap_ramp_pts)
            return trap_cent(tstart + echo_duration/2 + rf_pi2_duration/2+trap_ramp_duration/2+readout_duration/4, Grd, readout_duration/2,
                             trap_ramp_duration, trap_ramp_pts)
        else:
            return trap_cent(tstart + echo_duration/2-grad_readout_delay, Grd, readout_duration,
                             trap_ramp_duration, trap_ramp_pts)
        

    def phase_grad_wf(tstart, echo_idx, n_ph):
        t1, a1 = trap_cent(tstart + (rf_pi_duration+phase_grad_duration-trap_ramp_duration)/2+trap_ramp_duration-grad_phase_delay, phase_amps[n_ph-1], phase_grad_duration,
                           trap_ramp_duration, trap_ramp_pts)
        t2, a2 = trap_cent(tstart + (echo_duration + readout_duration+trap_ramp_duration)/2+trap_ramp_duration-grad_phase_delay, -phase_amps[n_ph-1], phase_grad_duration,
                           trap_ramp_duration, trap_ramp_pts)    
        if echo_idx == 0:
            return np.array([tstart]), np.array([0]) # keep on zero otherwise
#        elif echo_idx == echos_per_tr: # last echo, don't need 2nd trapezoids
#            return t1, a1
        else: # otherwise do both trapezoids
            return np.hstack([t1, t2]), np.hstack([a1, a2])

    def slice_grad_wf(tstart, echo_idx,  n_sl):
        t1, a1 = trap_cent(tstart + (rf_pi_duration+phase_grad_duration-trap_ramp_duration)/2+trap_ramp_duration-grad_phase_delay, slice_amps[n_sl-1], phase_grad_duration,
                           trap_ramp_duration, trap_ramp_pts)
        t2, a2 = trap_cent(tstart + (echo_duration + readout_duration+trap_ramp_duration)/2+trap_ramp_duration-grad_slice_delay, -slice_amps[n_sl-1], phase_grad_duration,
                           trap_ramp_duration, trap_ramp_pts)  
        if echo_idx == 0:
            return np.array([tstart]), np.array([0]) # keep on zero otherwise
#        elif echo_idx == echos_per_tr: # last echo, don't need 2nd trapezoids
#            return t1, a1
        else: # otherwise do both trapezoids
            return np.hstack([t1, t2]), np.hstack([a1, a2])
            
    expt = ex.Experiment(lo_freq=lo_freq, rx_t=rx_period, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    # gpa_fhdo_offset_time in microseconds; offset between channels to
    # avoid parallel updates (default update rate is 0.2 Msps, so
    # 1/0.2 = 5us, 5 / 3.1 gives the offset between channels; extra
    # 0.1 for a safety margin))

    global_t = 20 # start the first TR at 20us
    for nS in range(nScans):
        for sl in range(n_sl):
            for ph in range(n_ph):
                for echo_idx in range(2):
                    tx_t, tx_a = rf_wf(global_t, echo_idx)
                    tx_gate_t, tx_gate_a = tx_gate_wf(global_t, echo_idx)
                    readout_t, readout_a = readout_wf(global_t, echo_idx)
                    rx_gate_t, rx_gate_a = readout_wf(global_t, echo_idx)
                    readout_grad_t, readout_grad_a = readout_grad_wf(global_t, echo_idx)
                    phase_grad_t, phase_grad_a = phase_grad_wf(global_t, echo_idx,  n_ph)
                    slice_grad_t, slice_grad_a = slice_grad_wf(global_t, echo_idx,  n_sl)
    
                    expt.add_flodict({
                        'tx0': (tx_t, tx_a),
                        'grad_vx': (readout_grad_t, readout_grad_a/Gx_factor/10+shim_x),
                        'grad_vy': (phase_grad_t, phase_grad_a/Gy_factor/10+shim_y),
                        'grad_vz': (slice_grad_t, slice_grad_a/Gz_factor/10+shim_z), 
                        'rx0_en': (readout_t, readout_a),
                        'tx_gate': (tx_gate_t, tx_gate_a),
                        'rx_gate': (rx_gate_t, rx_gate_a),
                    })
                    global_t += echo_duration
                
                global_t += tr_duration-echo_duration
    
    expt.plot_sequence()
    plt.show()     
    
    rxd, msgs = expt.run()
    plt.plot( rxd['rx0'])
    plt.show()
#        

#    return rxd['rx0'], msgs
    
    expt.__del__()

    if nScans > 1:
        data_avg = np.average(np.reshape(rxd['rx0'], (nScans, n_rd*n_ph*n_sl)), axis=0)
    else:
        data_avg = rxd['rx0']

#
if __name__ == "__main__":
    
    spin_echo()

"""
@author: T. Guallart Naval, february 03th 2022
MRILAB @ I3M
"""

import sys
sys.path.append('../marcos_client')
import experiment as ex
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from datetime import date,  datetime 
import os
from scipy.io import savemat

def noiseStandalone(
    nReadout =2500,
    BW =50, 
    nScans = 1):

    init_gpa= False
    plotSeq = 0
    larmorFreq = 3.00
    oversamplingFactor=6
    BW=BW*1e-3
    if nReadout%2==0:
        nReadout = nReadout+1
    

    #ARRAY INITIALIZATIONS 
    rxTime = []
    rxAmp = []
    
    # RAWDATA FIELDS
    rawData = {}
    rawData['larmorFreq'] = larmorFreq   
    rawData['nReadout'] = nReadout
    rawData['BW'] = BW
    
    # RX PULSE
    def readoutGate(tRef,tRd,rxTimePrevious,  rxAmpPrevious):
        rxTime = np.array([tRef-tRd/2, tRef+tRd/2])
        rxAmp = np.array([1,0])
        rxTime=np.concatenate((rxTimePrevious, rxTime),  axis=0)
        rxAmp=np.concatenate((rxAmpPrevious, rxAmp),  axis=0)
        return rxTime,  rxAmp
    
    # SAVE DATA
    def saveData(rawData):
        dt = datetime.now()
        dt_string = dt.strftime("%Y.%m.%d.%0H.%M.%S")
        dt2 = date.today()
        dt2_string = dt2.strftime("%Y.%m.%d")
        if not os.path.exists('experiments/acquisitions/%s' % (dt2_string)):
            os.makedirs('experiments/acquisitions/%s' % (dt2_string))
                
        if not os.path.exists('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)):
            os.makedirs('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)) 
        
#        if not os.path.exists('/media/physiomri/TOSHIBA EXT/experiments/acquisitions/%s' % (dt2_string)):
#            os.makedirs('/media/physiomri/TOSHIBA EXT/experiments/acquisitions/%s'% (dt2_string) )
#            
#        if not os.path.exists('/media/physiomri/TOSHIBA EXT/experiments/acquisitions/%s/%s' % (dt2_string, dt_string)):
#            os.makedirs('/media/physiomri/TOSHIBA EXT/experiments/acquisitions/%s/%s'% (dt2_string, dt_string) )
        
        rawData['fileName'] = "%s.%s.mat" % ("NOISE",dt_string)
        savemat("experiments/acquisitions/%s/%s/%s.%s.mat" % (dt2_string, dt_string, "NOISE_standalone",dt_string),  rawData) 
#        savemat("/media/physiomri/TOSHIBA EXT/experiments/acquisitions/%s/%s.%s.%s.mat" %(dt2_string, dt_string, "NOISE_standalone" ,dt_string), dict)
        
        return rawData['fileName']
    
    #  STANDALONE FUNCTIONS
    def  plotData(data, tAdqReal, nRd):
       plt.figure(1)
       tPlot = np.linspace(0, tAdqReal, nReadout,  endpoint ='True')*1e-3
       plt.plot(tPlot[5:], np.abs(data[5:]))
       plt.plot(tPlot[5:], np.real(data[5:]))
       plt.plot(tPlot[5:], np.imag(data[5:]))
       plt.xlabel('t(ms)')
       plt.ylabel('A(mV)')
       vRMS=np.std(np.abs(data[5:]))
       titleRF= 'BW = '+ str(np.round(nRd/(tAdqReal)*1e3))+'kHz; Vrms ='+str(vRMS)
       plt.title(titleRF)
    
    
    def plotDataK(data, BW, nReadout, name):
            plt.figure(2)
            fAdq =  np.linspace(-BW/2, BW/2, nReadout, endpoint=True)*1e3
            dataFft = np.fft.fft(data[5:])
            dataOr1, dataOr2 = np.split(dataFft, 2, axis=0)
            dataFft= np.concatenate((dataOr2, dataOr1), axis=0)
            plt.plot(fAdq[5:], np.abs(dataFft), 'r-')
            plt.xlabel('f(kHz)')
            plt.ylabel('A(a.u.)')
            plt.title(name)



    # INIT EXPERIMENT
    BWov = BW*oversamplingFactor
    samplingPeriod = 1/BWov
    expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    samplingPeriod = expt.get_rx_ts()[0]
    BWReal = 1/samplingPeriod/oversamplingFactor
    tAdqReal = nReadout/BWReal  #            plt.xlim(-0.05,  0.05)
#            plt.legend()
    
    # SEQUENCE
    tStart = 20
    tRef = tStart+tAdqReal/2
    rxTime, rxAmp = readoutGate(tRef, tAdqReal, rxTime, rxAmp)
    expt.add_flodict({
                        'rx0_en': (rxTime, rxAmp),
                        'rx_gate': (rxTime, rxAmp),
                        })
    if plotSeq == 0:
        dataFull=[]
        for i in range(nScans):
            print(i)
            rxd, msgs = expt.run()
            rxd['rx0'] = rxd['rx0']*13.788
            data = rxd['rx0'][0:nReadout*oversamplingFactor]
            dataFull=np.concatenate((dataFull, data), axis = 0)
        
        expt.__del__()
        dataFull = np.reshape(dataFull, (-1,nReadout*oversamplingFactor))
        dataFull = np.reshape(dataFull, -1)
        dataFull = sig.decimate(dataFull, oversamplingFactor, ftype='fir', zero_phase=True)
        dataProv = np.reshape(dataFull, (nScans, nReadout))
        dataProv = np.average(dataProv, axis=0)
        data = dataProv
        rawData['data'] = data
        name = saveData(rawData)
        plotData(data,tAdqReal, nReadout)
        plotDataK(data, BWReal, nReadout, name)
        plt.show()
    elif plotSeq == 1:
        expt.plot_sequence()
        plt.show()
        expt.__del__()


#  MAIN  ######################################################################################################
if __name__ == "__main__":
    noiseStandalone()

"""
@author: Teresa Guallart Naval
MRILAB @ I3M
"""

import sys
sys.path.append('../marcos_client')
import experiment as ex
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.signal as sig
from datetime import date,  datetime 
import os
from scipy.io import savemat


def rabiflopStandalone(
    init_gpa= False,                 
    larmorFreq=3.038, 
    rfExAmp=0.3, 
    rfReAmp=None, 
    rfExPhase = 0,
    rfExTimeIni=10, 
    rfExTimeEnd = 120, 
    nExTime =10, 
    nReadout =800,
    tAdq = 4*1e3,
    tEcho = 20*1e3,
    tRepetition =1000*1e3, 
    plotSeq = 0, 
    shimming=[-80, -100, 10]):

#  INITALISATION OF VARIABLES  ################################################################################
    #CONTANTS
    tStart = 20
    txGatePre = 15
    txGatePost = 1
    oversamplingFactor=6
    shimming=np.array(shimming)*1e-4
    
    #ARRAY INITIALIZATIONS 
    txTime=[]
    txAmp=[]
    txGateTime=[]
    txGateAmp=[]
    rxTime = []
    rxAmp = []
    dataAll  =[]
    
    #RF PULSES
    if rfReAmp is None:
        rfReAmp = rfExAmp
    
    # Inputs for rawData
    rawData={}
    rawData['larmorFreq'] = larmorFreq      # Larmor frequency
    rawData['rfExAmp'] = rfExAmp             # rf excitation pulse amplitude
    rawData['rfReAmp'] = rfReAmp             # rf refocusing pulse amplitude
    rawData['rfExTimeIni'] = rfExTimeIni          # rf excitation pulse time
    rawData['rfExTimeEnd'] = rfExTimeEnd           # rf refocusing pulse time
    rawData['nExTime'] = nExTime
    rawData['echoSpacing'] = tEcho        # time between echoes
    rawData['repetitionTime'] = tRepetition     # TR
    rawData['nReadout'] = nReadout
    rawData['tAdq'] = tAdq
    rawData['shimming'] = shimming
    
    rfExPhase = rfExPhase*np.pi/180
    rfExAmp = rfExAmp*np.exp(1j*rfExPhase)
    rfRePhase = 0
    rfReAmp = rfReAmp *np.exp(1j*rfRePhase)
    #Excitation times
    rfExTime= np.linspace(rfExTimeIni, rfExTimeEnd, nExTime,  endpoint=True)
    
#  DEFINITION OF PULSES   ####################################################################################
    def rfPulse(tRef, rfAmp, rfDuration, txTimePrevious,txAmpPrevious,  txGateTimePrevious, txGateAmpPrevious):
        txTime = np.array([tRef-rfDuration/2,tRef+rfDuration/2])
        txAmp = np.array([rfAmp,0.])
        txGateTime = np.array([txTime[0]-txGatePre,txTime[1]+txGatePost])
        txGateAmp = np.array([1,0])
        txTime = np.concatenate((txTimePrevious,txTime),  axis=0)
        txAmp = np.concatenate((txAmpPrevious,txAmp ),  axis=0)
        txGateTime = np.concatenate((txGateTimePrevious,txGateTime),  axis=0)
        txGateAmp = np.concatenate((txGateAmpPrevious,txGateAmp),  axis=0)
        return txTime,  txAmp,  txGateTime,  txGateAmp
    
    def readoutGate(tRef,tRd,rxTimePrevious,  rxAmpPrevious):
        rxTime = np.array([tRef-tRd/2, tRef+tRd/2])
        rxAmp = np.array([1,0])
        rxTime=np.concatenate((rxTimePrevious, rxTime),  axis=0)
        rxAmp=np.concatenate((rxAmpPrevious, rxAmp),  axis=0)
        return rxTime,  rxAmp

#  SPECIFIC FUNCTIONS   ####################################################################################
    def  plotData(data, rfExTime, tAdqReal):
       plt.figure(1)
       colors = cm.rainbow(np.linspace(0, 0.8, len(rfExTime)))
       for indexExTime in range(nExTime):
            tPlot = np.linspace(-tAdqReal/2, tAdqReal/2, nReadout,  endpoint ='True')*1e-3
            leg = 'Time = '+ str(np.round(rfExTime[indexExTime]))+ 'us'
            plt.plot(tPlot[5:], np.abs(data[indexExTime, 5:]),  label = leg, color=colors[indexExTime])
#            plt.plot(tPlot[5:], np.real(data[indexExTime, 5:]))
#            plt.plot(tPlot[5:], np.imag(data[indexExTime, 5:]))
       plt.xlabel('t(ms)')
       plt.ylabel('A(mV)')
       plt.legend()
    
    def  getRabiFlopData(data, rfExTime, tAdqReal):
       for indexExTime in range(nExTime):
            if indexExTime == 0:
                maxEchoes = np.max(np.abs(data[indexExTime,5:]))
            else:
                maxEchoes=np.append(maxEchoes,np.max(np.abs(data[indexExTime, 5:])))
       rabiFlopData= np.transpose(np.array([rfExTime, maxEchoes]))
       return rabiFlopData 
    
    def  plotRabiFlop(rabiFlopData,  name):
       plt.figure(2)
       plt.plot(rabiFlopData[:, 0], rabiFlopData[:, 1])
       plt.xlabel('t(us)')
       plt.ylabel('A(mV)')
       titleRF= 'RF Amp = '+ str(np.real(rfExAmp)) + ';  ' + name
       plt.title(titleRF)
       return rabiFlopData 
 
    def saveMyData(rawData):
        # Save data
        dt = datetime.now()
        dt_string = dt.strftime("%Y.%m.%d.%H.%M.%S")
        dt2 = date.today()
        dt2_string = dt2.strftime("%Y.%m.%d")
        if not os.path.exists('experiments/acquisitions/%s' % (dt2_string)):
            os.makedirs('experiments/acquisitions/%s' % (dt2_string))
        if not os.path.exists('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)):
            os.makedirs('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)) 
        rawData['name'] = "%s.%s.mat" % ("RABIFLOP",dt_string)
        savemat("experiments/acquisitions/%s/%s/%s.%s.mat" % (dt2_string, dt_string, "RABIFLOP",dt_string),  rawData)
        return rawData['name']


#  SEQUENCE  ############################################################################################

    for indexExTime in range(nExTime):
        
        rfReTime = 2*rfExTime[indexExTime]
    
        txTime=[]
        txAmp=[]
        txGateTime=[]
        txGateAmp=[]
        rxTime = []
        rxAmp = []
        
        # INIT EXPERIMENT
        BW = nReadout/tAdq
        BWov = BW*oversamplingFactor
        samplingPeriod = 1/BWov
        expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
        samplingPeriod = expt.get_rx_ts()[0]
        BWReal = 1/samplingPeriod/oversamplingFactor
        tAdqReal = nReadout/BWReal  
        tIni=20  #us initial time
    # Shimming
        expt.add_flodict({
            'grad_vx': (np.array([tIni]),np.array([shimming[0]])), 
            'grad_vy': (np.array([tIni]),np.array([shimming[1]])),  
            'grad_vz': (np.array([tIni]),np.array([shimming[2]])),
        })
        # TR    
        tRef = tStart+rfExTime[indexExTime]/2+tIni+100
        txTime, txAmp,txGateTime,txGateAmp = rfPulse(tRef,rfExAmp, rfExTime[indexExTime], txTime, txAmp, txGateTime, txGateAmp)
        tRef = tRef+tEcho/2
        txTime, txAmp, txGateTime, txGateAmp = rfPulse(tRef,rfReAmp, rfReTime, txTime, txAmp, txGateTime, txGateAmp)
        tRef = tRef+tEcho/2
        rxTime, rxAmp = readoutGate(tRef, tAdqReal, rxTime, rxAmp)
        
        expt.add_flodict({
                            'tx0': (txTime, txAmp),
                            'tx_gate': (txGateTime, txGateAmp), 
                            'rx0_en': (rxTime, rxAmp),
                            'rx_gate': (rxTime, rxAmp),
                            })
        tEnd = tRepetition
        expt.add_flodict({
            'grad_vx': (np.array([tEnd]),np.array([0])), 
            'grad_vy': (np.array([tEnd]),np.array([0])), 
            'grad_vz': (np.array([tEnd]),np.array([0])),
        })

        if plotSeq == 0:
            print(indexExTime,  '.- Running...')
            rxd, msgs = expt.run()
            expt.__del__()
            print('   End')
            data = sig.decimate(rxd['rx0']*13.788, oversamplingFactor, ftype='fir', zero_phase=True)
            dataAll = np.concatenate((dataAll, data), axis=0)
            rawData['dataFull'] = dataAll
        elif plotSeq == 1:
            expt.plot_sequence()
            plt.show()
            expt.__del__()
   
    if plotSeq == 1:
        expt.plot_sequence()
        plt.show()
        expt.__del__()
    elif plotSeq == 0:
        data = np.reshape(dataAll,  (nExTime,  nReadout))
        plotData(data, rfExTime, tAdqReal)
        rabiFlopData = getRabiFlopData(data, rfExTime, tAdqReal)
        rawData['rabiFlopData'] = rabiFlopData
        name = saveMyData(rawData)
        plotRabiFlop(rabiFlopData,  name)
        plt.show()

#  MAIN  ######################################################################################################
if __name__ == "__main__":
    rabiflopStandalone()

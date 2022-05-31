"""
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia, Spain
@date: 19 tue Apr 2022
@email: josalggui@i3m.upv.es
"""

import sys
import os
#******************************************************************************
# Add path to the working directory
path = os.path.realpath(__file__)
ii = 0
for char in path:
    if (char=='\\' or char=='/') and path[ii+1:ii+14]=='PhysioMRI_GUI':
        # sys.path.append(path[0:ii])
        print("Path: ",path[0:ii+1])
        sys.path.append(path[0:ii+1]+'PhysioMRI_GUI')
        sys.path.append(path[0:ii+1]+'marcos_client')
    ii += 1
#******************************************************************************
import experiment as ex
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import configs.hw_config as hw # Import the scanner hardware config
import mrilabMethods.mrilabMethods as mri

def sweepGSE_standalone(
        init_gpa = False,
        larmorFreq = 3.08, # MHz
        rfExAmp = 0.0, # a.u.
        rfReAmp = 0.0, # a.u.
        rfExTime = 22, # us
        rfReTime = 0.0, # us
        nPoints = 60,
        acqTime = 4, # ms
        inversionTime = 0,  # ms
        echoTime = 20, # ms
        repetitionTime = 1000, # ms
        pulseShape = 'Rec',  # 'Rec' for square pulse shape, 'Sinc' for sinc pulse shape
        refMethod = 'Amp',  # 'Amp' for rfReAmp=2*rfExAmp, 'Time' for rfReTime=2*rfReTime
        shimming = [-70, -90, 10],
        sweepVar = 'echoTime',
        nSteps = 10,
        valIni = 10e-3, # in fundamental units
        valFin = 100e-3, # in fundamental units
    ):
    
    freqCal = False
    plotSeq = 0
    
    if plotSeq==0:
        plt.ion()
    
    # Refocusing pulse
    if refMethod=='Amp' and rfReAmp==0: rfReAmp  = 2*rfExAmp
    elif refMethod=='Time' and rfReTime==0: rfReTime = 2*rfExTime

    # Varibales to fundamental units
    rfExTime *= 1e-6
    rfReTime *= 1e-6
    acqTime *= 1e-3
    inversionTime *=1e3
    echoTime *= 1e-3
    repetitionTime *= 1e-3
    shimming = np.array(shimming) * 1e-4

    # Inputs for rawData
    rawData = {}
    rawData['seqName'] = 'gseSweep Standalone'
    rawData['larmorFreq'] = larmorFreq*1e6  # Larmor frequency
    rawData['rfExAmp'] = rfExAmp  # rf excitation pulse amplitude
    rawData['rfReAmp'] = rfReAmp
    rawData['rfExTime'] = rfExTime
    rawData['rfReTime'] = rfReTime
    rawData['nPoints'] = nPoints
    rawData['acqTime'] = acqTime
    rawData['echoTime'] = echoTime
    rawData['repetitionTime'] = repetitionTime
    rawData['pulseShape'] = pulseShape
    rawData['sweepVar'] = sweepVar
    rawData['nSteps'] = nSteps
    rawData['valIni'] = valIni
    rawData['valFin'] = valFin
    rawData['shimming'] = shimming
    rawData['inversionTime'] = inversionTime

    # Bandwidth
    bw = nPoints / acqTime * 1e-6  # MHz
    bwov = bw * hw.oversamplingFactor  # MHz
    samplingPeriod = 1 / bwov  # us
    rawData['bw'] = bw
    rawData['bwOV'] = bwov
    rawData['samplingPeriodOV'] = samplingPeriod

    # Create vector to sweep
    valSweep = np.linspace(valIni, valFin, nSteps, endpoint=True)

    # Create sequence
    def createSequence(rawData):
        # Set shimming
        mri.iniSequence(expt, 20, rawData['shimming'])

        # Initialize time
        tEx = 20e3+inversionTime
        
        # Inversion time
        t0 = tEx - rawData['inversionTime']*1e6 - rawData['rfReTime']*1e6 / 2 -hw.blkTime
        if pulseShape == 'Rec':
            mri.rfRecPulse(expt, t0, rawData['rfReTime']*1e6, rawData['rfExAmp'], 0)
        elif pulseShape == 'Sinc':
            mri.rfSincPulse(expt, t0, rawData['rfReTime']*1e6, 7, rawData['rfExAmp'], 0)
        
        # Excitation pulse
        t0 = tEx - hw.blkTime - rawData['rfExTime']*1e6 / 2
        if pulseShape == 'Rec':
            mri.rfRecPulse(expt, t0, rawData['rfExTime']*1e6, rawData['rfExAmp'], 0)
        elif pulseShape == 'Sinc':
            mri.rfSincPulse(expt, t0, rawData['rfExTime']*1e6, 7, rawData['rfExAmp'], 0)

        # Refocusing pulse
        t0 = tEx + rawData['echoTime']*1e6 / 2 - rawData['rfReTime'] / 2 - hw.blkTime
        if pulseShape == 'Rec':
            mri.rfRecPulse(expt, t0, rawData['rfReTime']*1e6, rawData['rfReAmp'], np.pi / 2)
        elif pulseShape == 'Sinc':
            mri.rfSincPulse(expt, t0, rawData['rfReTime']*1e6, 7, rawData['rfReAmp'], np.pi / 2)

        # Acquisition window
        t0 = tEx + rawData['echoTime']*1e6 - rawData['acqTime'] *1e6/ 2
        mri.rxGate(expt, t0, rawData['acqTime']*1e6)

        # End sequence
        mri.endSequence(expt, rawData['repetitionTime']*1e6)

    # Calibrate frequency
    if freqCal:
        mri.freqCalibration(rawData, bw=0.05)
        mri.freqCalibration(rawData, bw=0.005)
        larmorFreq = rawData['larmorFreq'] * 1e-6

    # Sweep experiment
    dataMax = []
    dataAll = []
    for step in range(nSteps):
        if sweepVar=='ShimX':
            shimming[0] = valSweep[step]
            rawData['shimming'] = shimming
        elif sweepVar=='ShimY':
            shimming[1] = valSweep[step]
            rawData['shimming'] = shimming
        elif sweepVar=='ShimZ':
            shimming[2] = valSweep[step]
            rawData['shimming'] = shimming
        else:
            rawData[sweepVar] = valSweep[step]
        expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa,
                             gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
        samplingPeriod = expt.get_rx_ts()[0]
        bw = 1 / samplingPeriod / hw.oversamplingFactor
        acqTime = nPoints / bw 
        rawData['acqTime'] = nPoints/bw*1e-6 # seconds
        rawData['bw'] = bw
        rawData['bwOV'] = bw * hw.oversamplingFactor
        createSequence(rawData)
        if plotSeq==1:
            expt.plot_sequence()
            plt.show()
            expt.__del__()
            break
        elif plotSeq==0:
            print('Step: ',step,'.- Running...')
            rxd, msgs = expt.run()
            expt.__del__()
            data = sig.decimate(rxd['rx0']*13.788, hw.oversamplingFactor, ftype='fir', zero_phase=True)
            dataAll = np.concatenate((dataAll, data), axis=0)
            maxVal = np.max(np.abs(data))
            dataMax.append(maxVal)
            # Plots
            plt.figure(1)
            plt.plot(rawData[sweepVar], maxVal, 'b.')
            plt.show(block=False)
            plt.pause(0.05)
            plt.xlabel(sweepVar)
            plt.ylabel('Max signal (mV)')
    rawData['dataFull'] = dataAll
    rawData['dataMax'] = np.array(dataMax)
    
    # Save data
    mri.saveRawData(rawData)
    
    plt.title(rawData['fileName'])


#  MAIN  ######################################################################################################
if __name__ == "__main__":
    sweepGSE_standalone()

# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 12:40:05 2021
@author: José Miguel Algarín Guisado
MRILAB @ I3M
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
import numpy as np
import experiment as ex
import matplotlib.pyplot as plt
import scipy.signal as sig
import pdb
import configs.hw_config as hw # Import the scanner hardware config
import mrilabMethods.mrilabMethods as mri # This import all methods inside the mrilabMethods module
st = pdb.set_trace

#*********************************************************************************
#*********************************************************************************
#*********************************************************************************


def rare_standalone(
    init_gpa=False, # Starts the gpa
    nScans = 1, # NEX
    larmorFreq = 3.075, # MHz, Larmor frequency
    rfExAmp = 0.0, # a.u., rf excitation pulse amplitude
    rfReAmp = 0.0, # a.u., rf refocusing pulse amplitude
    rfExTime = 40, # us, rf excitation pulse time
    rfReTime = 80, # us, rf refocusing pulse time
    echoSpacing = 10, # ms, time between echoes
    preExTime = 0., # ms, Time from preexcitation pulse to inversion pulse
    inversionTime = 0., # ms, Inversion recovery time
    repetitionTime = 200., # ms, TR
    fov = np.array([140., 150., 180.]), # mm, FOV along readout, phase and slice
    dfov = np.array([0., 0., 0.]), # mm, displacement of fov center
    nPoints = np.array([80, 1, 1]), # Number of points along readout, phase and slice
    etl = 10, # Echo train length
    acqTime = 4, # ms, acquisition time
    axes = np.array([2, 1, 0]), # 0->x, 1->y and 2->z defined as [rd,ph,sl]
    axesEnable = np.array([0, 0, 0]), # 1-> Enable, 0-> Disable
    sweepMode = 1, # 0->k2k (T2),  1->02k (T1),  2->k20 (T2), 3->Niquist modulated (T2)
    rdGradTime = 5,  # ms, readout gradient time
    rdDephTime = 2,  # ms, readout dephasing time
    phGradTime = 1, # ms, phase and slice dephasing time
    rdPreemphasis = 1.00, # readout dephasing gradient is multiplied by this factor
    drfPhase = 0, # degrees, phase of the excitation pulse
    dummyPulses = 1, # number of dummy pulses for T1 stabilization
    shimming = np.array([-70, -90., 10]), # a.u.*1e4, shimming along the X,Y and Z axes
    parFourierFraction = 1 # fraction of acquired k-space along phase direction
    ):
    
    freqCal = 1
    
    # rawData fields
    rawData = {}
    
    # Conversion of variables to non-multiplied units
    larmorFreq = larmorFreq*1e6
    rfExTime = rfExTime*1e-6
    rfReTime = rfReTime*1e-6
    fov = np.array(fov)*1e-3
    dfov = np.array(dfov)*1e-3
    echoSpacing = echoSpacing*1e-3
    acqTime = acqTime*1e-3
    shimming = shimming*1e-4
    repetitionTime= repetitionTime*1e-3
    preExTime = preExTime*1e-3
    inversionTime = inversionTime*1e-3
    rdGradTime = rdGradTime*1e-3
    rdDephTime = rdDephTime*1e-3
    phGradTime = phGradTime*1e-3
    
    # Inputs for rawData
    rawData['seqName'] = 'RARE_Standalone'
    rawData['nScans'] = nScans
    rawData['larmorFreq'] = larmorFreq      # Larmor frequency
    rawData['rfExAmp'] = rfExAmp             # rf excitation pulse amplitude
    rawData['rfReAmp'] = rfReAmp             # rf refocusing pulse amplitude
    rawData['rfExTime'] = rfExTime          # rf excitation pulse time
    rawData['rfReTime'] = rfReTime            # rf refocusing pulse time
    rawData['echoSpacing'] = echoSpacing        # time between echoes
    rawData['preExTime'] = preExTime
    rawData['inversionTime'] = inversionTime       # Inversion recovery time
    rawData['repetitionTime'] = repetitionTime     # TR
    rawData['fov'] = fov           # FOV along readout, phase and slice
    rawData['dfov'] = dfov            # Displacement of fov center
    rawData['nPoints'] = nPoints                 # Number of points along readout, phase and slice
    rawData['etl'] = etl                    # Echo train length
    rawData['acqTime'] = acqTime             # Acquisition time
    rawData['axesOrientation'] = axes       # 0->x, 1->y and 2->z defined as [rd,ph,sl]
    rawData['axesEnable'] = axesEnable # 1-> Enable, 0-> Disable
    rawData['sweepMode'] = sweepMode               # 0->k2k (T2),  1->02k (T1),  2->k20 (T2), 3->Niquist modulated (T2)
    rawData['rdPreemphasis'] = rdPreemphasis
    rawData['drfPhase'] = drfPhase 
    rawData['dummyPulses'] = dummyPulses                    # Dummy pulses for T1 stabilization
    rawData['parFourierFraction'] = parFourierFraction
    rawData['rdDephTime'] = rdDephTime
    rawData['shimming'] = shimming
    
    # Miscellaneous
    larmorFreq = larmorFreq*1e-6
    gradRiseTime = 400e-6       # Estimated gradient rise time
    gSteps = int(gradRiseTime*1e6/5)*0+1
    addRdPoints = 10             # Initial rd points to avoid artifact at the begining of rd
    randFactor = 0e-3                        # Random amplitude to add to the phase gradients
    if rfReAmp==0:
        rfReAmp = rfExAmp
    if rfReTime==0:
        rfReTime = 2*rfExTime
    resolution = fov/nPoints
    rawData['resolution'] = resolution
    rawData['gradRiseTime'] = gradRiseTime
    rawData['randFactor'] = randFactor
    rawData['addRdPoints'] = addRdPoints
    
    # Matrix size
    nRD = nPoints[0]+2*addRdPoints
    nPH = nPoints[1]*axesEnable[1]+(1-axesEnable[1])
    nSL = nPoints[2]*axesEnable[2]+(1-axesEnable[2])
    
    # ETL if nPH = 1
    if etl>nPH:
        etl = nPH
    
    # parAcqLines in case parAcqLines = 0
    parAcqLines = int(int(nPoints[2]*parFourierFraction)-nPoints[2]/2)
    rawData['partialAcquisition'] = parAcqLines
    
    # BW
    BW = nPoints[0]/acqTime*1e-6
    BWov = BW*hw.oversamplingFactor
    samplingPeriod = 1/BWov
    rawData['bw'] = BW
    rawData['samplingPeriod'] = samplingPeriod
    
    # Readout gradient time
    if rdGradTime>0 and rdGradTime<acqTime:
        rdGradTime = acqTime
    rawData['rdGradTime'] = rdGradTime
    
    # Phase and slice de- and re-phasing time
    if phGradTime==0 or phGradTime>echoSpacing/2-rfExTime/2-rfReTime/2-2*gradRiseTime:
        phGradTime = echoSpacing/2-rfExTime/2-rfReTime/2-2*gradRiseTime
    rawData['phGradTime'] = phGradTime
    
    # Max gradient amplitude
    rdGradAmplitude = nPoints[0]/(hw.gammaB*fov[0]*acqTime)*axesEnable[0]
    phGradAmplitude = nPH/(2*hw.gammaB*fov[1]*(phGradTime+gradRiseTime))*axesEnable[1]
    slGradAmplitude = nSL/(2*hw.gammaB*fov[2]*(phGradTime+gradRiseTime))*axesEnable[2]
    rawData['rdGradAmplitude'] = rdGradAmplitude
    rawData['phGradAmplitude'] = phGradAmplitude
    rawData['slGradAmplitude'] = slGradAmplitude

    # Readout dephasing amplitude
    rdDephAmplitude = 0.5*rdGradAmplitude*(gradRiseTime+rdGradTime)/(gradRiseTime+rdDephTime)
    rawData['rdDephAmplitude'] = rdDephAmplitude

    # Phase and slice gradient vector
    phGradients = np.linspace(-phGradAmplitude,phGradAmplitude,num=nPH,endpoint=False)
    slGradients = np.linspace(-slGradAmplitude,slGradAmplitude,num=nSL,endpoint=False)
    
    # Now fix the number of slices to partailly acquired k-space
    nSL = (int(nPoints[2]/2)+parAcqLines)*axesEnable[2]+(1-axesEnable[2])
    
    # Add random displacemnt to phase encoding lines
    for ii in range(nPH):
        if ii<np.ceil(nPH/2-nPH/20) or ii>np.ceil(nPH/2+nPH/20):
            phGradients[ii] = phGradients[ii]+randFactor*np.random.randn()
    kPH = hw.gammaB*phGradients*(gradRiseTime+phGradTime)
    rawData['phGradients'] = phGradients
    rawData['slGradients'] = slGradients
    
    # Set phase vector to given sweep mode
    ind = mri.getIndex(etl, nPH, sweepMode)
    rawData['sweepOrder'] = ind
    phGradients = phGradients[ind]

    def createSequence():
        phIndex = 0
        slIndex = 0
        nRepetitions = int(nPH*nSL/etl+dummyPulses)
        scanTime = 20e3+nRepetitions*repetitionTime
        rawData['scanTime'] = scanTime*1e-6
        if rdGradTime==0:   # Check if readout gradient is dc or pulsed
            dc = True
        else:
            dc = False
        # Set shimming
        mri.iniSequence(expt, 20, shimming)
        for repeIndex in range(nRepetitions):
            # Initialize time
            tEx = 20e3+repetitionTime*repeIndex+inversionTime+preExTime
            
            # First I do a noise measurement.
            if repeIndex==0:
                t0 = tEx-preExTime-inversionTime-4*acqTime
                mri.rxGate(expt, t0, acqTime+2*addRdPoints/BW)
            
            # Pre-excitation pulse
            if repeIndex>=dummyPulses and preExTime!=0:
                t0 = tEx-preExTime-inversionTime-rfExTime/2-hw.blkTime
                mri.rfRecPulse(expt, t0,rfExTime,rfExAmp/90*90,0)
                mri.gradTrap(expt, t0+hw.blkTime+rfReTime, gradRiseTime, preExTime*0.5, -0.005, gSteps, axes[0], shimming)
                mri.gradTrap(expt, t0+hw.blkTime+rfReTime, gradRiseTime, preExTime*0.5, -0.005, gSteps, axes[1], shimming)
                mri.gradTrap(expt, t0+hw.blkTime+rfReTime, gradRiseTime, preExTime*0.5, -0.005, gSteps, axes[2], shimming)
                
            # Inversion pulse
            if repeIndex>=dummyPulses and inversionTime!=0:
                t0 = tEx-inversionTime-rfReTime/2-hw.blkTime
                mri.rfRecPulse(expt, t0,rfReTime,rfReAmp/180*180,0)
                mri.gradTrap(expt, t0+hw.blkTime+rfReTime, gradRiseTime, inversionTime*0.5, 0.005, gSteps, axes[0], shimming)
                mri.gradTrap(expt, t0+hw.blkTime+rfReTime, gradRiseTime, inversionTime*0.5, 0.005, gSteps, axes[1], shimming)
                mri.gradTrap(expt, t0+hw.blkTime+rfReTime, gradRiseTime, inversionTime*0.5, 0.005, gSteps, axes[2], shimming)
            
            # DC gradient if desired
            if (repeIndex==0 or repeIndex>=dummyPulses) and dc==True:
                t0 = tEx-10e3
                mri.gradTrap(expt, t0, gradRiseTime, 10e3+echoSpacing*(etl+1), rdGradAmplitude, gSteps, axes[0], shimming)
            
            # Excitation pulse
            t0 = tEx-hw.blkTime-rfExTime/2
            mri.rfRecPulse(expt, t0,rfExTime,rfExAmp,drfPhase)
        
            # Dephasing readout
            t0 = tEx+rfExTime/2-hw.gradDelay
            if (repeIndex==0 or repeIndex>=dummyPulses) and dc==False:
                mri.gradTrap(expt, t0, gradRiseTime, rdDephTime, rdDephAmplitude*rdPreemphasis, gSteps, axes[0], shimming)
            
            # Echo train
            for echoIndex in range(etl):
                tEcho = tEx+echoSpacing*(echoIndex+1)
                
                # Refocusing pulse
                t0 = tEcho-echoSpacing/2-rfReTime/2-hw.blkTime
                mri.rfRecPulse(expt, t0, rfReTime, rfReAmp, drfPhase+np.pi/2)
    
                # Dephasing phase and slice gradients
                if repeIndex>=dummyPulses:         # This is to account for dummy pulses
                    t0 = tEcho-echoSpacing/2+rfReTime/2-hw.gradDelay
                    mri.gradTrap(expt, t0, gradRiseTime, phGradTime, phGradients[phIndex], gSteps, axes[1], shimming)
                    mri.gradTrap(expt, t0, gradRiseTime, phGradTime, slGradients[slIndex], gSteps, axes[2], shimming)
                
                # Readout gradient
                if (repeIndex==0 or repeIndex>=dummyPulses) and dc==False:         # This is to account for dummy pulses
                    t0 = tEcho-rdGradTime/2-gradRiseTime-hw.gradDelay
                    mri.gradTrap(expt, t0, gradRiseTime, rdGradTime, rdGradAmplitude, gSteps, axes[0], shimming)
    
                # Rx gate
                if (repeIndex==0 or repeIndex>=dummyPulses):
                    t0 = tEcho-acqTime/2-addRdPoints/BW
                    mri.rxGate(expt, t0, acqTime+2*addRdPoints/BW)
    
                # Rephasing phase and slice gradients
                t0 = tEcho+acqTime/2+addRdPoints/BW-hw.gradDelay
                if (echoIndex<etl-1 and repeIndex>=dummyPulses):
                    mri.gradTrap(expt, t0, gradRiseTime, phGradTime, -phGradients[phIndex], gSteps, axes[1], shimming)
                    mri.gradTrap(expt, t0, gradRiseTime, phGradTime, -slGradients[slIndex], gSteps, axes[2], shimming)
                elif(echoIndex==etl-1 and repeIndex>=dummyPulses):
                    mri.gradTrap(expt, t0, gradRiseTime, phGradTime, +phGradients[phIndex], gSteps, axes[1], shimming)
                    mri.gradTrap(expt, t0, gradRiseTime, phGradTime, +slGradients[slIndex], gSteps, axes[2], shimming)
    
                # Update the phase and slice gradient
                if repeIndex>=dummyPulses:
                    if phIndex == nPH-1:
                        phIndex = 0
                        slIndex += 1
                    else:
                        phIndex += 1
                
            if repeIndex==nRepetitions-1:
                mri.endSequence(expt, scanTime)
    
    # Changing time parameters to us
    rfExTime = rfExTime*1e6
    rfReTime = rfReTime*1e6
    echoSpacing = echoSpacing*1e6
    repetitionTime = repetitionTime*1e6
    gradRiseTime = gradRiseTime*1e6
    phGradTime = phGradTime*1e6
    rdGradTime = rdGradTime*1e6
    rdDephTime = rdDephTime*1e6
    inversionTime = inversionTime*1e6
    preExTime = preExTime*1e6
    
    # Calibrate frequency
    if freqCal==1: 
        mri.freqCalibration(rawData, bw=0.05)
        mri.freqCalibration(rawData, bw=0.005)
        larmorFreq = rawData['larmorFreq']*1e-6
        drfPhase = rawData['drfPhase']
    
    # Create full sequence
    expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    samplingPeriod = expt.get_rx_ts()[0]
    BW = 1/samplingPeriod/hw.oversamplingFactor
    acqTime = nPoints[0]/BW        # us
    createSequence()

    # Plot sequence:
#    expt.plot_sequence()
        
    # Run the experiment
    dataFull = []
    dummyData = []
    overData = []
    noise = []
    for ii in range(nScans):
        print("Scan %s ..." % (ii+1))
        rxd, msgs = expt.run()
        rxd['rx0'] = rxd['rx0']*13.788   # Here I normalize to get the result in mV
        # Get noise data
        noise = np.concatenate((noise, rxd['rx0'][0:nRD*hw.oversamplingFactor]), axis = 0)
        rxd['rx0'] = rxd['rx0'][nRD*hw.oversamplingFactor::]
        # Get data
        if dummyPulses>0:
            dummyData = np.concatenate((dummyData, rxd['rx0'][0:nRD*etl*hw.oversamplingFactor]), axis = 0)
            overData = np.concatenate((overData, rxd['rx0'][nRD*etl*hw.oversamplingFactor::]), axis = 0)
        else:
            overData = np.concatenate((overData, rxd['rx0']), axis = 0)
    expt.__del__()
    print('Scans done!')
    rawData['noiseData'] = noise
    rawData['overData'] = overData
    
    # Fix the echo position using oversampled data
    if dummyPulses>0:
        dummyData = np.reshape(dummyData,  (nScans, etl, nRD*hw.oversamplingFactor))
        dummyData = np.average(dummyData, axis=0)
        rawData['dummyData'] = dummyData
        overData = np.reshape(overData, (nScans, int(nPH/etl*nSL), etl,  nRD*hw.oversamplingFactor))
        for ii in range(nScans):
            overData[ii, :, :, :] = mri.fixEchoPosition(dummyData, overData[ii, :, :, :])
        overData = np.squeeze(np.reshape(overData, (1, nRD*hw.oversamplingFactor*nPH*nSL*nScans)))
    
    # Generate dataFull
    dataFull = sig.decimate(overData, hw.oversamplingFactor, ftype='fir', zero_phase=True)
    
    # Get index for krd = 0
    # Average data
    dataProv = np.reshape(dataFull, (nScans, nRD*nPH*nSL))
    dataProv = np.average(dataProv, axis=0)
    # Reorganize the data acording to sweep mode
    dataProv = np.reshape(dataProv, (nSL, nPH, nRD))
    dataTemp = dataProv*0
    for ii in range(nPH):
        dataTemp[:, ind[ii], :] = dataProv[:,  ii, :]
    dataProv = dataTemp
    # Check where is krd = 0
    dataProv = dataProv[int(nPoints[2]/2), int(nPH/2), :]
    indkrd0 = np.argmax(np.abs(dataProv))
    if  indkrd0 < nRD/2-addRdPoints or indkrd0 > nRD/2+addRdPoints:
        indkrd0 = int(nRD/2)

    # Get individual images
    dataFull = np.reshape(dataFull, (nScans, nSL, nPH, nRD))
    dataFull = dataFull[:, :, :, indkrd0-int(nPoints[0]/2):indkrd0+int(nPoints[0]/2)]
    dataTemp = dataFull*0
    for ii in range(nPH):
        dataTemp[:, :, ind[ii], :] = dataFull[:, :,  ii, :]
    dataFull = dataTemp
    imgFull = dataFull*0
    for ii in range(nScans):
        imgFull[ii, :, :, :] = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(dataFull[ii, :, :, :])))
    rawData['dataFull'] = dataFull
    rawData['imgFull'] = imgFull    
    
    # Average data
    data = np.average(dataFull, axis=0)
    data = np.reshape(data, (nSL, nPH, nPoints[0]))
    
    # Do zero padding
    dataTemp = np.zeros((nPoints[2], nPoints[1], nPoints[0]))
    dataTemp = dataTemp+1j*dataTemp
    dataTemp[0:nSL, :, :] = data
    data = np.reshape(dataTemp, (1, nPoints[0]*nPoints[1]*nPoints[2]))
    
    # Fix the position of the sample according to dfov
    kMax = np.array(nPoints)/(2*np.array(fov))*np.array(axesEnable)
    kRD = np.linspace(-kMax[0],kMax[0],num=nPoints[0],endpoint=False)
#        kPH = np.linspace(-kMax[1],kMax[1],num=nPoints[1],endpoint=False)
    kSL = np.linspace(-kMax[2],kMax[2],num=nPoints[2],endpoint=False)
    kPH = kPH[::-1]
    kPH, kSL, kRD = np.meshgrid(kPH, kSL, kRD)
    kRD = np.reshape(kRD, (1, nPoints[0]*nPoints[1]*nPoints[2]))
    kPH = np.reshape(kPH, (1, nPoints[0]*nPoints[1]*nPoints[2]))
    kSL = np.reshape(kSL, (1, nPoints[0]*nPoints[1]*nPoints[2]))
    dPhase = np.exp(-2*np.pi*1j*(dfov[0]*kRD+dfov[1]*kPH+dfov[2]*kSL))
    data = np.reshape(data*dPhase, (nPoints[2], nPoints[1], nPoints[0]))
    rawData['kSpace3D'] = data
    img=np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data)))
    rawData['image3D'] = img
    data = np.reshape(data, (1, nPoints[0]*nPoints[1]*nPoints[2]))
    
    # Create sampled data
    kRD = np.reshape(kRD, (nPoints[0]*nPoints[1]*nPoints[2], 1))
    kPH = np.reshape(kPH, (nPoints[0]*nPoints[1]*nPoints[2], 1))
    kSL = np.reshape(kSL, (nPoints[0]*nPoints[1]*nPoints[2], 1))
    data = np.reshape(data, (nPoints[0]*nPoints[1]*nPoints[2], 1))
    rawData['kMax'] = kMax
    rawData['sampled'] = np.concatenate((kRD, kPH, kSL, data), axis=1)
    data = np.reshape(data, (nPoints[2], nPoints[1], nPoints[0]))
    
    # Save data
    mri.saveRawData(rawData)
    
    # Plot data for 1D case
    if (nPH==1 and nSL==1):
        # Plot k-space
        plt.figure(3)
        dataPlot = data[0, 0, :]
        plt.subplot(1, 2, 1)
        if axesEnable[0]==0:
            tVector = np.linspace(-acqTime/2, acqTime/2, num=nPoints[0],endpoint=False)*1e-3
            sMax = np.max(np.abs(dataPlot))
            indMax = np.argmax(np.abs(dataPlot))
            timeMax = tVector[indMax]
            sMax3 = sMax/3
            dataPlot3 = np.abs(np.abs(dataPlot)-sMax3)
            indMin = np.argmin(dataPlot3)
            timeMin = tVector[indMin]
            T2 = np.abs(timeMax-timeMin)
            plt.plot(tVector, np.abs(dataPlot))
            plt.plot(tVector, np.real(dataPlot))
            plt.plot(tVector, np.imag(dataPlot))
            plt.xlabel('t (ms)')
            plt.ylabel('Signal (mV)')
            print("T2 = %s us" % (T2))
            plt.title(rawData['fileName'])
            plt.legend(['Abs', 'Real', 'Imag'])
        else:
            plt.plot(kRD[:, 0], np.abs(dataPlot))
            plt.yscale('log')
            plt.xlabel('krd (mm^-1)')
            plt.ylabel('Signal (mV)')
            echoTime = np.argmax(np.abs(dataPlot))
            echoTime = kRD[echoTime, 0]
            print("Echo position = %s mm^{-1}" %round(echoTime, 1))
            plt.title(rawData['fileName'])
        
        # Plot image
        plt.subplot(122)
        img = img[0, 0, :]
        if axesEnable[0]==0:
            xAxis = np.linspace(-BW/2, BW/2, num=nPoints[0], endpoint=False)*1e3
            plt.plot(xAxis, np.abs(img), '.')
            plt.xlabel('Frequency (kHz)')
            plt.ylabel('Density (a.u.)')
            print("Smax = %s mV" % (np.max(np.abs(img))))
            plt.title(rawData['fileName'])
        else:
            xAxis = np.linspace(-fov[0]/2*1e2, fov[0]/2*1e2, num=nPoints[0], endpoint=False)
            plt.plot(xAxis, np.abs(img))
            plt.xlabel('Position RD (cm)')
            plt.ylabel('Density (a.u.)')
            plt.title(rawData['fileName'])
    else:
        # Plot k-space
        plt.figure(3)
        dataPlot = data[round(nSL/2), :, :]
        plt.subplot(131)
        plt.imshow(np.log(np.abs(dataPlot)),cmap='gray')
        plt.axis('off')
        # Plot image
        if sweepMode==3:
            imgPlot = img[round(nSL/2), round(nPH/4):round(3*nPH/4), :]
        else:
            imgPlot = img[round(nSL/2), :, :]
        plt.subplot(132)
        plt.imshow(np.abs(imgPlot), cmap='gray')
        plt.axis('off')
        plt.title(rawData['fileName'])
        plt.subplot(133)
        plt.imshow(np.angle(imgPlot), cmap='gray')
        plt.axis('off')
        
    # plot full image
    if nSL>1:
        plt.figure(4)
        img2d = np.zeros((nPoints[1], nPoints[0]*nPoints[2]))
        img2d = img2d+1j*img2d
        for ii in range(nPoints[2]):
            img2d[:, ii*nPoints[0]:(ii+1)*nPoints[0]] = img[ii, :, :]
        plt.imshow(np.abs(img2d), cmap='gray')
        plt.axis('off')
        plt.title(rawData['fileName'])
    
    plt.show()


#*********************************************************************************
#*********************************************************************************
#*********************************************************************************


if __name__ == "__main__":

    rare_standalone()

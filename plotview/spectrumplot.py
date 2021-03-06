"""
Plotview Spectrum (1D Plot)

@author:    David Schote
@contact:   david.schote@ovgu.de
@version:   2.0 (Beta)
@change:    13/07/2020

@summary:   Class for plotting a spectrum

@status:    Plots x and y data
@todo:      Implement more feature from pyqtgraph

"""
from PyQt5.QtWidgets import QLabel
from pyqtgraph import GraphicsLayoutWidget
from warnings import warn
from datetime import datetime 
import pyqtgraph as pg


class SpectrumPlot (GraphicsLayoutWidget):
    def __init__(self,
                 xData: list,
                 yData: list,
                 yData2:list, 
                 yData3:list, 
                 xlabel: str,
                 ylabel: str, 
                 title:str, 
                 ):
        super(SpectrumPlot, self).__init__()
        self.yData = yData
        self.yData2 = yData2
        self.yData3 = yData3
        
        if len(xData) != len(yData):
            warn("Length of x and y data does not match.")
            return
            
        dt = datetime.now()
        dt_string = dt.strftime("%d-%m-%Y_%H:%M:%S")

        self.plotitem = self.addPlot(row=0, col=0)
        self.plotitem.addLegend()
#        self.label = pg.TextItem(color=(200, 200, 200), anchor=(1, 1))
#        self.plotitem.addItem(self.label)
        self.plotitem.plot(xData, yData, pen=[255, 0, 0], name="Magnitude")
        self.plotitem.setXRange(xData[0], xData[len(xData)-1],padding=0)        
        self.label = pg.TextItem(color=(200, 200, 200), anchor=(0, 0))
        self.plotitem.addItem(self.label)
        if yData2 !=[]:
            self.plotitem.plot(xData, yData2, pen=[0, 255, 0], name="Real part")
            self.plotitem.plot(xData, yData3, pen=[0, 0, 255], name="Imaginary part")
        self.plotitem.setTitle("%s %s" % (title, dt_string))
        self.plotitem.setLabel('bottom', xlabel)
        self.plotitem.setLabel('left', ylabel)  
#        self.plotitem.setXRange(xData[0], xData[len(xData)-1],padding=0)
        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False)
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False)
        self.plotitem.addItem(self.crosshair_v, ignoreBounds=True)
        self.plotitem.addItem(self.crosshair_h, ignoreBounds=True)
        self.proxy = pg.SignalProxy(self.plotitem.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
 
    def mouseMoved(self, e):
        pos = e[0] ## using signal proxy turns original arguments into a tuple
        if self.plotitem.sceneBoundingRect().contains(pos):
            mousePoint = self.plotitem.vb.mapSceneToView(pos)
            self.label.setText("x=%0.4f, y(Magnitude)=%0.4f" % (mousePoint.x(), mousePoint.y()))
            self.crosshair_v.setPos(mousePoint.x())
            self.crosshair_h.setPos(mousePoint.y()) 


class Spectrum2DPlot(GraphicsLayoutWidget):
    def __init__(self,
                 Data:tuple, 
                 title:str
                 ):
        super(Spectrum2DPlot, self).__init__()

        dt = datetime.now()
        dt_string = dt.strftime("%d-%m-%Y_%H:%M:%S")
        image = pg.ImageItem(Data) 
        plotitem = self.addPlot(row=0, col=0)
        plotitem.addItem(image)
        plotitem.setTitle("%s %s" % (title, dt_string))

class Spectrum3DPlot(GraphicsLayoutWidget):
    def __init__(self,
                 Data:tuple, 
                 title:str
                 ):
        super(Spectrum3DPlot, self).__init__()

        dt = datetime.now()
        dt_string = dt.strftime("%d-%m-%Y_%H:%M:%S")
        image = pg.ImageItem(Data) 
        plotitem = self.addPlot(row=0, col=0)
        plotitem.addItem(image)
            
#        self.addItem(image)
#        vbox = image.getView()
#        vbox.addItem(pg.TextItem("%s %s" % (title, dt_string)))

"""
Operations Controller

@author:    David Schote
@contact:   david.schote@ovgu.de
@version:   2.0 (Beta)
@change:    13/06/2020

@summary:   TBD

@status:    Under development
@todo:      Extend construction of parameter section (headers, more categories, etc. )

"""
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QRegExp
from sessionmodes import defaultsessions
from sequencesnamespace import Namespace as nmspc
from sequencesnamespace import Tooltip_label as tlt_l
from sequencesnamespace import Tooltip_inValue as tlt_inV
from PyQt5.uic import loadUiType
from PyQt5.QtGui import QRegExpValidator


Parameter_Form, Parameter_Base = loadUiType('ui/inputparameter.ui')
#Parameter_FormG, Parameter_BaseG = loadUIType('ui/gradients.ui')


class SessionList(QComboBox):
    """
    Sequence List Class
    """
    def __init__(self, parent=None):
        """
        Initialization
        @param parent:  Mainviewcontroller (access to parameter layout)
        """
        super(SessionList, self).__init__(parent)

        # Add sequences to sequences list
        self.addItems(list(defaultsessions.keys()))
        
        if hasattr(parent, 'onSessionChanged'):
            parent.onSessionChanged.connect(self.triggeredSessionChanged)
            # Make parent reachable from outside __init__
            self.parent = parent
            self._currentSession = "Subject"
            self.setParametersUI("Subject")
#        self._currentSequence = None
        else:
            self._currentSession=parent.session
    
   
    def triggeredSessionChanged(self, session: str = None) -> None:
        self._currentSession = session
        self.setParametersUI(session)

    def getCurrentSession(self) -> str:
        return self._currentSession

    def setParametersUI(self, session: str = None) -> None:
        """
        Set input parameters from sequence object
        @param sequence:  Sequence object
        @return:    None
        """
        # Reset row layout for input parameters
        for i in reversed(range(self.parent.layout_parameters.count())):
            self.parent.layout_parameters.itemAt(i).widget().setParent(None)

        # Add input parameters to row layout
        inputwidgets: list = []

        if hasattr(defaultsessions[session], 'demographics'):
            dem_prop = defaultsessions[session].demographics
            inputwidgets += [self.generateLabelItem(nmspc.demographics)]
            inputwidgets += (self.generateWidgetsFromDict(dem_prop, session))
                               
        for item in inputwidgets:
            self.parent.layout_parameters.addWidget(item)
       
    @staticmethod
    def generateWidgetsFromDict(obj: dict = None, session: str = None) -> list:
        widgetlist: list = []
        for key in obj:
            widget = SessionParameter(key, obj[key], session)
            widgetlist.append(widget)
        return widgetlist

    @staticmethod
    def generateLabelItem(text):
        label = QLabel()
        label.setText(text)
        label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        return label
        
#    def generateTickItem(text):
#        Qx = QCheckBox("Qx")
#        Qx.setChecked(True)
        
#        Qy = QRadioButton('Qy')
#        Qy.setChecked(True)
#        Qz = QRadioButton('Qz')
#        return Qx

    def get_items(self, struct: dict = None, session:str = None) -> list:
        itemlist: list = []
        for key in list(struct.keys()):
            if type(struct[key]) == dict:
                itemlist.append(self.generateLabelItem(key))
                itemlist += self.get_items(struct[key])
            else:
                item = SessionParameter(key, struct[key], session)
                itemlist.append(item)

        return itemlist


class SessionParameter(Parameter_Base, Parameter_Form):
    """
    Operation Parameter Widget-Class
    """
    # Get reference to position in operation object
    def __init__(self, name, parameter, session):
        super(SessionParameter, self).__init__()
        self.setupUi(self)

        # Set input parameter's label and value
        self.session = session
        self.parameter = parameter
        self.label_name.setText(name)
        
        # Tooltips
        temp = vars(defaultsessions[self.session])
        for item in temp:
            label = 'nmspc.%s' %item
            res=eval(label)
            if (res == name):
                lab = 'tlt_l.%s' %(item)
                if (hasattr(tlt_l, item)):
                    res2=eval(lab)
                    self.label_name.setToolTip(res2)
                inV = 'tlt_inV.%s' %(item)
                if (hasattr(tlt_inV, item)):
                    res3 = eval(inV)
                    self.input_value.setToolTip(res3)    
                    
        self.input_value.setText(str(parameter[0]))
        
        
        # Connect text changed signal to getValue function
        self.input_value.textChanged.connect(self.get_value)
        
    def get_value(self) -> None:

        temp = vars(defaultsessions[self.session])
        for item in temp:
            lab = 'nmspc.%s' %(item)
            res=eval(lab)
            if (res == self.label_name.text()):
                t = type(getattr(defaultsessions[self.session], item))     
                inV = 'tlt_inV.%s' %(item)
                if (hasattr(tlt_inV, item)):
                    res3 = eval(inV)
                    if res3 == 'Value between 0 and 1':  
                        val=self.validate_input()
                        if val == 1:           
                            if (t is float): 
                                value: float = float(self.input_value.text())
                                setattr(defaultsessions[self.session], item, value)
                            elif (t is int): 
                                value: int = int(self.input_value.text())
                                setattr(defaultsessions[self.session], item, value)  
                else:
                    if (t is float): 
                        value: float = float(self.input_value.text())
                        setattr(defaultsessions[self.session], item, value)
                    elif (t is int): 
                        value: int = int(self.input_value.text())
                        setattr(defaultsessions[self.session], item, value)
                    elif (t is str):
                        value: str = str(self.input_value.text())
                        setattr(defaultsessions[self.session], item, value)
                    else:
                        v = self.input_value.text()
                        v=v.replace("[", "")
                        v=v.replace("]", "")
                        v2 = list(v.split(","))
                        if item=='shim':
                            value: list = list([float(v2[0]), float(v2[1]), float(v2[2])])
                        else:
                            if (v2[0] != ''):
                                value: list = list([int(v2[0]), int(v2[1]), int(v2[2])])
                        setattr(defaultsessions[self.session], item, value)

                
    def validate_input(self):
        reg_ex = QRegExp('^(?:0*(?:\.\d+)?|1(\.0*)?)$')
        input_validator = QRegExpValidator(reg_ex, self.input_value)
        self.input_value.setValidator(input_validator)
        state = input_validator.validate(self.input_value.text(), 0)
        if state[0] == QRegExpValidator.Acceptable:
            return 1
        else:
            return 0
        
    def set_value(self, key, value: str) -> None:
        print("{}: {}".format(self.label_name.text(), self.input_value.text()))
        

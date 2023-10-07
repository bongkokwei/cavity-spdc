import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor, QFont
from PyQt5.QtWidgets import (
    QApplication, 
    QMainWindow, 
    QSlider, 
    QLabel,
    QVBoxLayout,
    QHBoxLayout, 
    QWidget,
    QLineEdit     
)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from utils import *
from plot_utils import *
from cost_func import *

class SPDC_GUI(QMainWindow):
    def __init__(
            self,
            material_coeff = 1
        ):
        super().__init__()

        # Set the window title
        self.setWindowTitle("cavity SPDC")

        # default values
        self.param_dict = {
            'delta': 0.0005E-9*9,
            'zoom': 1,
            'numGrid':150+1,
            'pump_fwhm':0.00111E-9,
            'crystal_length':15e-3, # crystal class
            'domain_width':3.800416455460981E-6 , # crystal class
            'material_coeff': ktp().get_coeff(), # tuple of arrays
            'temperature':32.49,
            'central_pump':388E-9, 
            'central_signal':780.24E-9, 
            'R1':0.99, 
            'R2':0.8,
            'prop_loss':0.022
        }

        # self.ideal_jsa, _ = ideal_jsa(self.param_dict, 450E6)

        # Set the dark palette
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(palette)

        # Create a main widget to hold the layout
        widget = QWidget()
        self.setCentralWidget(widget)

        # Create a vertical layout to hold the widgets
        self.verticalLayout = QVBoxLayout()

        # Add a slider
        self.addParamSlider("Range (nm)", "zoom", min=1, max=200)
        self.addParamSlider("Num Grids", "numGrid", min=100, max=400)
        self.addParamSlider("Pump FWHM (pm)", "pump_fwhm", min=0.5E-12, max=5E-12)
        self.addParamSlider("Pump wavelength (nm)", "central_pump", min=380E-9, max=400E-9)
        self.addParamSlider("Signal wavelength (nm)", "central_signal", min=750E-9, max=800E-9)
        # self.addParamSlider("Crystal length (mm)", "crystal_length")
        # self.addParamSlider("Domain width (micron)", "domain_width")
        self.addParamSlider("Temperature (deg C)", "temperature", min=30, max=60)
        self.addParamSlider("R1", "R1", min=0, max=1.0)
        self.addParamSlider("R2", "R2", min=0, max=1.0)
        self.addParamSlider("Propagation loss", "prop_loss", min=0, max=1.0)
        
        self.fig = Figure(figsize=(7, 7), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.addPlot()
        widget.setLayout(self.verticalLayout)

        # End of constructor

##########################################################################################

    def addParamSlider(self, field, key, min=0, max=100, set_invert_appearance = False):
        '''
        field: Slider's label
        key: param_dict's keyword
        min, max: min and max for that param
        default: default param values
        '''
        slider_min, slider_max  = (0,500000)
        
        sliderLayout = QHBoxLayout()
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(slider_min)
        slider.setMaximum(slider_max)
        slider.setFixedWidth(300)

        default = self.paramToSilderVal(
            self.param_dict[key],
            slider_min, slider_max, 
            min, max
        )

        if set_invert_appearance:
            slider.setInvertedAppearance(True)
        slider.setValue(default)
        
        label = QLabel(field)
        label.setFont(QFont("Arial", 10))
        label.setFixedWidth(150)  # Set a fixed width for the label widget
        
        param_val = self.sliderToParamVal(
            default, 
            slider_min, slider_max, 
            min, max
        )

        lineEdit = QLineEdit(self.formatParamVal(param_val,key))
        lineEdit.setFixedWidth(50)

        # Connect sliders to plot update function
        lineEdit.returnPressed.connect(
            lambda : self.handleLineEditChanged(                
                key,
                lineEdit.text()
            )
        )
        slider.sliderReleased.connect(
            lambda : self.handleSliderRelease(
                key,
                slider.value(), 
                lineEdit,
                min, max,
                slider_min, slider_max
            )
        )

        slider.valueChanged.connect(
            lambda value: self.updateLineEdit(
                key,
                value, 
                lineEdit,
                min, max,
                slider_min, slider_max
            
            )
        )

        # Layout for a slider param
        sliderLayout.addWidget(label)
        sliderLayout.addWidget(slider)
        sliderLayout.addWidget(lineEdit)
        self.verticalLayout.addLayout(sliderLayout)

    def paramToSilderVal(
        self, 
        param,          
        slider_min, slider_max,
        min, max
    ): 
        ''' Takes in param value and returns integer slider value'''

        slider_val = int((param-min)/(max-min)*(slider_max-slider_min) + slider_min)
        return slider_val
    
    def sliderToParamVal(
        self, 
        value,         
        slider_min, slider_max,
        min, max
    ):
        param_val = (value)/(slider_max-slider_min)*(max-min) + min
        return param_val

    def formatParamVal(self, param_val, key):
        # formart param_val to update display
        if key == 'numGrid':
            self.param_dict[key] = int(float(param_val))
            param_str = str(int(float(param_val)))
        elif key == 'delta': # Check this case
            self.param_dict[key] = param_val*self.delta
            # No display for delta
        elif key == 'central_pump':
            self.param_dict[key] = float(param_val)
            param_str = f"{param_val*1E9:0.2f}" #convert to nm for display
        elif key == 'central_signal':
            self.param_dict[key] = float(param_val)
            param_str = f"{param_val*1E9:0.2f}" #convert to nm for display            
        elif key == 'pump_fwhm':
            self.param_dict[key] = float(param_val)
            param_str = f"{param_val*1E12:0.2f}" #convert to pm for display
        else:
            self.param_dict[key] = float(param_val)
            param_str = f"{param_val:0.2f}"

        return param_str
        
    def handleLineEditChanged(self, key, text):
        print(f"{key}: {text}")      

    def handleSliderRelease(
        self, 
        key, 
        value, 
        lineEdit, 
        min, max,
        slider_min, slider_max
    ):
        ''' The plot will only update after slider is released'''
        self.updateLineEdit(
            key, value, lineEdit, 
            min, max,
            slider_min, slider_max
        )
        self.addPlot()

    def updateLineEdit(
        self, 
        key, 
        value, 
        lineEdit, 
        min, max,
        slider_min, slider_max
    ):
        '''Only the text box value is changed'''
        param_val = self.sliderToParamVal(
            value,
            slider_min, slider_max,
            min, max
        )

        param_str = self.formatParamVal(param_val, key)

        lineEdit.setText(param_str) # this will trigger handleLineEditChanged

    def addPlot(self):
        self.verticalLayout.addWidget(self.canvas)
        self.fig.clear()
        pef, pmf, cavity_response, jsa, x, y = calculate_jsa(self.param_dict)
        plot_jsa(self.fig, jsa, pmf, pef, cavity_response, x, y)

        self.canvas.draw_idle()

if __name__ == "__main__":


    # Create the application
    app = QApplication(sys.argv)

    # Set the application style to Fusion
    app.setStyle("Fusion")

    # Create the main window
    window = SPDC_GUI()
    window.show()

    # Run the event loop
    sys.exit(app.exec_())

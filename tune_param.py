from utils import *

param_dict = {
    'delta': 0.0005E-9*9,
    'numGrid':150,
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

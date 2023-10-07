import numpy as np
import matplotlib.pyplot as plt

#############################################################################
# GLOBAL VARIABLE
PI = np.pi
C = 299792458
PERMITTIVITY = 8.85418782e-12
EFFICICIENCYFACTOR = 1.0
CHI2EFF = 1e-8
HBAR = 1.05457173e-34

#############################################################################
""" crystal and material class"""


class ktp:
    def __init__(self):
        """Material properties"""
        self.coeff_y = [2.09930, 0.922683, 0.0467695, 0, 0, 0.0138408]
        self.coeff_z = [2.12725, 1.18431, 5.14852e-2, 0.6603, 100.00507, 9.68956e-3]

        self.temperature_coeff_y = [
            [6.2897e-6, 6.3061e-6, -6.0629e-6, 2.6486e-6],
            [-0.14445e-8, 2.2244e-8, -3.5770e-8, 1.3470e-8],
        ]
        self.temperature_coeff_z = [
            [9.9587e-6, 9.9228e-6, -8.9603e-6, 4.1010e-6],
            [-1.1882e-8, 10.459e-8, -9.8136e-8, 3.1481e-8],
        ]

    def get_coeff(self):
        return (
            self.coeff_y,
            self.temperature_coeff_y,
            self.coeff_z,
            self.temperature_coeff_z,
        )


class sellmeier:
    # coeff must be an array of length 6
    # add temperature dependence
    def __init__(self, coeff, temp_coeff=np.zeros((2, 4)), temperature=25):
        self.coeff = coeff
        self.temp_coeff = temp_coeff
        self.temperature = temperature
        self.n = lambda x: np.sqrt(
            self.coeff[0]
            + self.coeff[1] * x**2 / (x**2 - self.coeff[2])
            + self.coeff[3] * x**2 / (x**2 - self.coeff[4])
            - self.coeff[5] * x**2
        )

    def get_coeff(self):
        return self.coeff

    def get_refractive_index(self):
        return lambda x: self.n(x) + self.temp_dependence()(x)

    def temp_dependence(self):
        # temp dependence for KTP
        dn1 = lambda x: (
            self.temp_coeff[0][0] / x**0
            + self.temp_coeff[0][1] / x**1
            - self.temp_coeff[0][2] / x**2
            + self.temp_coeff[0][3] / x**3
        )
        dn2 = lambda x: (
            self.temp_coeff[1][0] / x**0
            + self.temp_coeff[1][1] / x**1
            - self.temp_coeff[1][2] / x**2
            + self.temp_coeff[1][3] / x**3
        )
        delta_n = (
            lambda x: dn1(x) * (self.temperature - 25)
            + dn2(x) * (self.temperature - 25) ** 2
        )
        return delta_n


class crystal:
    def __init__(self, domain_width, poling_config):
        self.domain_width = domain_width
        self.poling_config = poling_config
        self.num_domain = len(poling_config)

    def set_domain_width(self, lc):
        self.domain_width = lc

    def set_poling_config(self, config):
        self.poling_config = config
        self.num_domain = len(config)

    def get_domain_width(self):
        return self.domain_width

    def get_num_domain(self):
        return self.num_domain

    def get_poling_config(self):
        return self.poling_config


#############################################################################
def freq_bw(central_wavelength, wavelength_bandwith):
    return C / central_wavelength**2 * wavelength_bandwith


def wavelength_bw(central_wavelength, freq_bandwidth):
    return freq_bandwidth / C * central_wavelength**2


def wavelength2freq(wavelength):
    return C / wavelength


def freq2wavelength(freq):
    return C / freq


def wavelength2angfreq(wavelength):
    return (2 * PI * C) / wavelength


def angfreq2wavelength(angFreq):
    return 2 * PI * C / angFreq


def wavelength2freq(wavelength):
    return C / wavelength


def freq2wavelength(freq):
    return C / freq


def fwhm2sigma(fwhm, wavelength):
    # approximation to convert FWHM in wavelength to FWHM in angular frequency
    # Using FWHM in ang freq to calc standard deviation for a Gaussian

    ang_fhwm = (2 * PI) * (C / wavelength**2) * fwhm  # angular freq FWHM
    sigma = ang_fhwm / (2 * np.sqrt(np.log(2)))  # FWHM to gauss sigma
    return ang_fhwm, sigma


def diff(func, h=1e-8):
    return lambda x: (func(x + h) - func(x - h)) / (2 * h)


def group_index(n):
    """
    Calculate the group index of the material,
    Input is the refractive index as a function of wavelength in microns
    """
    return lambda x: n(x) - x * diff(n)(x)


def delta_k_meshgrid(
    signal, idler, central_signal, central_idler, dk0, inverse_group_vel
):
    angfreq_signal = wavelength2angfreq(signal)
    angfreq_idler = wavelength2angfreq(idler)
    angfreq_signal_central = wavelength2angfreq(central_signal)
    angfreq_idler_central = wavelength2angfreq(central_idler)

    delta_signal = angfreq_signal - angfreq_signal_central
    delta_idler = angfreq_idler - angfreq_idler_central

    dk = (
        dk0
        + (inverse_group_vel["pump"] - inverse_group_vel["signal"]) * (delta_signal)
        + (inverse_group_vel["pump"] - inverse_group_vel["idler"]) * (delta_idler)
    )
    return dk


def inverse_group_vel(central_wavelengths, ny, nz):
    pump, signal, idler = central_wavelengths

    inverse_group_vel_pump = group_index(ny)(pump * 1e6) / C
    inverse_group_vel_signal = group_index(ny)(signal * 1e6) / C
    inverse_group_vel_idler = group_index(nz)(idler * 1e6) / C

    return {
        "pump": inverse_group_vel_pump,
        "signal": inverse_group_vel_signal,
        "idler": inverse_group_vel_idler,
    }


def delta_k_central(central_wavelengths, ny, nz):
    pump, signal, idler = central_wavelengths

    k_pump = wavelength2angfreq(pump) * ny(pump * 1e6) / C
    k_signal = wavelength2angfreq(signal) * ny(signal * 1e6) / C
    k_idler = wavelength2angfreq(idler) * nz(idler * 1e6) / C
    dk0 = k_pump - k_signal - k_idler

    return dk0


def pmf_axis(central_wavelengths, ny, nz):
    inv_grp_vel = inverse_group_vel(central_wavelengths, ny, nz)

    axis = np.degrees(
        np.arctan(
            -(inv_grp_vel["pump"] - inv_grp_vel["idler"])
            / (inv_grp_vel["pump"] - inv_grp_vel["signal"])
        )
    )

    return axis


def pump_env_func(signal_range, idler_range, central_wavelengths, fwhm, shape="gauss"):
    pump, signal, idler = central_wavelengths

    angfreq_signal = wavelength2angfreq(signal_range)
    angfreq_idler = wavelength2angfreq(idler_range)

    angfreq_signal_central = wavelength2angfreq(signal)
    angfreq_idler_central = wavelength2angfreq(idler)

    ang_fwhm, sigma_pef = fwhm2sigma(fwhm, pump)

    delta_signal = angfreq_signal - angfreq_signal_central
    delta_idler = angfreq_idler - angfreq_idler_central

    # The sech and Gaussian PEFs have equal width when tau ~ 0.712sigmaPEF
    if shape == "gauss":
        PEF = np.exp(-((delta_signal + delta_idler) ** 2) / (2 * sigma_pef**2))
    elif shape == "sech":
        PEF = (
            np.cosh(0.5 * PI * (1.122 / ang_fwhm) * (delta_signal + delta_idler)) ** -1
        )
    return PEF


def phase_matching(
    sellmeier_y,
    sellmeier_z,
    crystal,  # nonlinear crystal properties
    signal_range,
    idler_range,
    central_wavelengths,
):
    """Aggie's derivation"""

    central_pump, central_signal, central_idler = central_wavelengths
    lc = crystal.get_domain_width()
    num_domain = crystal.get_num_domain()
    poling_config = crystal.get_poling_config()
    ny = sellmeier_y.get_refractive_index()
    nz = sellmeier_z.get_refractive_index()
    crystal_length = lc * num_domain
    chi_0 = 1  # nonlinearity profile

    zn = (np.arange(num_domain) + 0.5) * lc

    dk = delta_k_meshgrid(
        signal_range,
        idler_range,
        central_signal,
        central_idler,
        delta_k_central(central_wavelengths, ny, nz),
        inverse_group_vel(central_wavelengths, ny, nz),
    )

    constant = lc / crystal_length * chi_0
    sinc = np.sinc(dk * lc / 2)
    integral = np.einsum(
        "ijk -> ij",
        np.einsum(
            "ijk, k -> ijk",
            np.exp(np.einsum("ij, k -> ijk", -1j * dk, zn)),
            poling_config,
        ),
    )
    pmf = constant * sinc * integral

    return pmf


def cavity(
    wavelength, crystal, refractive_index, reflectivity_1, reflectivity_2, prop_loss
):
    # Cavity response function

    crystal_length = crystal.get_domain_width() * crystal.get_num_domain()
    angfreq = wavelength2angfreq(wavelength)
    phase = 2 * angfreq * refractive_index(wavelength * 1e6) * crystal_length / C
    cavity_response = (
        np.sqrt((1 - reflectivity_1) * (1 - reflectivity_2))
        * np.exp(-prop_loss * crystal_length / 2)
    ) / (
        1
        - np.sqrt(reflectivity_1 * reflectivity_2)
        * np.exp(-prop_loss * crystal_length)
        * np.exp(1j * phase)
    )

    return cavity_response


def get_purity(jsa):
    # From the Schmidt decomposition, we can see that w is entangled
    # if and only if w has Schmidt rank strictly greater than 1
    # https://en.wikipedia.org/wiki/Schmidt_decomposition

    u, s, vh = np.linalg.svd(jsa, full_matrices=True)
    s /= np.sqrt(np.sum(s**2))  # Normalise Schmidt coefficients

    # From Dosseva et al. 2016, pg 3
    entropy = -np.sum(np.abs(s) ** 2 * np.log(np.abs(s) ** 2))
    purity = np.sum(s**4)
    return purity, entropy


def generate_yz_sellmeier(material_coeff, temperature):
    """Create material class"""
    coeff_y, temp_coeff_y, coeff_z, temp_coeff_z = material_coeff
    sellmeier_y = sellmeier(coeff_y, temp_coeff_y, temperature)
    sellmeier_z = sellmeier(coeff_z, temp_coeff_z, temperature)

    return sellmeier_y, sellmeier_z


def calculate_jsa(param_dict):
    material_coeff = param_dict["material_coeff"]  # tuple of arrays

    try:
        delta = param_dict["delta"] * param_dict["zoom"]
    except KeyError:
        delta = param_dict["delta"]

    numGrid = param_dict["numGrid"]
    pump_fwhm = param_dict["pump_fwhm"]
    crystal_length = param_dict["crystal_length"]  # crystal class
    domain_width = param_dict["domain_width"]  # crystal class
    temperature = param_dict["temperature"]
    central_pump = param_dict["central_pump"]
    central_signal = param_dict["central_signal"]
    reflectivity_1 = param_dict["R1"]
    reflectivity_2 = param_dict["R2"]
    prop_loss = param_dict["prop_loss"]

    """Calculating the idler central wavelength"""
    central_idler = freq2wavelength(
        wavelength2freq(central_pump) - wavelength2freq(central_signal)
    )
    central_wavelengths = (central_pump, central_signal, central_idler)

    """ Countour axis range """
    signal_wavelength = np.linspace(
        central_signal - delta, central_signal + delta, numGrid
    )
    idler_wavelength = np.linspace(
        central_idler - delta, central_idler + delta, numGrid
    )
    # There is a need to reverse the idlerWavelength array, or else the origin will
    # start on the top left corner of grid instead of bottom left.
    idler_wavelength = idler_wavelength[::-1]
    signal_range, idler_range = np.meshgrid(signal_wavelength, idler_wavelength)

    """Create material class"""
    sellmeier_y, sellmeier_z = generate_yz_sellmeier(material_coeff, temperature)

    ny = sellmeier_y.get_refractive_index()  # returns function
    nz = sellmeier_z.get_refractive_index()  # returns function

    pef = pump_env_func(signal_range, idler_range, central_wavelengths, pump_fwhm)

    """ Create crystal to calculate pmf"""
    num_domain = int(crystal_length / domain_width)
    poling = [(-1) ** k for k in range(num_domain)]
    ppktp = crystal(domain_width, poling)

    pmf = phase_matching(
        sellmeier_y,
        sellmeier_z,
        ppktp,  # nonlinear crystal properties
        signal_range,
        idler_range,
        central_wavelengths,
    )

    signal_response = cavity(
        signal_wavelength, ppktp, ny, reflectivity_1, reflectivity_2, prop_loss
    )

    idler_response = cavity(
        idler_wavelength, ppktp, nz, reflectivity_1, reflectivity_2, prop_loss
    )

    cavity_response = np.einsum("i, j -> ij", signal_response, idler_response)
    jsa = pmf * cavity_response * pef

    return (pef, pmf, cavity_response, jsa, signal_wavelength, idler_wavelength)


def jsa_marginals(jsa, axis=1):
    return np.sum(jsa, axis)


def bandpass_filter(jsa, x, center, bandwidth):
    lower = center - bandwidth / 2
    upper = center + bandwidth / 2
    jsa_filtered = np.copy(jsa)
    jsa_filtered[:, (x < lower) | (x > upper)] = 0
    return jsa_filtered


#############################################################################

if __name__ == "__main__":
    param_dict = {
        "delta": 0.0005e-9 * 9,
        "numGrid": 200,
        "pump_fwhm": 0.00111e-9,
        "crystal_length": 15e-3,  # crystal class
        "domain_width": 3.800416455460981e-6,  # crystal class
        "material_coeff": ktp().get_coeff(),  # tuple of arrays
        "temperature": 32.49,
        "central_pump": 388e-9,
        "central_signal": 780.24e-9,
        "R1": 0.99,
        "R2": 0.8,
        "prop_loss": 0.022,
    }

    ###############################################################################

    # pef, pmf, cavity_response, jsa, x, y = calculate_jsa(param_dict)
    central_idler = freq2wavelength(
        wavelength2freq(param_dict["central_pump"])
        - wavelength2freq(param_dict["central_signal"])
    )

    central_wavelengths = (
        param_dict["central_pump"],
        param_dict["central_signal"],
        central_idler,
    )

    """ Countour axis range """
    signal_wavelength = np.linspace(
        param_dict["central_signal"] - param_dict["delta"],
        param_dict["central_signal"] + param_dict["delta"],
        param_dict["numGrid"],
    )
    idler_wavelength = np.linspace(
        central_idler - param_dict["delta"],
        central_idler + param_dict["delta"],
        param_dict["numGrid"],
    )[::-1]

    signal_range, idler_range = np.meshgrid(signal_wavelength, idler_wavelength)

    pef = pump_env_func(
        signal_range,
        idler_range,
        central_wavelengths,
        param_dict["pump_fwhm"],
        shape="gauss",
    )

    plt.contourf(signal_range, idler_range, pef, levels=100)
    plt.show()

###############################################################################
# ideal_lc = PI/np.abs(delta_k_central(central_wavelengths, ny, nz))*1E6
# purity, entropy = get_purity(jsa)

# print(f"Crystal's length: {num_domain*domain_width*1E3} mm")
# print(f"PMF Aixs: {pmf_axis(central_wavelengths, ny, nz)} degree")
# print(f"Ideal domain width: {ideal_lc} micron")
# print(f"Purity of heralded single photon: {purity}")
# print(f"Entropy of biphoton state: {entropy}")

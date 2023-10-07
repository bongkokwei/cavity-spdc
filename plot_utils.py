"""
Plotting script for all things related to joint spectrum
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from utils import *


def plot_jsa(fig, jsa, pmf, pef, cavity_response, signal, idler):
    # Calculate the jsa purity and entropy
    purity, entropy = get_purity(jsa)

    # Create a GridSpec object with square subplots
    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

    # Create the subplots
    jsa_ax = fig.add_subplot(gs[0])
    pmf_ax = fig.add_subplot(gs[1])
    sig_ax = fig.add_subplot(gs[2])
    idl_ax = fig.add_subplot(gs[3])

    sig_marginal = jsa_marginals(np.abs(jsa), 0)
    idl_marginal = jsa_marginals(np.abs(jsa), 1)

    mid_sig = (np.max(signal * 1e9) + np.min(signal * 1e9)) / 2
    mid_idl = (np.max(idler * 1e9) + np.min(idler * 1e9)) / 2

    # Plot the data on the subplot
    jsa_ax.contourf(signal * 1e9, idler * 1e9, np.abs(cavity_response), levels=50)
    jsa_ax.contour(signal * 1e9, idler * 1e9, pef, levels=2)
    jsa_ax.axvline(x=mid_sig, color="k", linestyle="--")

    pmf_ax.contourf(signal * 1e9, idler * 1e9, np.abs(pmf), levels=50)
    pmf_ax.contour(signal * 1e9, idler * 1e9, pef, levels=2)

    sig_ax.plot(signal * 1e9, sig_marginal, label="Signal Marginal", color="r")
    idl_ax.plot(idler * 1e9, idl_marginal, label="Idler Marginal")

    jsa_ax.set_xticklabels([])
    jsa_ax.set_yticklabels([])
    jsa_ax.set_title("Joint Spectral Amplitude", fontsize=10)
    jsa_ax.set_aspect("equal")
    jsa_ax.text(
        0.95,
        0.05,
        f"Purity: {purity * 1e2:0.2f}%",
        fontsize=12,
        ha="right",
        va="bottom",
        color="white",
        transform=jsa_ax.transAxes,
    )
    pmf_ax.set_xticklabels([])
    pmf_ax.set_yticklabels([])
    pmf_ax.set_title("Phase-matching function", fontsize=10)
    pmf_ax.set_aspect("equal")
    idl_ax.yaxis.tick_right()
    sig_ax.legend()
    idl_ax.legend()

    sig_ax.axvline(x=mid_sig, color="k", linestyle="--")
    idl_ax.axvline(x=mid_idl, color="k", linestyle="--")


if __name__ == "__main__":
    param_dict = {
        "delta": 2.3e-9,
        "numGrid": 500,
        "pump_fwhm": 10e-9 / 1000,
        "crystal_length": 5e-3,  # crystal class
        "domain_width": 3.8e-6,  # crystal class
        "material_coeff": ktp().get_coeff(),  # tuple of arrays
        "temperature": 31,  # 32.495
        "central_pump": 388e-9,
        "central_signal": 780e-9,
        "R1": 0.99,
        "R2": 0.8,
        "prop_loss": 0.022,
    }

    ###############################################################################
    # bandwidth = wavelength_bw(780.24E-9, 800E6)
    pef, pmf, cavity_response, jsa, x, y = calculate_jsa(param_dict)
    # jsa_filtered = bandpass_filter(jsa, x, 780.24E-9, bandwidth)
    ###############################################################################
    fig = plt.figure(figsize=plt.figaspect(1))
    plot_jsa(fig, jsa, pmf, pef, cavity_response, x, y)
    plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script processes ingests the HWO yaml files, 
collates parameters relevant to exposure time calculation,
and outputs neat files and dictionaries compatable with AYO and EDITH.

Author: Miles Currie, NASA Goddard
Created: Dec 18, 2024
"""

import numpy as np
import astropy.units as u
import os
import yaml
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pickle


def generate_wavelength_grid(lammin, lammax, R):
    # working in log space to maintain a constant resolving power across the grid

    lammin_log = np.log(lammin)
    lammax_log = np.log(lammax)
    dlam_log = np.log(1 + 1/R) # step size in log space

    lam_log = np.arange(lammin_log, lammax_log, dlam_log)
    lam = np.exp(lam_log)
    return lam

def load_yaml(fl_path):
    # fl_path assumed to originate from SCI_ENG_DIR environment variable location
    # For example, in my .zshrc I have this line:
    # export SCI_ENG_DIR=/Users/mhcurrie/science/packages/Sci-Eng-Interface/hwo_sci_eng
    with open(os.getenv("SCI_ENG_DIR") + fl_path, "r") as fl:
        fl_dict = yaml.load(fl, Loader=yaml.SafeLoader)
    return fl_dict

def interp_arr(old_lam, old_vals, new_lam):
    old_lam = old_lam.to(new_lam.unit)
    assert old_lam.unit == new_lam.unit
    interp_func = interp1d(old_lam, old_vals, kind="linear", bounds_error=False, fill_value=0)
    new_vals = interp_func(new_lam)
    new_vals = np.clip(new_vals, 0, None) #clip negative values to 0
    return new_vals

def write_ayo_file(tele, inst, det, fl_name="output.ayo"):
    #####
    # This is a work in progress... basically the goal is to output an ayo file identical to the one
    # that yaml2ayo.pro outputs...
    # eventually I will need to ask Chris for this or run it myself?? 
    #####

    nf = None

    #--- GENERAL PARAMETERS ---
    AYO_version = 'v16' #version of the AYO code to run
    total_survey_time = 2.0         #(years) {scalar} total observation time including overheads to be devoted to survey

    #--- CORONAGRAPH INPUT FILES ---
    #AYO can handle up to 8 different coronagraphs, 4 for detections and 4 for spec char
    coronagraph1 = coronagraphdir #{scalar} name of the folder where detection coronagraph files are located
    coronagraph2 = '' #{scalar} name of the folder where detection coronagraph files are located
    coronagraph3 = '' #{scalar} name of the folder where detection coronagraph files are located
    coronagraph4 = '' #{scalar} name of the folder where detection coronagraph files are located
    sc_coronagraph1 = coronagraphdir #{scalar} name of the folder where spec char coronagraph files are located
    sc_coronagraph2 = '' #{scalar} name of the folder where spec char coronagraph files are located
    sc_coronagraph3 = '' #{scalar} name of the folder where spec char coronagraph files are located
    sc_coronagraph4 = '' #{scalar} name of the folder where spec char coronagraph files are located

    #--- CORONAGRAPH PERFORMANCE MODIFIERS ---
    #The following are OPTIONAL inputs that override what is in the coronagraph YIP (or provides the value if it's not in the YIP).
    #THESE SHOULD TYPICALLY BE COMMENTED OUT.
    #Modifying the YIP may be necessary to perform sensitivity studies
    #nrolls1 = 1 #{scalar} number of roll angles to assume for coronagraph1, if specified this overrides what is in the coronagraph YIP header
    #nrolls2 = 1 #{scalar} number of roll angles to assume for coronagraph2, if specified this overrides what is in the coronagraph YIP header
    #nrolls3 = 1 #{scalar} number of roll angles to assume for coronagraph3, if specified this overrides what is in the coronagraph YIP header
    #nrolls4 = 1 #{scalar} number of roll angles to assume for coronagraph4, if specified this overrides what is in the coronagraph YIP header
    #sc_nrolls1 = 1 #{scalar} number of roll angles to assume for sc_coronagraph1, if specified this overrides what is in the coronagraph YIP header
    #sc_nrolls2 = 1 #{scalar} number of roll angles to assume for sc_coronagraph2, if specified this overrides what is in the coronagraph YIP header
    #sc_nrolls3 = 1 #{scalar} number of roll angles to assume for sc_coronagraph3, if specified this overrides what is in the coronagraph YIP header
    #sc_nrolls4 = 1 #{scalar} number of roll angles to assume for sc_coronagraph4, if specified this overrides what is in the coronagraph YIP header
    #coro_bw_multiplier = 1.0 #{scalar} multiplies the coronagraph design BW by this factor, shows up in coronagraph plots--to take advantage of this you may have to adjust SR (see below)
    #coro_contrast_multiplier = 1.0 #{scalar} multiplies the coronagraph design contrast by this factor (stellar_intens gets multiplied), prior to evaluating noise floor
    #coro_throughput_multiplier = 1.0 #{scalar} multiplies the coronagraph design throughput by this factor (stellar_intens, offax_psf, and skytrans get multiplied), shows up in coronagraph plots
    #coro_pixscale_multiplier = 1.0 #{scalar} multiplies the coronagraph design pixelscale by this factor, shows up in coronagraph plots
    #sc_coro_bw_multiplier = 1.0 #{scalar} multiplies the coronagraph design BW by this factor, shows up in coronagraph plots
    #sc_coro_contrast_multiplier = 1.0 #{scalar} multiplies the coronagraph design contrast by this factor (stellar_intens gets multiplied), prior to evaluating noise floor, shows up in coronagraph plots
    #sc_coro_throughput_multiplier = 1.0 #{scalar} multiplies the coronagraph design core throughput by this factor (stellar_intens, offax_psf, and skytrans get multiplied), shows up in coronagraph plots
    #sc_coro_pixscale_multiplier = 1.0 #{scalar} multiplies the coronagraph design pixelscale by this factor, shows up in coronagraph plots
    #corosensitivity=1 #{scalar} setting this flag to 1 evaluates sensitivity of yield to coro params, but MULTIPLIES RUNTIME BY 5x

    #--- CORONGRAPH PARAMETERS AND PERFORMANCE CUTS ---
    raw_contrast_floor = 0. #{scalar} raw contrast is set to larger of design value and this, 0 means impose no floor
    IWA = 1.0  #(lambda/D) {scalar} hard IWA cutoff regardless of coronagraph design (making this non-zero helps reduce run time of code)
    OWA = 60. #(lambda/D) {scalar} hard OWA cutoff regardless of coronagraph design

    #--- PSF SUBTRACTION / NOISE FLOOR ---
    CRb_multiplier = 2 #{scalar} all background terms multiplied by this. 1=perfect model-based PSF subtraction, 2=ADI (2 probably also roughly appropriate for RDI)
    #Only use one of the following two options to specify how the noise floor is calculate
    if nf is not None:
        noisefloor_contrast = nf
    else:
        #noisefloor_contrast = 3e-12  #{scalar} 1-sigma post-processed noise floor contrast, uniform over field of view and indepent of raw contrast (cannot specify this and noisefloor_PPF at same time)
        noisefloor_PPF = 30.0 #{scalar} the post-processing factor that sets the 1-sigma noise floor contrast, noisefloor = raw_contrast / PPF (cannot specify this and nosiefloor_contrast at same time)


    #--- TARGET LIST CUTS ---
    target_vmag_cut = 9. #(mags) {scalar} maximum magnitude at V band, removes fainter targets, speeds up calculations but runs the risk of artificially limiting results
    target_distance_cut = 30. #(pc) {scalar} maximum distance, removes more distant targets, speeds up calculations but runs the risk of artificially limiting results

    #--- MISC ---
    nexozodis = 3.0 #("zodis") {scalar} exozodi level of all stars
    Tcontam = 0.95 #{scalar} effective throughput factor to budget for contamination, scalar value only
    temperature = 290.0 #(K) {scalar} temperature of warm optics
    photap_rad = 0.85    #(l/D) {scalar} photometric aperture radius used for plots and some basic exp time estimates, recommended = 0.7 l/D_inscribed ~ 0.85 l/D
    sc_photap_rad = 0.85 #(l/D) {scalar} photometric aperture radius used for plots and some basic exp time estimates, recommended = 0.7 l/D_inscribed ~ 0.85 l/D
    PSF_trunc_ratio = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00] #{array} PSF truncation ratios defining possible photometric apertures to use for all exp times, using a single value (recommended [0.35]) will speed up calculations

    #--- DETECTION OBSERVATIONS ---
    minvisits =                6 #{scalar} minimum # of detection obs. per star
    broadband = 		   1 #{scalar} setting to 1 overrides SR below and sets the BW of the observation to equal that of the coronagraph (usually this = 1 for detections)
    td_limit =                  5.184e6   #(s) {scalar} detection time upper limit, 2 months = 5.184e6 s
    toverhead_fixed =     8381.30 #(s) {scalar} static overhead for all detection observations, slew + settle + initial digging of dark hole  
    toverhead_multi =     1.1       #{scalar} dynamic overhead for all detection observations, i.e. a "tax" on the science exposure time, time to touch up dark hole
    #lambda =              [  0.35,  0.40,  0.45,  0.50,  0.55,  0.60,  0.65,  0.70,  0.75,  0.80,  0.85,  0.90,  0.95,  1.00] #(microns) {array of length NLD} possible detection wavelengths
    #lambda =              [ 0.50] #(microns) {array of length NLD} possible detection wavelengths
    SR     =              5. #{scalar or array of length NLD} desired spectral resolution for detections, code uses the smaller of this bandpass and the coronagraph's design
    SNR    =              7. #{scalar or array of length NLD} minimum SNR for detections
    det_tread =           1000. #(s) {scalar or array of length NLD} detector read times for detection observations

    #--- CHARACTERIZATION OBSERVATIONS ---
    tsc_limit =              5.184e6   #(s) {scalar} spec char time upper limit, 2 months = 5.184e6 s
    sc_broadband = 	         0  #{scalar} setting to 1 overrides sc_SR below and sets the BW of the observation to equal that of the coronagraph (usually this = 0 for characterizations)
    sc_toverhead_fixed =     8381.30 #(s) {scalar} static overhead for all spec char observations, slew + settle + initial digging of dark hole  
    sc_toverhead_multi =     1.1     #{scalar} dynamic overhead for all spec char observations, i.e. a "tax" on the science exposure time, time to touch up dark hole
    sc_det_tread =           1000.      #(s) {scalar or array of length NLD} detector read times for detection observations
    if char == None:
        assert False, "Must specify char: 'h2o', 'o2', 'o3', 'ch4', 'co', or 'co2'."
    elif char.lower() not in ['h2o','o2','o3','ch4','co','co2']:
        assert False, "Must specify char: 'h2o', 'o2', 'o3', 'ch4', 'co', or 'co2'."

    if char.lower() == "h2o":
        sc_lambda =              [   0.7271, 0.7458, 0.7645, 0.7832, 0.8019, 0.8206, 0.8393, 0.8591, 0.8778, 0.8965, 0.9152, 0.9339, 0.9526, 0.9713, 1.0000] #(microns) {array of length NLSC} possible spec char wavelengths
        sc_SR =                  [      140,    140,    140,    140,    140,    140,    140,    140,    140,    140,    140,    140,    140,    140,    140] #{array of length NLSC} desired spectral resolution for spec char
        sc_SNR =                 [       14,     13,     13,     12,     12,     12,     10,     10,     10,     10,     11,     11,      9,      8,      5] #{array of length NLSC} required SNR for spec char
        #sc_lambda =              [   1.0000] #(microns) {array of length NLSC} possible spec char wavelengths
        #sc_SR =                  [   140] #{array of length NLSC} desired spectral resolution for spec char
        #sc_SNR =                 [   5] #{array of length NLSC} required SNR for spec char

    if char.lower() == "o2":
        sc_lambda =              [   0.76]*1.1 #(microns) {array of length NLSC} possible spec char wavelengths
        sc_SR =                  [   140.] #{array of length NLSC} desired spectral resolution for spec char
        sc_SNR =                 [    10.] #{array of length NLSC} required SNR for spec char
    if char.lower == "o3":
        sc_lambda =              [   0.2, 0.25, 0.3] #(microns) {array of length NLSC} possible spec char wavelengths
        sc_SR =                  [     7,    7,   7] #{array of length NLSC} desired spectral resolution for spec char
        sc_SNR =                 [    10,   10,  10] #{array of length NLSC} required SNR for spec char
    if char.lower() == "ch4":
        sc_lambda =              [    0.89]*1.1 #(microns) {array of length NLSC} possible spec char wavelengths
        sc_SR =                  [     140] #{array of length NLSC} desired spectral resolution for spec char
        sc_SNR =                 [      10] #{array of length NLSC} required SNR for spec char
    if char.lower() == "co":
        sc_lambda =              [     1.6] #(microns) {array of length NLSC} possible spec char wavelengths
        sc_SR =                  [      70] #{array of length NLSC} desired spectral resolution for spec char
        sc_SNR =                 [      10] #{array of length NLSC} required SNR for spec char
    if char.lower() == "co2":
        sc_lambda =              [     1.6] #(microns) {array of length NLSC} possible spec char wavelengths
        sc_SR =                  [      70] #{array of length NLSC} desired spectral resolution for spec char
        sc_SNR =                 [      20] #{array of length NLSC} required SNR for spec char
    with open(fl_name, 'w') as f:
        f.write(";This is an input file for AYO\n")
        f.write(";Comments are preceded with a # or a ;\n")
        f.write(";In the comments below, ( ) indicates the units and { } indicates the data format\n")
        f.write("\n")
        f.write(";--- GENERAL PARAMETERS ---\n")
        f.write(f"AYO_version = '{ayo_version}' ;version of the AYO code to run, default if not supplied is 'v15'\n")
        f.write(f"D = {D} ;(m) {scalar} circumscribed diameter of telescope\n")
        f.write(f"total_survey_time = {total_survey_time} ;(years) {scalar} total observation time including overheads\n")
        f.write("\n")
        f.write(";--- CORONAGRAPH INPUT FILES ---\n")
        for i in range(1, 5):
            f.write(f"coronagraph{i} = '{params[f'coronagraph{i}']}' ;{scalar} name of the folder where detection coronagraph files are located\n")
        for i in range(1, 5):
            f.write(f"sc_coronagraph{i} = '{params[f'sc_coronagraph{i}']}' ;{scalar} name of the folder where spec char coronagraph files are located\n")
        f.write("\n")
        
        f.write(";--- CORONAGRAPH PERFORMANCE MODIFIERS ---\n")
        f.write(";The following are OPTIONAL inputs that override what is in the coronagraph YIP\n")
        f.write(";These should typically be commented out.\n")
        f.write("\n")
        
        f.write(";--- CORONAGRAPH PARAMETERS AND PERFORMANCE CUTS ---\n")
        f.write(f"nchannels = {params['nchannels']} ;{scalar} number of parallel detection channels\n")
        f.write(f"raw_contrast_floor = {params['raw_contrast_floor']} ;{scalar} raw contrast floor\n")
        f.write(f"IWA = {params['IWA']}  ;(lambda/D) {scalar} hard IWA cutoff\n")
        f.write(f"OWA = {params['OWA']} ;(lambda/D) {scalar} hard OWA cutoff\n")
        f.write("\n")
        
        f.write(";--- DETECTION OBSERVATIONS ---\n")
        f.write(f"minvisits = {params['minvisits']} ;{scalar} minimum # of detection obs. per star\n")
        f.write(f"broadband = {params['broadband']} ;{scalar} setting to 1 overrides SR\n")
        f.write(f"td_limit = {params['td_limit']} ;(s) {scalar} detection time upper limit\n")
        
        f.write(";--- TARGET LIST CUTS ---\n")
        f.write(f"target_vmag_cut = {params['target_vmag_cut']} ;(mags) {scalar} max magnitude at V band\n")
        f.write(f"target_distance_cut = {params['target_distance_cut']} ;(pc) {scalar} max distance\n")
        f.write("\n")
        
        f.write(";--- MISC ---\n")
        f.write(f"nexozodis = {params['nexozodis']} ;(zodis) {scalar} exozodi level of all stars\n")
        f.write(f"Tcontam = {params['Tcontam']} ;{scalar} effective throughput factor for contamination\n")
        f.write(f"temperature = {params['temperature']} ;(K) {scalar} temperature of warm optics\n")
        f.write(f"photap_rad = {params['photap_rad']} ;(l/D) {scalar} photometric aperture radius\n")
        f.write("\n")
        
        # f.write(";--- PSF TRUNCATION RATIOS ---\n")
        # psf_trunc_ratio = ', '.join(map(str, params['PSF_trunc_ratio']))
        # f.write(f"PSF_trunc_ratio = [{psf_trunc_ratio}] ;{array} PSF truncation ratios\n")
        # f.write("\n")
        
        # f.write(";--- DETECTION WAVELENGTHS ---\n")
        # lambda_vals = ', '.join(map(str, params['lambda']))
        # f.write(f"lambda = [{lambda_vals}] ;(microns) {array} possible detection wavelengths\n")
        # f.write("\n")
        
        # f.write(";--- SPECTRAL RESOLUTION ---\n")
        # sr_vals = ', '.join(map(str, params['SR']))
        # f.write(f"SR = [{sr_vals}] ;{scalar OR array} desired spectral resolution for detections\n")
        # f.write("\n")
        
        # f.write(";--- SNR FOR DETECTIONS ---\n")
        # snr_vals = ', '.join(map(str, params['SNR']))
        # f.write(f"SNR = [{snr_vals}] #{scalar OR array} minimum SNR for detections\n")
        # f.write("\n")
        
        # f.write(";--- DETECTOR PARAMETERS ---\n")
        # det_qe_vals = ', '.join(map(str, params['det_QE']))
        # f.write(f"det_QE = [{det_qe_vals}] #{scalar OR array} detector QE for detection observations\n")
        
        # f.write(";--- CHARACTERIZATION OBSERVATIONS ---\n")
        # f.write(f"tsc_limit = {params['tsc_limit']} #(s) {scalar} spec char time upper limit\n")
        # f.write(f"sc_broadband = {params['sc_broadband']} #{scalar} setting to 1 overrides sc_SR\n")
        # f.write("\n")


def create_edict(tele, inst, det, name="output.pk"):
    # params that I need to include: 

    params = {}
    for key in tele.__dict__:
        params[key] = tele.__dict__[key]
    for key in inst.__dict__:
        params[key] = inst.__dict__[key]
    for key in det.__dict__:
        params[key] = det.__dict__[key]
    
    return params 


reflectivity_path = "/obs_config/reflectivities/"
detectors_path =  "/obs_config/Detectors/"

class TELESCOPE:
    def __init__(self, lam):
        self.lam = lam

    def plot(self):
        fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
        axes[0].plot(self.lam, self.total_tele_refl, label="Total telescope refl")
        axes[1].plot(self.lam, self.M1_refl, label="M1_refl")
        axes[1].plot(self.lam, self.M2_refl, label="M2_refl")
        axes[1].set_xlabel("Wavelength [um]")
        axes[0].set_ylabel("Trans/refl")
        axes[1].legend()
        plt.show()

    def load_EAC1(self):

        eac1_fl = "/obs_config/Tel/EAC1.yaml"
        print(f"Loading file: {eac1_fl}")
        eac1_dict = load_yaml(eac1_fl) 

        diam_insc = eac1_dict["PM_aperture"]["segmentation_parameters"]["inscribing_diameter"][0] # meters* u.Unit(eac1_dict["PM_aperture"]["segmentation_parameters"]["inscribing_diameter"][1])
        diam_circ = eac1_dict["PM_aperture"]["segmentation_parameters"]["circumscribing_diameter"][0] # meters * u.Unit(eac1_dict["PM_aperture"]["segmentation_parameters"]["circumscribing_diameter"][1])

        print("Calculating telescope throughput...")
        # M1 reflectivity
        M1_reflectivity_fl = eac1_dict["PM_aperture"]["M1_reflectivity"]
        M1_reflectivity_dict = load_yaml(reflectivity_path + M1_reflectivity_fl.split("/")[-1])
        M1_refl = M1_reflectivity_dict["reflectivity"]
        M1_lam = M1_reflectivity_dict["wavelength"] * u.nm
        # interpolate onto our wavelength grid
        M1_refl = interp_arr(M1_lam, M1_refl, self.lam)

        # M2 reflectivity
        M2_reflectivity_fl = eac1_dict["SM"]["reflectivity"]
        M2_reflectivity_dict = load_yaml(reflectivity_path + M1_reflectivity_fl.split("/")[-1])
        M2_refl = M2_reflectivity_dict["reflectivity"]
        M2_lam = M2_reflectivity_dict["wavelength"] * u.nm
        # interpolate onto our wavelength grid
        M2_refl = interp_arr(M2_lam, M2_refl, self.lam)


        # M3 reflectivity
        print("Warning: M3 reflectivity not included in YAML")
        M3_refl = np.ones_like(self.lam.value)

        # M4 reflectivity
        print("Warning: M4 reflectivity not included in YAML")
        M4_refl = np.ones_like(self.lam.value)

        total_tele_refl = M1_refl * M2_refl * M3_refl * M4_refl

        print("Done.")

        # save parameters as class properties
        self.diam_insc = diam_insc
        self.diam_circ = diam_circ
        self.M1_refl = M1_refl
        self.M2_refl = M2_refl
        self.M3_refl = M3_refl
        self.M4_refl = M4_refl
        self.total_tele_refl = total_tele_refl

    def load_EAC2(self):
        eac2_fl = "/obs_config/Tel/EAC2.yaml"
        print(f"Loading file: {eac2_fl}")
        eac2_dict = load_yaml(eac2_fl) 
        print(eac2_dict)
        print("EAC2 LOAD SCRIPT NOT YET IMPLEMENTED")

    def load_EAC3(self):
        eac3_fl = "/obs_config/Tel/EAC3.yaml"
        print(f"Loading file: {eac3_fl}")
        eac3_dict = load_yaml(eac3_fl) 
        print(eac3_dict)
        print("EAC3 LOAD SCRIPT NOT YET IMPLEMENTED")

    def load_custom(self, diam_insc, diam_circ, total_tele_refl):
        self.diam_insc = diam_insc
        self.diam_circ = diam_circ
        self.total_tele_refl = total_tele_refl



class CI:
    def __init__(self, lam):
        print("Initializing Coronagraph Instrument")
        self.lam = lam # the native wavelength grid we are working in (in um)
    def plot(self):
 
        fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
        try:
            axes[0].plot(self.lam, self.total_inst_refl, label="Total instrument refl")
        except AttributeError:
            print("####################################################")
            print("Nothing to plot. Try running CI.calculate_throughput")
            print("####################################################")


        axes[1].plot(self.lam, self.TCA, label="TCA")
        axes[1].plot(self.lam, self.wb_tran, label="wave_beamsplitter tran")
        axes[1].plot(self.lam, self.wb_refl, label="wave_beamsplitter refl")
        axes[1].plot(self.lam, self.pol_beamsplitter, label="pol_beamsplitter")
        axes[1].plot(self.lam, self.FSM, label="FSM")
        axes[1].plot(self.lam, self.OAPs_forward, label="OAPs_forward")
        axes[1].plot(self.lam, self.DM1, label="DM1")
        axes[1].plot(self.lam, self.DM2, label="DM2")
        axes[1].plot(self.lam, self.Fold, label="Fold")
        axes[1].plot(self.lam, self.OAPs_back, label="OAPs_back")
        axes[1].plot(self.lam, self.Apodizer, label="Apodizer")
        axes[1].plot(self.lam, self.Focal_Plane_Mask, label="Focal_Plane_Mask")
        axes[1].plot(self.lam, self.Lyot_Stop, label="Lyot_Stop")
        axes[1].plot(self.lam, self.Field_Stop, label="Field_Stop")
        axes[1].plot(self.lam, self.filters, label="filters")
        axes[1].set_xlabel("Wavelength [um]")
        axes[0].set_ylabel("Trans/refl")
        axes[1].legend()
        plt.show()
        



    def calculate_throughput(self):
        ci_fl = "/obs_config/CI/CI.yaml"
        print(f"Loading file: {ci_fl}\n")
        

        # Full optical Path: 'PM','SM','TCA','TCA','TCA','TCA','wave_beamsplitter', 'pol_beamsplitter', 'FSM', 'OAPs_forward', 'OAPs_forward', 'DM1', 'DM2', 'OAPs_forward', 'Fold', 'OAPs_back', 'Apodizer', 'OAPs_back', 'Focal_Plane_Mask', 'OAPs_back', 'Lyot_Stop', 'OAPs_back', 'Field_Stop', 'OAPs_back', 'filters', 'OAPs_back', 'Detector'
        OP_full_txt = ['PM','SM','TCA','TCA','TCA','TCA','wave_beamsplitter', 'pol_beamsplitter', 'FSM', 'OAPs_forward', 'OAPs_forward', 'DM1', 'DM2', 'OAPs_forward', 'Fold', 'OAPs_back', 'Apodizer', 'OAPs_back', 'Focal_Plane_Mask', 'OAPs_back', 'Lyot_Stop', 'OAPs_back', 'Field_Stop', 'OAPs_back', 'filters', 'OAPs_back', 'Detector']
        OP_tele_txt = ['PM','SM']
        OP_inst_txt = ['TCA','TCA','TCA','TCA','wave_beamsplitter', 'pol_beamsplitter', 'FSM', 'OAPs_forward', 'OAPs_forward', 'DM1', 'DM2', 'OAPs_forward', 'Fold', 'OAPs_back', 'Apodizer', 'OAPs_back', 'Focal_Plane_Mask', 'OAPs_back', 'Lyot_Stop', 'OAPs_back', 'Field_Stop', 'OAPs_back', 'filters', 'OAPs_back']
        OP_det_txt = ['Detector']

        # save the optical paths
        self.OP_full = OP_full_txt
        self.OP_tele = OP_tele_txt
        self.OP_inst = OP_inst_txt
        self.OP_det = OP_det_txt

        ci_dict = load_yaml(ci_fl)
        print("Optical path:")
        print(ci_dict["opticalpath"]["full_path"])

        print("Calculating throughput...")

        # TCA
        TCA_reflectivity_fl = ci_dict["TCA"]["reflectivity"]
        TCA_reflectivity_dict = load_yaml(reflectivity_path + TCA_reflectivity_fl.split("/")[-1])
        TCA_refl = TCA_reflectivity_dict["reflectivity"]
        TCA_lam = TCA_reflectivity_dict["wavelength"] * u.nm
        # interpolate onto our wavelength grid
        TCA_refl = interp_arr(TCA_lam, TCA_refl, self.lam)

        # wave_beamsplitter
        wb_reflectivity_fl = ci_dict["wave_beamsplitter"]["reflectivity"] # < 1 um
        wb_transmission_fl = ci_dict["wave_beamsplitter"]["transmission"] # > 1 um
        wb_reflectivity_dict = load_yaml(reflectivity_path + wb_reflectivity_fl.split("/")[-1])
        wb_transmission_dict = load_yaml(reflectivity_path + wb_transmission_fl.split("/")[-1])
        wb_refl = wb_reflectivity_dict["reflectivity"]
        wb_refl_lam = wb_reflectivity_dict["wavelength"] * u.nm
        wb_tran = wb_transmission_dict["reflectivity"]
        wb_tran_lam = wb_transmission_dict["wavelength"] * u.nm
        # interpolate onto our wavelength grid    
        wb_refl = interp_arr(wb_refl_lam, wb_refl, self.lam)
        wb_tran = interp_arr(wb_tran_lam, wb_tran, self.lam)
        wb_tot = wb_refl + wb_tran 
        # plt.figure()
        # plt.plot(self.lam, wb_tot)
        # plt.show()
        # assert False

        # pol_beamsplitter
        # no transmission/reflectivity profiles here
        pb_refl = np.ones_like(self.lam.value)

        # FSM
        FSM_reflectivity_fl = ci_dict["FSM"]["reflectivity"]
        FSM_reflectivity_dict = load_yaml(reflectivity_path + FSM_reflectivity_fl.split("/")[-1])
        FSM_refl = FSM_reflectivity_dict["reflectivity"]
        FSM_lam = FSM_reflectivity_dict["wavelength"] * u.nm
        # interpolate onto our wavelength grid
        FSM_refl = interp_arr(FSM_lam ,FSM_refl, self.lam)

        # OAPs_forward
        OAPsf_reflectivity_fl = ci_dict["OAPs_forward"]["reflectivity"]
        OAPsf_reflectivity_dict = load_yaml(reflectivity_path + OAPsf_reflectivity_fl.split("/")[-1])
        OAPsf_refl = OAPsf_reflectivity_dict["reflectivity"]
        OAPsf_lam = OAPsf_reflectivity_dict["wavelength"] * u.nm
        # interpolate onto our wavelength grid
        OAPsf_refl = interp_arr(OAPsf_lam ,OAPsf_refl, self.lam)

        # DM1 
        DM1_reflectivity_fl = ci_dict["DM1"]["reflectivity"]
        DM1_reflectivity_dict = load_yaml(reflectivity_path + DM1_reflectivity_fl.split("/")[-1])
        DM1_refl = DM1_reflectivity_dict["reflectivity"]
        DM1_lam = DM1_reflectivity_dict["wavelength"] * u.nm
        # interpolate onto our wavelength grid
        DM1_refl = interp_arr(DM1_lam ,DM1_refl, self.lam)

        # DM2 
        DM2_reflectivity_fl = ci_dict["DM2"]["reflectivity"]
        DM2_reflectivity_dict = load_yaml(reflectivity_path + DM2_reflectivity_fl.split("/")[-1])
        DM2_refl = DM2_reflectivity_dict["reflectivity"]
        DM2_lam = DM2_reflectivity_dict["wavelength"] * u.nm
        # interpolate onto our wavelength grid
        DM2_refl = interp_arr(DM2_lam ,DM2_refl, self.lam)

        # Fold
        Fold_reflectivity_fl = ci_dict["Fold"]["reflectivity"]
        Fold_reflectivity_dict = load_yaml(reflectivity_path + Fold_reflectivity_fl.split("/")[-1])
        Fold_refl = Fold_reflectivity_dict["reflectivity"]
        Fold_lam = Fold_reflectivity_dict["wavelength"] * u.nm
        # interpolate onto our wavelength grid
        Fold_refl = interp_arr(Fold_lam ,Fold_refl, self.lam)

        # OAPs_back
        OAPsb_reflectivity_fl = ci_dict["OAPs_back"]["reflectivity"]
        OAPsb_reflectivity_dict = load_yaml(reflectivity_path + OAPsb_reflectivity_fl.split("/")[-1])
        OAPsb_refl = OAPsb_reflectivity_dict["reflectivity"]
        OAPsb_lam = OAPsb_reflectivity_dict["wavelength"] * u.nm
        # interpolate onto our wavelength grid
        OAPsb_refl = interp_arr(OAPsb_lam ,OAPsb_refl, self.lam)

        # Apodizer
        Apodizer_reflectivity_fl = ci_dict["Apodizer"]["reflectivity"]
        Apodizer_reflectivity_dict = load_yaml(reflectivity_path + Apodizer_reflectivity_fl.split("/")[-1])
        Apodizer_refl = Apodizer_reflectivity_dict["reflectivity"]
        Apodizer_lam = Apodizer_reflectivity_dict["wavelength"] * u.nm
        # interpolate onto our wavelength grid
        Apodizer_refl = interp_arr(Apodizer_lam ,Apodizer_refl, self.lam)

        # Focal_Plane_Mask
        FPM_reflectivity_fl = ci_dict["Focal_Plane_Mask"]["transmission"]
        FPM_reflectivity_dict = load_yaml(reflectivity_path + FPM_reflectivity_fl.split("/")[-1])
        FPM_refl = FPM_reflectivity_dict["reflectivity"]
        FPM_lam = FPM_reflectivity_dict["wavelength"] * u.nm
        # interpolate onto our wavelength grid
        FPM_refl = interp_arr(FPM_lam ,FPM_refl, self.lam)

        # Lyot_Stop
        Lyot_reflectivity_fl = ci_dict["Lyot_Stop"]["reflectivity"]
        Lyot_reflectivity_dict = load_yaml(reflectivity_path + Lyot_reflectivity_fl.split("/")[-1])
        Lyot_refl = Lyot_reflectivity_dict["reflectivity"]
        Lyot_lam = Lyot_reflectivity_dict["wavelength"] * u.nm
        # interpolate onto our wavelength grid
        Lyot_refl = interp_arr(Lyot_lam ,Lyot_refl, self.lam)

        # Field_Stop
        FStop_reflectivity_fl = ci_dict["Field_Stop"]["transmission"]
        FStop_reflectivity_dict = load_yaml(reflectivity_path + FStop_reflectivity_fl.split("/")[-1])
        FStop_refl = FStop_reflectivity_dict["reflectivity"]
        FStop_lam = FStop_reflectivity_dict["wavelength"] * u.nm
        # interpolate onto our wavelength grid
        FStop_refl = interp_arr(FStop_lam ,FStop_refl, self.lam)

        # filters 
        # Filters not implemented yet
        filters = np.ones_like(self.lam.value)

        # save parameters as class properties
        self.TCA = TCA_refl
        self.wb_tran = wb_tran
        self.wb_refl = wb_refl
        self.wave_beamsplitter = self.wb_tran + self.wb_refl
        self.pol_beamsplitter = pb_refl
        self.FSM = FSM_refl
        self.OAPs_forward = OAPsf_refl
        self.DM1 = DM1_refl
        self.DM2 = DM2_refl
        self.Fold = Fold_refl
        self.OAPs_back = OAPsb_refl
        self.Apodizer = Apodizer_refl
        self.Focal_Plane_Mask = FPM_refl
        self.Lyot_Stop = Lyot_refl
        self.Field_Stop = FStop_refl
        self.filters = filters

        total_inst_refl = np.ones_like(self.lam.value)
        print("Calculating instrument throughput...")
        for element in OP_inst_txt:
            print(f"--including {element}")
            refl_temp = self.__dict__[element]
            total_inst_refl *= refl_temp
        self.total_inst_refl = total_inst_refl
            
        # for total reflectivity/transmission, follow the optical path:
        # Full optical Path: 'PM','SM','TCA','TCA','TCA','TCA','wave_beamsplitter', 'pol_beamsplitter', 'FSM', 'OAPs_forward', 'OAPs_forward', 'DM1', 'DM2', 'OAPs_forward', 'Fold', 'OAPs_back', 'Apodizer', 'OAPs_back', 'Focal_Plane_Mask', 'OAPs_back', 'Lyot_Stop', 'OAPs_back', 'Field_Stop', 'OAPs_back', 'filters', 'OAPs_back', 'Detector'
        #total_inst_refl = TCA_refl * TCA_refl * TCA_refl * TCA_refl * (wb_tran + wb_refl) * pb_refl * FSM_refl * OAPsf_refl * OAPsf_refl * DM1_refl * DM2_refl * OAPsf_refl * Fold_refl * OAPsb_refl * Apodizer_refl * OAPsb_refl * FPM_refl * OAPsb_refl * Lyot_refl * OAPsb_refl * FStop_refl * OAPsb_refl * filters * OAPsb_refl

        print("Done.")

        #ci = CI(lam, total_inst_refl, TCA_refl, wb_tran, wb_refl, pb_refl, FSM_refl, OAPsf_refl, DM1_refl, DM2_refl, Fold_refl, OAPsb_refl, Apodizer_refl, FPM_refl, Lyot_refl, FStop_refl, filters)



class DETECTOR:
    def __init__(self, lam):
        self.lam = lam

    def load_imager(self):

        print("Loading Broadband Imager...")

        ci_dict = load_yaml("/obs_config/CI/CI.yaml")

        # visible channels
        vis_imager = ci_dict["Visible_Channels"]["Detectors"]["Broadband_Imager"]
        qe_vis_fl = vis_imager["QE"]
        qe_vis_dict = load_yaml(detectors_path + qe_vis_fl.split("/")[-1])
        qe_vis = qe_vis_dict["QE"]
        qe_vis_lam = qe_vis_dict["wavelength"] * u.nm
        # interpolate onto our wavelength grid
        qe_vis = interp_arr(qe_vis_lam ,qe_vis, self.lam)

        rn_vis = vis_imager["RN"][0]# electrons/pix      * u.Unit(vis_imager["RN"][1])
        dc_vis = vis_imager["DC"][0]# electrons/pix/s    * u.Unit(vis_imager["DC"][1])
        cic_vis = None # NOT IMPLEMENTED YET 

        # save parameters as class properties
        self.qe_vis = qe_vis
        self.rn_vis = rn_vis
        self.dc_vis = dc_vis
        self.cic_vis = cic_vis

        # nir channels
        nir_imager = ci_dict["NIR_Channels"]["Detectors"]["Broadband_Imager"]
        qe_nir_fl = nir_imager["QE"]
        qe_nir_dict = load_yaml(detectors_path + qe_nir_fl.split("/")[-1])
        qe_nir = qe_nir_dict["QE"]
        qe_nir_lam = qe_nir_dict["wavelength"] * u.nm
        # interpolate onto our wavelength grid
        qe_nir = interp_arr(qe_nir_lam ,qe_nir, self.lam)

        rn_nir = nir_imager["RN"][0] # electrons/pix        * u.Unit(nir_imager["RN"][1]) 
        dc_nir = nir_imager["DC"][0] # electrons/pix/s       * u.Unit(nir_imager["DC"][1]) 
        cic_nir = None # NOT IMPLEMENTED YET

        # save parameters as class properties
        self.qe_nir = qe_nir
        self.rn_nir = rn_nir
        self.dc_nir = dc_nir
        self.cic_nir = cic_nir


    def load_IFS(self):
        print("Loading IFS...")

        ci_dict = load_yaml("/obs_config/CI/CI.yaml")

        # visible channels
        vis_ifs = ci_dict["Visible_Channels"]["Detectors"]["IFS"]
        qe_vis_fl = vis_ifs["QE"]
        qe_vis_dict = load_yaml(detectors_path + qe_vis_fl.split("/")[-1])
        qe_vis = qe_vis_dict["QE"]
        print(qe_vis)
        qe_vis_lam = qe_vis_dict["wavelength"] * u.nm
        # interpolate onto our wavelength grid
        qe_vis = interp_arr(qe_vis_lam, qe_vis, self.lam)
        print(qe_vis)


        rn_vis = vis_ifs["RN"][0] # electrons/pix
        dc_vis = vis_ifs["DC"][0] # electrons/pix/s
        cic_vis = None # NOT IMPLEMENTED YET

        # save parameters as class properties
        self.qe_vis = qe_vis
        self.rn_vis = rn_vis
        self.dc_vis = dc_vis
        self.cic_vis = cic_vis


        # nir channels
        nir_ifs = ci_dict["NIR_Channels"]["Detectors"]["IFS"]
        qe_nir_fl = nir_ifs["QE"]
        qe_nir_dict = load_yaml(detectors_path + qe_nir_fl.split("/")[-1])
        qe_nir = qe_nir_dict["QE"]
        qe_nir_lam = qe_nir_dict["wavelength"] * u.nm
        # interpolate onto our wavelength grid
        qe_nir = interp_arr(qe_nir_lam, qe_nir, self.lam)

        rn_nir = nir_ifs["RN"][0] # electrons/pix
        dc_nir = nir_ifs["DC"][0] # electrons/pix/s
        cic_nir = None # NOT IMPLEMENTED YET

        # save parameters as class properties
        self.qe_nir = qe_nir
        self.rn_nir = rn_nir
        self.dc_nir = dc_nir
        self.cic_nir = cic_nir

    def plot(self):

        plt.figure()
        plt.plot(self.lam, self.qe_vis, label="QE VIS")
        plt.plot(self.lam, self.qe_nir, label="QE NIR")
        plt.xlabel("Wavelength [um]")
        plt.ylabel("Quantum Efficiency")
        plt.legend()
        plt.show()
        print("VISIBLE")
        print("-------")
        print(f"RN: {self.rn_vis}")
        print(f"DC: {self.dc_vis}")
        print("\n")
        print("NIR")
        print("---")
        print(f"RN: {self.rn_nir}")
        print(f"DC: {self.dc_nir}")


def run(telescope_name, instrument_name, detector_name, output_format):

    if detector_name == "IFS":
        R = 1000 # resolution of wavelength grid
        lammin = 0.5 # minimum wavelength (um)
        lammax = 2.0 # maximum wavelength (um)
        internal_lam = generate_wavelength_grid(lammin, lammax, R) * u.um

    elif detector_name == "IMAGER":
        lam_center = 0.5
        internal_lam = [0.5] * u.um
    plotting=False

    # load the telescope
    telescope = TELESCOPE(internal_lam)
    if telescope_name == "EAC1":
        telescope.load_EAC1()
    elif telescope_name == "EAC2":
        telescope.load_EAC2()
    elif telescope_name == "EAC3":
        telescope.load_EAC3()
    else:
        assert False, f"Telescope name {telescope_name} not recognized!"
    if plotting:
        telescope.plot()
    
    # load the instrument (assuming CI for now)
    instrument = CI(internal_lam)
    instrument.calculate_throughput()
    if plotting:
        instrument.plot()
    
    # load the detector
    detector = DETECTOR(internal_lam)
    if detector_name == "IMAGER":
        detector.load_imager()
    elif detector_name == "IFS":
        detector.load_IFS()
    if plotting:
        detector.plot()

    # put everything in a python dict
    params = create_edict(telescope, instrument, detector)
    return params

if __name__ == "__main__":
    
    # User: specify names of telesope, instrument, and detector 
    telescope_name = "EAC1"
    instrument_name = "CI"
    detector_name = "IMAGER"

    output_format = "idk"

    #params_all = run(telescope_name, instrument_name, detector_name, output_format)
    params_all = run("EAC1", "CI", "IFS", "edith_dictionary")


    print(params_all)
    print("Here")

    

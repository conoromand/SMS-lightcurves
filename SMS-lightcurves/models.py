import SMS_lightcurve_classes as SMSclasses
import SMS_lightcurve_interface as SMSinterface
import numpy as np
from scipy import optimize
from scipy import integrate
from scipy.interpolate import interp1d
from redback.utils import (calc_kcorrected_properties, citation_wrapper, logger, get_csm_properties, nu_to_lambda, get_optimal_time_array,
                           lambda_to_nu, velocity_from_lorentz_factor, build_spectral_feature_list)
from redback.constants import day_to_s, solar_mass, km_cgs, au_cgs, speed_of_light, sigma_sb
from collections import namedtuple
import redback.sed as sed
from redback.sed import flux_density_to_spectrum, blackbody_to_spectrum

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2026MNRAS.545f1949J/abstract')
def smssn_dynamics(time, logeej, logmej, logr0, logA, n_csm, **kwargs):
    """
    Explosion of a supermassive star interacting with optically thick CSM.
    :param time: time in days in source frame
    :param logeej: logarithm of the kinetic energy of the ejecta in erg s^-1
    :param logmej: logarithm of the ejecta mass in solar masses
    :param logr0: logarithm of the initial shock radius in cm
    :param logA: logarithm of the normalization of csm density profile at 1e15 cm (rho ~ A (r/1e15 cm)^(-n_csm)) in g cm^-3
    :param n_csm: power-law index of csm density profile (rho ~ A (r/1e15 cm)^(-n_csm))
    :param n_ej: power-law index of ejecta density profile [NOTE: only values of 0, 1, and 2 are supported]
    :return: nametuple with the following properties
                L_bol: bolometric luminosity in erg s^-1
                R_phot: Photospheric radius in cm
                v_sh: Shock velocity in cm s^-1
                eta_therm: thermal coupling coefficient, as defined by Nakar & Sari 2010 (EARLY SUPERNOVAE LIGHT CURVES FOLLOWING THE SHOCK BREAKOUT)
                T_bb: surface temperature as inferred using the Stefan-Boltzmann law in K
                T_col: observable photosphere (colour) temperature in K
                tau: optical depth of electron scattering in thin shock region
                nu_abs_ff: free-free critical absorption frequency in Hz (Eq 11 from Irwin & Hotokezaka 2025)
    """
    
    n_ej = kwargs.get('n_ej', 0)
    e_ej = 10**logeej
    m_ej = 10**(logmej)
    r_0 = 10**logr0
    A15 = 10**logA
    A = A15 * (1e15)**n_csm
    output = SMSinterface.interface_single_sms_lightcurve(e_ej, m_ej*solar_mass, r_0, A, n_csm, n_ej)
    dynamics_output = namedtuple('dynamics_output', ['time', 'L_bol', 'R_phot',
                                                     'v_sh', 'eta_therm', 'T_bb', 'T_col', 
                                                     'tau', 'nu_abs_ff'])
    bol_func = interp1d(output[0]/day_to_s, y=output[1]) 
    rad_func = interp1d(output[0]/day_to_s, y=output[2]) 
    vsh_func = interp1d(output[0]/day_to_s, y=output[3]) 
    eta_func = interp1d(output[0]/day_to_s, y=output[4]) 
    tbb_func = interp1d(output[0]/day_to_s, y=output[5]) 
    tcol_func = interp1d(output[0]/day_to_s, y=output[6]) 
    tau_func = interp1d(output[0]/day_to_s, y=output[7]) 
    nuabs_func = interp1d(output[0]/day_to_s, y=output[8])                                                
    dynamics_output.time = time
    dynamics_output.L_bol = bol_func(time)
    dynamics_output.R_phot = rad_func(time)
    dynamics_output.v_sh = vsh_func(time)
    dynamics_output.eta_therm = eta_func(time)
    dynamics_output.T_bb = tbb_func(time)
    dynamics_output.T_col = tcol_func(time)
    dynamics_output.tau = tau_func(time)
    dynamics_output.nu_abs_ff = nuabs_func(time)
    return dynamics_output

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2026MNRAS.545f1949J/abstract')
def smssn_bolometric(time, logeej, logmej, logr0, logA, n_csm, **kwargs):
    """
    Explosion of a supermassive star interacting with optically thick CSM.
    :param time: time in days in source frame
    :param logeej: logarithm of the kinetic energy of the ejecta in erg s^-1
    :param logmej: logarithm of the ejecta mass in solar masses
    :param logr0: logarithm of the initial shock radius in cm
    :param logA: logarithm of the normalization of csm density profile at 1e15 cm (rho ~ A (r/1e15 cm)^(-n_csm)) in g cm^-3
    :param n_csm: power-law index of csm density profile (rho ~ A (r/1e15 cm)^(-n_csm))
    :param n_ej: power-law index of ejecta density profile [NOTE: only values of 0, 1, and 2 are supported]
    :return: bolometric luminosity
    """
    output = smssn_dynamics(time, logeej, logmej, logr0, logA, n_csm, **kwargs)
    return output.L_bol

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2026MNRAS.545f1949J/abstract')
def smssn_bolometric(time, redshift,logeej, logmej, logr0, logA, n_csm, **kwargs):
    """
    Explosion of a supermassive star interacting with optically thick CSM.
    :param time: time in days in observer frame
    :param redshift: redshift of the source
    :param logeej: logarithm of the kinetic energy of the ejecta in erg s^-1
    :param logmej: logarithm of the ejecta mass in solar masses
    :param logr0: logarithm of the initial shock radius in cm
    :param logA: logarithm of the normalization of csm density profile at 1e15 cm (rho ~ A (r/1e15 cm)^(-n_csm)) in g cm^-3
    :param n_csm: power-law index of csm density profile (rho ~ A (r/1e15 cm)^(-n_csm))
    :param n_ej: power-law index of ejecta density profile [NOTE: only values of 0, 1, and 2 are supported]
    :return: flux density or AB magnitude
    """
    kwargs['sed'] = kwargs.get("sed", sed.Blackbody)
    dl = cosmo.luminosity_distance(redshift).cgs.value

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        output = smssn_dynamics(time, logeej, logmej, logr0, logA, n_csm, **kwargs)
        sed_1 = kwargs['sed'](time=time, luminosity=output.L_bol, temperature=output.T_col,
                                              r_photosphere=output.R_phot, frequency=frequency, luminosity_distance=dl)
        flux_density = sed_1.flux_density
        return flux_density.to(uu.mJy).value * (1 + redshift)
    else:
        time_obs = time
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
        time_temp = get_optimal_time_array(0.1, 3000, 300) # in days
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        output = smssn_dynamics(time, logeej, logmej, logr0, logA, n_csm, **kwargs)
        sed_1 = kwargs['sed'](temperature=output.T_col, r_photosphere=output.R_phot,
                              frequency=frequency[:,None], luminosity_distance=dl)
        fmjy = sed_1.flux_density.T
        spectra = flux_density_to_spectrum(fmjy, redshift, lambda_observer_frame)
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                          lambdas=lambda_observer_frame,
                                                                          spectra=spectra)
        else:
            return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                              spectra=spectra, lambda_array=lambda_observer_frame,
                                                              **kwargs)  

#SMS_modifiedblackbody




import SMS_lightcurve_classes as SMSclasses
from matplotlib import pyplot as plt
# ==============================================================================================================================

def interface_single_sms_lightcurve(Ekin_ejecta, Mass_ejecta, Radius_initial, csm_powerlaw_prefactor, csm_powerlaw_index, ejecta_index=0):
	'''
	This function integrates the bolometric light curve for an optically-thick shell, powered by CSM shock interaction.
	[all quantities are in cgs units]
	INPUTS: 
	Ekin_ejecta [kinetic energy of ejecta], 
	Mass_ejecta [otal mas sof ejecta], 
	Radius_initial [initial radius of shock], 
	csm_powerlaw_prefactor [normalization factor for CSM density profile], 
	csm_powerlaw_index [index of powerlaw CSM profile], 
	ejecta_index [powerlaw index for the ejecta density distribution \rho ~ (\gamma\beta)^-n, ONLY integer n = 0, 1, 2 are possible, other values are NOT SUPPORTED]
	OUTPUTS:
	time [array contianing the timestamps of the lightcurve in the restframe],
	Luminosity_bolometric [bolometric luminosity of light curve],
	radius_photosphere [radius of the photpsphere],
	velocity_shock [expansion velocity of the optically-thick shock shell],
	eta_thermal_coupling_coefficient [thermal coupling coefficient, as defined by Nakar&Sari2010 (EARLY SUPERNOVAE LIGHT CURVES FOLLOWING THE SHOCK BREAKOUT)],
	Temperature_blackbody [surface temperature as inferred using the Stefan-Boltzmann law],
	Temperature_colour [observable photosphere temperature]
	'''
	# global constants:
	opacity = 0.35 # [cm^2/g] # opacity of primordial gas
	Tion = 6000. # [Kelvin] ionisation/recombination tmeperature of Hydrogen
	day = 60*60*24 # seconds in a day
	year = 365.25*day
	
    # class constructor:
	model = SMSclasses.SMSlightcurve_sphericalCSMshock_modified(Ekin_ejecta, Mass_ejecta, Radius_initial, Tion, opacity, n_density_in=ejecta_index)
	
	# use a powerlaw CSM of the form rho(r) = A * r^(-n) , where A = CSM_powerlaw_simple_prefactor_cgs ; and n = CSM_power_law_exponent
	model.use_CSM_powerlaw_simple = True
	model.CSM_powerlaw_simple_prefactor_cgs = csm_powerlaw_prefactor
	model.CSM_power_law_exponent = csm_powerlaw_index
	# set some integrator options, then integrate:
	model.max_integration_time = 4.*year
	model.max_timestep = 0.5*day
	model.stop_after_opt_thick_phase = True
	model.integrate_model()
	
	# allocate the output quantities:
	time = model.time_arr - model.time_arr[0] # shift the time coordinate so that the array starts at t_0 = 0 seconds
	Luminosity_bolometric = model.Lbol_arr
	radius_photosphere = model.Rphotosphere_arr # photosphere is located at forward shock location (assuming geometrically thin shock shell)
	velocity_shock = model.v_shock_shell_arr
	eta_thermal_coupling_coefficient = model.eta_factor_arr # if eta > 1, then the radiation field is out of equilibrium and the spectrum is not a blackbody
	Temperature_blackbody = model.Temp_BB_surface # temperature that is inferred using the Stefan-Boltzmann law: Lbol=4pi \sigma R^2 T^4
	Temperature_colour = model.Temp_eff_arr # photosphere temperature that we would observe. Is equal to Temperature_blackbody if thermal_coupling_coefficient is <1, but is larger if thermal_coupling_coefficient>1
	optical_depth_shock_electron_scattering = model.optical_depth_shock_arr # optical depth due to electron scattering

	return time, Luminosity_bolometric, radius_photosphere, velocity_shock, eta_thermal_coupling_coefficient, Temperature_blackbody, Temperature_colour, optical_depth_shock_electron_scattering

# ==============================================================================================================================
if __name__ == "__main__":
	
	# minimal working example for the interface:
	M_sun_gram = 1.988e33
	R_sun_cm = 6.957e10
	# inputs:
	Ekin_ejecta = 2e53 # in erg
	Mass_ejecta = 100.*M_sun_gram # in gram
	Radius_initial = 2.*R_sun_cm # in cm
	csm_powerlaw_prefactor = 4e16 # Parameter A .CSM power law is rho ~ A r^(-n). in cgs units
	csm_powerlaw_index = 2.0 # parameter n. CSM power law is rho ~ A r^(-n). floating point number between 0 < n < 3 should be reasonable
	ejecta_index = 0 # ejecta power law is rho ~ (\Gamma\beta)^(-n_eje). Integer n = 0, 1, 2. Other values are NOT SUPPORTED

	time, Luminosity_bolometric, radius_photosphere, velocity_shock, eta_thermal_coupling_coefficient, Temperature_blackbody, Temperature_colour, optical_depth_shock_electron_scattering = interface_single_sms_lightcurve(Ekin_ejecta, Mass_ejecta, Radius_initial, csm_powerlaw_prefactor, csm_powerlaw_index, ejecta_index)

	print("Done!")

	# plot the quantities:
	day = 60*60*24
	plt.plot(time/day, Luminosity_bolometric, label="Luminosity_bolometric")
	plt.plot(time/day, radius_photosphere, label="radius_photosphere")
	plt.plot(time/day, velocity_shock, label="velocity_shock")
	plt.plot(time/day, eta_thermal_coupling_coefficient, label="eta_thermal_coupling_coefficient")
	plt.plot(time/day, Temperature_blackbody, label="Temperature_blackbody")
	plt.plot(time/day, Temperature_colour, label="Temperature_colour")
	plt.plot(time/day, optical_depth_shock_electron_scattering, label="optical_depth_shock_electron_scattering")
	
	plt.semilogy()
	plt.ylim(1e-2, 1e46)
	plt.legend()
	plt.show()
	plt.close()

	exit()
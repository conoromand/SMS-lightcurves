import SMS_lightcurve_classes as SMSclasses
from matplotlib import pyplot as plt
# ==============================================================================================================================

def interface_single_sms_lightcurve(Ekin_ejecta, Mass_ejecta, Radius_initial, csm_powerlaw_prefactor, csm_powerlaw_index, ejecta_index=0):
	'''
	This function integrates the bolometric light curve for an optically-thick shell, powered by CSM shock interaction.
	[all quantities are in cgs units]
	INPUTS: 
	Ekin_ejecta [kinetic energy of ejecta], 
	Mass_ejecta [total mass of ejecta], 
	Radius_initial [initial radius of shock], 
	csm_powerlaw_prefactor [normalization factor for CSM density profile], 
	csm_powerlaw_index [index of powerlaw CSM profile], 
	#ejecta_index [powerlaw index for the ejecta density distribution $rho ~ (Gamma * beta)^-n$, ONLY integer n = 0, 1, 2 are possible, other values are NOT SUPPORTED]
	OUTPUTS:
	time [array contianing the timestamps of the lightcurve in the restframe],
	Luminosity_bolometric [bolometric luminosity of light curve],
	radius_photosphere [radius of the photpsphere],
	velocity_shock [expansion velocity of the optically-thick shock shell],
	eta_thermal_coupling_coefficient [thermal coupling coefficient, as defined by Nakar&Sari2010 (EARLY SUPERNOVAE LIGHT CURVES FOLLOWING THE SHOCK BREAKOUT)],
	Temperature_blackbody [surface temperature as inferred using the Stefan-Boltzmann law],
	Temperature_colour [observable photosphere temperature]
	optical_depth_shock_electron_scattering [optical depth to electron scattering of the thin shock region]
	nu_absorption_ff [free-free critical absorption requency according to Eq. (11) in Irwin 2025 et al. MNRAS 543, 2917–2942 (2025) https://academic.oup.com/mnras/article/543/3/2917/8262825?login=false]
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
	model.max_timestep = 1.*day
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
	nu_absorption_ff = model.nu_absorption_ff # free-free critical absorption requency according to Eq. (11) in Irwin 2025 et al. MNRAS 543, 2917–2942 (2025) https://academic.oup.com/mnras/article/543/3/2917/8262825?login=false

	return time, Luminosity_bolometric, radius_photosphere, velocity_shock, eta_thermal_coupling_coefficient, Temperature_blackbody, Temperature_colour, optical_depth_shock_electron_scattering, nu_absorption_ff

def single_star_color_evolution_ZTF(model_name_in, Ekin_ejecta, Mass_ejecta, Radius_initial, csm_powerlaw_prefactor, csm_powerlaw_index, ejecta_index=0, redshift_in=1., xrange=[0,1500],yrange=[25,12], num_filters_in=2):
	
	# choose which model to run:
	model_name = model_name_in
	num_filters = num_filters_in
	#-------------------------------------------
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
	model.max_integration_time = 5.*year
	model.max_timestep = 2.*day
	model.stop_after_opt_thick_phase = True
	model.force_turn_off_non_thermal_effects = False
	model.integrate_model()
	#
	#plt.plot(model.time_arr/year, model.Lbol_arr, linewidth=1.5, label= "Case H2-S: $E_{kin}$ = " + '{:.2e}'.format(E_kin_eje) + "$erg, M_{eje}$ = " + '{:.2e}'.format(M_eje/M_sun_gram) + "$M_\odot, R_{0}$ = " + '{:.2e}'.format(R_0) + "$cm$", ls="-", c="b")
	
	#plt.figure(figsize=(6.4,4.8))
	#fig, ax1 = plt.subplots(figsize=(6.4,4.8))

	# init filters:
	filter_ZTF_g_band = SMSclasses.Telescope_filter("ZTF_g_band")
	filter_ZTF_r_band = SMSclasses.Telescope_filter("ZTF_r_band")
	filter_ATLAS_cyan_band = SMSclasses.Telescope_filter("ATLAS_cyan_band")
	filter_ATLAS_orange_band = SMSclasses.Telescope_filter("ATLAS_orange_band")

	#
	filter_arr = [filter_ZTF_g_band,filter_ZTF_r_band,filter_ATLAS_cyan_band,filter_ATLAS_orange_band]
	filter_output_arr = [None, None, None, None]
	model_name_arr = ["ZTF-g","ZTF-r","ATLAS-cyan","ATLAS-orange"]
	colour_arr = ["green","red","cyan","orange"]
	#colour_arr = ["#ffad7f","#dffd7f","#7fda7f","#7fd2da","#7faded","#7f7fdd","#ab7fce","#bf7fc8"]

	redshift = redshift_in
	#ax1.scatter([-10], [-10], marker="o", c="white", label=r"z = " + str(redshift))
	for i in range(0,num_filters):
		ABmag_lightcurve = SMSclasses.ABmagnitude_lightcurve(model, redshift, filter_arr[i])
		ABmag_lightcurve.debug = False
		ABmag_lightcurve.use_Irwin_spectrum = True
		ABmag_lightcurve.use_GP_through = False
		ABmag_lightcurve.compute_AB_magnitude()
		#
		#ABmag_lightcurve.write_spectrum_into_file("irwin_spectrum.txt"); exit()
		filter_output_arr[i] = ABmag_lightcurve.ABmag_arr

		# split the light curve into two parts
		'''
		time_arr_shock_opt_thick = ABmag_lightcurve.time_arr[0:model.t_shock_transparent_index]
		time_arr_shock_transparent = ABmag_lightcurve.time_arr[model.t_shock_transparent_index:-1]
		mAB_arr_shock_opt_thick = ABmag_lightcurve.ABmag_arr[0:model.t_shock_transparent_index]
		mAB_arr_shock_transparent = ABmag_lightcurve.ABmag_arr[model.t_shock_transparent_index:-1]
		#plt.plot((1+redshift)*ABmag_lightcurve.time_arr/year, ABmag_lightcurve.ABmag_arr, linewidth=2.5, color=colour_arr[i], linestyle="-")
		ax1.plot((1+redshift)*time_arr_shock_opt_thick/day, mAB_arr_shock_opt_thick, linewidth=2.5, color=colour_arr[i], linestyle="-", label=model_name_arr[i])
		ax1.plot((1+redshift)*time_arr_shock_transparent/day, mAB_arr_shock_transparent, linewidth=2.5, color=colour_arr[i], linestyle=":")'''
			
	'''ax1.legend(loc="upper right", ncol=1, fontsize = 8, markerscale=0)

	plt.title("Apparent magnitude for model "+model_name, fontsize=14) # add title to the whole plots
	plt.xlabel("$t_{obs} = (1+z)t_{source}$ [days]", fontsize=14)
	plt.ylabel("$m_{AB} (1000s$ exposure) [mag]", fontsize=14)
	max_xlim=xrange[1] # year
	plt.xlim(xrange[0],xrange[1]) # time in year
	plt.ylim(yrange[0],yrange[1]) # AB magnitude
	
	# filter magnitude bounds:
	for j in range(num_filters):
		print(filter_arr[j].filter_magnitude_bound)
		# add nice-looking filter constraints
		#j = num_filters-1-j
		xarr = [j/(num_filters)*max_xlim, (j+1)/(num_filters)*max_xlim]
		yarr1 = [filter_arr[j].filter_magnitude_bound, filter_arr[j].filter_magnitude_bound]
		yarr2 = [100, 100]
		ax1.fill_between(xarr, yarr1, yarr2, alpha = 0.3, color=colour_arr[j])
		#ax1.text((j+0.5)/(num_filters)*max_xlim, filter_arr[j].filter_magnitude_bound+0.1, model_name_arr[j], fontsize=8, color="#"+mean_color(colour_arr[j],"#2f4f4f"), va="top",ha="center")
	
	picturename = 'ENT_mAB_model_'+model_name+'_z_'+str(redshift)+'.pdf'
	'''
	#plt.show()
	#plt.savefig(picturename,dpi=300,bbox_inches='tight', pad_inches=0.1) #bbox_inches='tight', pad_inches=0.016
	plt.close()
	#print("Picture saved as: " + picturename)

	# function output:
	time = ABmag_lightcurve.time_arr*(1+redshift)
	photometric_lightcurve_filter_ZTF_g_band = filter_output_arr[0]
	photometric_lightcurve_filter_ZTF_r_band = filter_output_arr[1]
	photometric_lightcurve_filter_ATLAS_cyan_band = filter_output_arr[2]
	photometric_lightcurve_filter_ATLAS_orange_band = filter_output_arr[3]

	return time, photometric_lightcurve_filter_ZTF_g_band, photometric_lightcurve_filter_ZTF_r_band, photometric_lightcurve_filter_ATLAS_cyan_band, photometric_lightcurve_filter_ATLAS_orange_band

# ==============================================================================================================================
if __name__ == "__main__":
	plot_bolometric = True
	plot_photometry = False
	
	# minimal working example for the interface:
	M_sun_gram = 1.988e33
	R_sun_cm = 6.957e10
	day = 60*60*24

	# example inputs:
	Ekin_ejecta = 2e53 # in erg
	Mass_ejecta = 100.*M_sun_gram # in gram
	Radius_initial = 2.*R_sun_cm # in cm
	csm_powerlaw_prefactor = 9e16 # Parameter A .CSM power law is rho ~ A r^(-n). in cgs units
	csm_powerlaw_index = 2.0 # parameter n. CSM power law is rho ~ A r^(-n). floating point number between 0 < n < 3 should be reasonable
	ejecta_index = 0 # ejecta power law is rho ~ (Gamma * beta)^(-n_eje). Integer n = 0, 1, 2. Other values are NOT SUPPORTED

	if plot_bolometric:
		time, Luminosity_bolometric, radius_photosphere, velocity_shock, eta_thermal_coupling_coefficient, Temperature_blackbody, Temperature_colour, optical_depth_shock_electron_scattering, nu_absorption_ff = interface_single_sms_lightcurve(Ekin_ejecta, Mass_ejecta, Radius_initial, csm_powerlaw_prefactor, csm_powerlaw_index, ejecta_index)
		# plot the quantities:
		plt.plot(time/day, Luminosity_bolometric, label="Luminosity_bolometric")
		#plt.plot(time/day, radius_photosphere, label="radius_photosphere")
		#plt.plot(time/day, velocity_shock, label="velocity_shock")
		#plt.plot(time/day, eta_thermal_coupling_coefficient, label="eta_thermal_coupling_coefficient")
		#plt.plot(time/day, Temperature_blackbody, label="Temperature_blackbody")
		#plt.plot(time/day, Temperature_colour, label="Temperature_colour")
		#plt.plot(time/day, optical_depth_shock_electron_scattering, label="optical_depth_shock_electron_scattering")
			
		plt.xlabel("time [days]")
		#plt.ylim(1e-2, 1e46)
		plt.semilogy()
		plt.legend()
		plt.show()
		plt.close()
		exit()

	if plot_photometry:
		model_name = "ENT-test"
		redshift_source = 0.995
		# inputs:
		Ekin_ejecta = 5e53 # in erg
		Mass_ejecta = 240.*M_sun_gram # in gram
		Radius_initial = 2.*R_sun_cm # in cm
		csm_powerlaw_prefactor = 2e17 # Parameter A .CSM power law is rho ~ A r^(-n). in cgs units
		csm_powerlaw_index = 2.05 # parameter n. CSM power law is rho ~ A r^(-n). floating point number between 0 < n < 3 should be reasonable
		time, photometric_lightcurve_filter_ZTF_g_band, photometric_lightcurve_filter_ZTF_r_band, photometric_lightcurve_filter_ATLAS_cyan_band, photometric_lightcurve_filter_ATLAS_orange_band = single_star_color_evolution_ZTF(model_name,Ekin_ejecta,Mass_ejecta,Radius_initial,csm_powerlaw_prefactor,csm_powerlaw_index,redshift_in=redshift_source, num_filters_in=4, xrange=[0,1500],yrange=[25,16])

		plt.plot(time/day, photometric_lightcurve_filter_ZTF_g_band, label="ZTF g band", c="green")
		plt.plot(time/day, photometric_lightcurve_filter_ZTF_r_band, label="ZTF r band", c="red")
		plt.plot(time/day, photometric_lightcurve_filter_ATLAS_cyan_band, label="ATLAS cyan band", c="cyan")
		plt.plot(time/day, photometric_lightcurve_filter_ATLAS_orange_band, label="ATLAS orange band", c="orange")
					
		plt.xlabel("$t_{obs} = (1+z)t_{source}$ [days]", fontsize=14)
		plt.ylabel("$m_{AB} (1000s$ exposure) [mag]", fontsize=14)
		plt.ylim(25, 16)
		plt.legend()
		plt.show()
		plt.close()

		print("Done!")
		exit()

	
# 						SPECTRA COMPARE
#_______________________________________________________________________

#!!!!!!!!!!! IN ORDER TO RUN, SAVE the SpeX_IRTF_Library.sav file !!!!!!  
#!!!!!!!!!!! 		and object spectrum 		!!!!!!!!!!!!!!!!!!

"""(format: object-name_band_Spectrum.txt, ex. GOI28_K1band_Spectrum.txt )"""

#!!!!!!!!!!! to the same directory where this script is saved !!!!!!!!!!!



# ==========================  PURPOSE  ===============================
# The purpose of this code is to compare spectra of
# different objects to a known database and Identify
# the object.

# ====================================================================
# --------------------------------------------------------------------
# ========================== Imports =================================

import numpy as np
import matplotlib.pyplot as plt
import time
import datetime as dt
import scipy.io as sc
import scipy.optimize as opt
import glob
import os
import sys
import pandas as pd
import matplotlib
import random
from matplotlib.axes import Subplot
from matplotlib.ticker import AutoMinorLocator
import matplotlib.colors as colors
import matplotlib.cm as cmx

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'cm'

# ====================================================================
# --------------------------------------------------------------------
# ============================= ABOUT SPECTRUM FILES =================

""" the lib_path file contains: 4 LIBRARIES
	[library_wl,library_fl,library_spt,library_name]

	library_wl — (37,5) array giving the wavelength in angstroms for 
		GPI spectra in the five bands (Y, J, H, K1, K2). 
		For now, I'll be using the H-band, so library_wl[:,2].
		(wavelength in angstrom)

	library_fl — (37,5,622) array containing the GPI resolution Y, J, H, K1 
		and K2 spectrum of the 622 comparison objects. 

	library_spt — (622) array containing the spectral type of each of the 
		comparison objects. 60 = M0, 70 = L0, 80 = T0.

	library_name — (622) array containing the filename of the original 
		spectrum of each comparison object. """
		
# ------------------------------------ TXT OBJECT DATA -----------------------------
""" + the Hband_object_old have a shape of (37,3) i.e. (wave_37,flux_37, error_37)
		(wavelength given in microns)

	+ the Hband_object_new have a shape of (36,3) i.e. (wave_36,flux_36, error_36)
		therefore the fit alpaha fucntion will not work on the current condition becasue 
		the array from the library dont have the same dimession
		(wavelength given in microns)

	+ the K1Band_object text file has a shape of (36,3) (wave_36,flux_36, error_36) 
		therefore the fit alpaha fucntion will not work on the current condition becasue
		the array from the library dont have the same dimession """
# ----------------------------------------------------------------------------
# ============================= READ IN LIBARY ===============================

# check the about spectrum files to learn more about lib_path
#lib_path = "SpeX_IRTF_Library.sav"
lib_path = "SpeX-IRTF-GPI-Resolution.sav"

# the band spectrum contains the wavelength, flux, and error for recently
# 			identified objects (wavelength in microns).
#Hband_object = "GOI28_Hband_Spectrum.txt"
#K1band_object = "GOI28_K1band_Spectrum.txt"
#K2band_object = "GOI28_K2band_Spectrum.txt"
Yband_object = "betapic_Yband_Spectrum.txt"
Jband_object = "betapic_Jband_Spectrum.txt"
Hband_object = "betapic_Hband_Spectrum.txt"
K1band_object = "betapic_K1band_Spectrum.txt"
K2band_object = "betapic_K2band_Spectrum.txt"


def read_sav(file_path):
	""" just reads in the library """
	lib	= sc.readsav( file_path)#, verbose = True)
	return lib


def read_txt(file_path):
	""" just reads in the Hband spectra file """
	spec = np.loadtxt(file_path, skiprows = 0, dtype = 'float' ,delimiter = '	')
	wave, flux, error = spec[:, 0], spec[:, 1], spec[:, 2]
	scale = np.nanmean(flux)
	flux /= scale
	error /= scale
	wave = micron_angstrom(wave)
	return wave, flux, error



def band_in_lib(library, band):
	""" reads out band (37,5) 37 not sure?
		5 is the index of band (Y,J,H,K1,K2)"""
	global lib_path

	libraries = read_sav(lib_path)
	#print(np.transpose(libraries[library][:,1,:]))
	# the [:,j] iterates over all 37 rows the the jth column (there are only 5 columns
		# corresponding to 5 bands)
	
	bands = ['Yband', 'Jband', 'Hband', 'K1band', 'K2band' ]
	lib = []

	if band in bands:
		if band == "Yband":
			print(np.shape(libraries[library]))
			lib = libraries[library][:,0,:]
			print(np.shape(lib))
		if band == "Jband":
			lib = libraries[library][:,1,:]
		if band == "Hband":
			lib = libraries[library][:,2,:]
		if band == "K1band":
			lib = libraries[library][:,3,:]
		if band == "K2band":
			lib = libraries[library][:,4,:]
	elif band == 'all':
		for i in range(5):
			lib.append(libraries[library][:,i,:])
		#lib = np.reshape(lib,(185,812))
		lib = np.reshape(lib,(len(lib)*len(lib[0]),len(lib[0][0])))

	else:
		print("BAND NOT IN LIBRARY")
	
	return lib

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ========================= Convert to proper units ==========================

def micron_angstrom(data):
	# if wave length is in microns use this to convert to angstrom
	flux = data*1e4
	return flux

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ========================= scale factor and chi**2  ==========================


def fit_alpha(alpha,model,obs,error):
	return np.nansum((((alpha*model)-obs)/error)**2.0)

def alpha_to_num(name):
	name = name.decode("utf-8")
	x = name[0]
	if x == "B":
		num = 10.
	if x == "A":
		num = 20.
	if x == "F":
		num = 30.
	if x == "G":
		num = 40.
	if x == "K":
		num = 50.
	if x == "M":
		num = 60.
	if x == "L":
		num = 70.
	if x == "T":
		num = 80.
	return num + float(name[1:])


def fit_alpha2(p, model, obs, error):
	#p[0] contains the normal alpha parameter to be applied to all bands
	#p[1] Yband adjustment
	#p[2] Jband adjustment
	#p[3] Hband adjustment
	#p[4] K1band adjustment
	#p[5] K2band adjustment

	#make a copy so original isn't modified, not sure if this is necessary!
	this_obs = np.copy(obs)
	this_obs[0:37] *= p[1]
	this_obs[37:74] *= p[2]
	this_obs[74:111] *= p[3]
	this_obs[111:148] *= p[4]
	this_obs[148:185] *= p[5]

	chi2 = np.nansum((((model*p[0])-this_obs)/error)**2.0)
	
	#Count number of non-nans in the flux array
	len_y = np.sum(np.isfinite(this_obs[0:37]*model[0:37]))
	len_j = np.sum(np.isfinite(this_obs[37:74]*model[37:74]))
	len_h = np.sum(np.isfinite(this_obs[74:111]*model[74:111]))
	len_k1 = np.sum(np.isfinite(this_obs[111:148]*model[111:148]))
	len_k2 = np.sum(np.isfinite(this_obs[148:185]*model[148:185]))

	#we take the logarithm of the adjustment because we are comparing to the uncertainties in the Maire et al. 2014 GPI paper which are in magnitudes
	#Using mag_a - mag_b = -2.5*log10(flux_a/flux_b), if you set flux_b = 1 (i.e. no adjustment to the flux in that band) 
	#then the adjustment in magnitudes is mag_a — mag_b = -2.5*log10(flux_a), which is your observed-expected (mag_a - mag_b) term on the top of the chi2 function.
	#this is then divided by the uncertainty on the spot ratio in each band, which is hard-coded below (0.05mag in Y, 0.03mag in J...)
	#it is then multiplied by the number of non-nans so it properly weights this constraint (otherwise it would only cause a very small change in the final chi2). 

	chi2 += np.nansum((((-2.5*np.log10(p[1:6])) / np.array([0.05, 0.03, 0.06, 0.07, 0.21]))**2.0) * np.array([len_y, len_j, len_h, len_k1, len_k2]))



	return chi2


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ========================= RESIDUALS AND CHI**2 ===============================

def optimize(object_path,lib_band,image=False,specify_compare=None):
	global lib_path
	print('optimizing: '+ str(lib_band)+'...')
	print('saving compared images: ' + str(image))
	library = read_sav(lib_path)
	#spt = library["library_spt"]
	#spt1 = library["optical_spt"] #i commented this out becaue we are changing the order of optical_spt and ir_spt
	#spt2 = library["ir_spt"]
	spt1 = library["ir_spt"]
	spt2 = library["optical_spt"]
	spt3 = library["simbad_spt"]
	grav1 = library["grav_a"]
	grav2 = library["grav_b"]
	grav3 = library["grav_c"]
	name = library["simbad_name"]
	lum = library["simbad_lum"]

	flux = read_sav(lib_path)['gpi_fl']

	symbol = []
	
	spt_final = np.array([])
	
	object_wave,object_fl,object_error = read_txt(object_path)
	#print(np.shape(object_fl), 'object')
	
	object_name = name_slplit(object_path)[0]
	newpath = str(object_name)+'_'+str(lib_band)+'_image'
	gpi_flux = np.transpose(band_in_lib('gpi_fl', lib_band))
	print(np.shape(band_in_lib('gpi_fl', lib_band)))
	print(np.shape(gpi_flux),'gpi_flux')
	print(k)
	#if specify_compare != None:
	#	print('comparing object to specific gpi library object: ' + specify_compare)
	#	gpi_flux = gpi_flux[:,][np.where(name == str.encode(specify_compare))][0]
	#	lib_band = 'all'
	#print(np.shape(gpi_flux))
	#print(np.shape(gpig_flux))
	#print(t)
	resid = np.array([])
	n_dof = np.array([])
	adjust_y, adjust_j, adjust_h, adjust_k1, adjust_k2 = [],[],[],[],[]
	all_alpha = []

	n = 0
	for i in gpi_flux:
		if spt1[n].decode("utf-8") == "NULL":
			if spt2[n].decode("utf-8") == "NULL":
				if spt3[n].decode("utf-8") == "NULL":
					spt_final = np.append(spt_final,np.nan)
				else:
					spt_final = np.append(spt_final,alpha_to_num(spt3[n]) )
			else:
				spt_final = np.append(spt_final,alpha_to_num(spt2[n]) )
		else:
			spt_final = np.append(spt_final,alpha_to_num(spt1[n]) )

		if grav1[n].decode("utf-8") == 'gamma (opt)' or grav2[n].decode("utf-8") == "gamma (opt)" or grav3[n].decode("utf-8") == "gamma (opt)" or \
		   grav1[n].decode("utf-8") == 'delta (opt)' or grav2[n].decode("utf-8") == "delta (opt)" or grav3[n].decode("utf-8") == "delta (opt)" or \
		   grav1[n].decode("utf-8") == 'VL-G' or grav2[n].decode("utf-8") == "VL-G" or grav3[n].decode("utf-8") == "VL-G":
			
			symbol = np.append(symbol, "*")


		elif grav1[n].decode("utf-8") == 'beta (opt)' or grav2[n].decode("utf-8") == "beta (opt)" or grav3[n].decode("utf-8") == "beta (opt)" or \
		   grav1[n].decode("utf-8") == 'INT-G' or grav2[n].decode("utf-8") == "INT-G" or grav3[n].decode("utf-8") == "INT-G":
			
			symbol = np.append(symbol, "D")

		elif grav1[n].decode("utf-8") == 'alpha (opt)' or grav2[n].decode("utf-8") == "alpha (opt)" or grav3[n].decode("utf-8") == "alpha (opt)" or \
		   grav1[n].decode("utf-8") == 'FLD-G' or grav2[n].decode("utf-8") == "FLD-G" or grav3[n].decode("utf-8") == "FLD-G":
			
			symbol = np.append(symbol, "s")

		else:
			symbol = np.append(symbol, "o")	

		if np.nansum(i) == 0.:
			#If there is no flux for the comparison object
			resid = np.append(resid,np.nan)
			n_dof = np.append(n_dof, np.nan)
			adjust_y.append(np.nan)
			adjust_j.append(np.nan)
			adjust_h.append(np.nan)
			adjust_k1.append(np.nan)
			adjust_k2.append(np.nan)
			all_alpha.append(np.nan)
		else:
			guess = np.nanmean(object_fl)/np.nanmean(i)
			index = np.isfinite(object_fl) & np.isfinite(i)


			if lib_band != 'all':
				#If fitting only one band, we don't need to run opt.minimize, since we can simply differentiate the chi2 formula
				this_dof = np.sum(index) - 2 #n_dof = no. measurements - no. fitted parameters - 1 
				alpha = np.nansum((object_fl[index] * i[index])/(object_error[index]**2.0)) / np.nansum((i[index]**2.0)/(object_error[index]**2.0))
				chi2 = np.nansum(((object_fl[index] - (alpha*i[index]))**2.0)/(object_error[index]**2.0)) / this_dof
				adjust_y.append(np.nan)
				adjust_j.append(np.nan)
				adjust_h.append(np.nan)
				adjust_k1.append(np.nan)
				adjust_k2.append(np.nan)
				
					
			else:
				#If fitting all bands, we will have to optimize as we need to fit more than one parameter...
				this_dof = np.sum(index) - 7 #This will change once we add the adjustment parameters
				result = opt.minimize(fit_alpha2, [guess,1.,1.,1.,1.,1.], args=(i, object_fl, object_error), bounds=((1e-20, 1e20),(0.5,2),(0.5,2),(0.5,2),(0.5,2),(0.5,2)))
				#result = opt.minimize(fit_alpha, guess, args=(i[index], object_fl[index], object_error[index]), bounds=((1e-20, 1e20),))
				#print(result['status'])
				#print(results.keys())
				#print(result['x'])
				#print(np.shape(results['x']))
				#print(t)
				alpha = result['x'][0]
				chi2 = result['fun'] / this_dof
				adjust_y.append( result['x'][1])
				#print(adjust_y)
				adjust_j.append( result['x'][2])
				adjust_h.append( result['x'][3])
				adjust_k1.append( result['x'][4])
				adjust_k2.append( result['x'][5])

			#ignore object if it is a subdwarf
			if lum[n].decode("utf-8") == "sd" or lum[n].decode("utf-8") == "esd":
				chi2 = np.nan
				this_dof = np.nan

			resid = np.append(resid,chi2)
			n_dof = np.append(n_dof, this_dof)
			all_alpha +=  [alpha]

		#savefig = spectrum_image(image,object_wave,object_fl,alpha[0],i,n,lib_band,newpath,object_error)		
		n+=1
	#print(np.shape(adjust_y))
	#print(adjust_y)
	adjust = [adjust_y] + [adjust_j] + [adjust_h] + [adjust_k1] + [adjust_k2]
	print(np.shape(adjust))
	#print(adjust)
	print(np.shape(all_alpha),'this is the shape of all alpha')
	#print(np.shape(adjust[0]))

	return [spt_final, resid, symbol,name, n_dof, adjust, all_alpha] 


#yband*alpha*adjust[0]

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ========================= PLOTTING FUCNTIONS ===============================



def spectrum_image(image,object_wave,object_flux,alpha,library_fluxi,n,lib_band,newpath,object_error,ax = plt.subplots(2,sharex=True,figsize =(6,7.5),squeeze=False)[1],shift=0,color='red'):
	""" this function plots and saves images wavelength vs flux for both
		object  and spectrum. The images are save to current directory.
		"""
	if image == True:
		#print('saving '+ str(lib_band)+ ' images...')

		ax.plot(object_wave/(1e4),(library_fluxi)+shift,color= color,linewidth=3,zorder=-10)
		ax.errorbar(object_wave/(1e4),(object_flux/alpha) + shift, yerr=(object_error/alpha), color = 'k',fmt='.',capsize=0,markersize=4)
		#ax.set_xlabel('Wavelength ($\mu$m)')
		ax.set_ylabel('Normalized Flux ')
		#ax.ylim(0,1.2)
		ax.set_xlim(.9,2.45)
		#ax.title('Flux vs wavelength: '+ str(lib_band), fontsize=20)
		#ax.legend(loc='best')
		#ax.tight_layout()
		#ax.savefig(str(newpath)+'/spectrum_' +str(lib_band)+'_' +str(n)+'.png',bbox_inches='tight')
		#ax.show()
		#ax.close()







#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#===================================== TEST FUNCTION ==================================================


# next step is to loop optimize function through different bands.
# def optimize(object_path,lib_band,image=False):
#library bands = ['Y', 'J', 'H', 'K1', 'K2' ]
# object bands = path_H, path_k1,path_k2 "../RAW_DATA/GOI28_Hband_Spectrum_NEW.txt"

def opto_loop(objetname,image=False, allbands=False):
	print('Object files must have this format: ' 
		+'obj-name_band_Spectrum.txt, Ex: GOI28_K1band_Spectrum.txt')

	#if allbands == True:
	#	band = 'all'
	#	_path = 
	#
	#spt,resid,symbol,name = optimize(i,band,image)

	path_object_bands = multiple_files(objetname+'_*band_Spectrum.txt')
	fig, ax = plt.subplots(5,sharex=True, figsize=(6,14),squeeze=False)
	fig.subplots_adjust(wspace=0, hspace=0)
	for k in range(5):
		ax[k][0].set_ylim(0.1,400.) # this is the ylim for the 5 panels
	
	#subplot_num = 0
	#bands = ['Y', 'J', 'H', 'K1', 'K2' ]

	#First loop fit bands individually to average the five chi2
	resid5 = []
	n_dof5 = []
	for i in path_object_bands:
		band = name_slplit(i)[1]
		data = optimize(i,band,image)
		spt,resid,symbol,name, n_dof, adjust, all_alpha = data[0] , data[1] , data[2] , data[3] , data[4] , data[5] , data[6]
		resid5 += [resid] 
		n_dof5 += [n_dof]
		opt_plot(spt,resid,symbol,band,ax,fig,name)

	#This throws a 'mean of empty slice' error because subdwarfs have no chi2 in any band.
	resid5 = np.array(resid5)
	n_dof5 = np.array(n_dof5)
	resid5 *= n_dof5 #We need to do this to convert it back to non-reduced chi2

	resid5 = np.nansum(resid5,axis=0) #Sum of non-reduced chi2
	resid5 /= (np.nansum(n_dof5, axis=0)+4.0) #Re-reduce using sum of degrees of freedom. The four here is to take into the account of the -1 in nu = N-n-1, which we don't want to count 4 times 
	resid5[np.where(resid5 == 0.0)] = np.nan

	plt.savefig(str(objetname)+'individual.pdf',format='pdf',bbox_inches='tight')

	path_object_bands = multiple_files(objetname+'_full_Spectrum.txt')
	fig,ax = plt.subplots(2,sharex=True,figsize =(6,7.5),squeeze=False)
	fig.subplots_adjust(wspace=0, hspace=0)
	for i in range(2):
		ax[i][0].set_ylim(0.3,200) # this is the y lim for the 2 panels 

	band = 'all'
	for i in path_object_bands:
		#spt,resid,symbol,name, n_dof, adjust, all_alpha = optimize(i,band,image)
		data = optimize(i,band,image)
		spt,resid,symbol,name, n_dof, adjust, all_alpha = data[0] , data[1] , data[2] , data[3] , data[4] , data[5] , data[6]

		opt_plot(spt,resid,symbol,band,ax,fig,name)
	opt_plot(spt,resid5,symbol,'resid5',ax,fig,name)

	plt.savefig(str(objetname)+'all_bands.pdf',format='pdf',bbox_inches='tight')

	#plt.savefig('test.png',dpi=300)
	#plt.savefig(str(objetname)+'_allbands_fitted_.pdf',format='pdf')#,dpi=300)
	#return plt.show()

    
def multiple_files(object_name):
	# this function locates the path of multiple files, i.e, 
	#	object spectrum
	philes = sorted(glob.glob(object_name))
	print ( 'number of band files: ', str(len(philes)))
	return philes
	
def name_slplit(fyle): 
	# splits the filename of object (SUB FUNCTION)
    fyle_band = fyle.split("_")
    return fyle_band



def opt_plot(spt,resid,symbol,lib_band,ax,fig,name):

	""" this fucntion plots the reduced chi**2 vs spectral type"""
	if lib_band == "Yband":
		plotnum = 0
	elif lib_band == "Jband":
		plotnum = 1
	elif lib_band == "Hband":
		plotnum = 2
	elif lib_band == "K1band":
		plotnum = 3
	elif lib_band == "K2band":
		plotnum = 4
	elif lib_band == "all":
		plotnum = 0
	elif lib_band == 'resid5':
		plotnum = 1


	for i in ["*", "D", "s", "o"]:
		index = np.where(symbol==i)
	
		if i == "*" :
			zorder = 10
			color = "yellow"
			alpha = 1
			markersize = 9
			label = r'$\gamma$ / $\delta$ / VL-G'
			markeredgecolor = 'black'
		
		elif i == "D":
			zorder = 9
			color = "green"
			alpha = 1
			markersize = 5
			label = r"$\beta$ / INT-G"
			markeredgecolor = 'black'
		
		elif i == "s":
			zorder = 8
			color = "red"
			alpha = 1
			markersize = 5
			label = r"$\alpha$ / FLD-G"
			markeredgecolor = 'black'
		
		else:
			zorder = 7
			color = "blue"
			alpha = .2
			markersize = 5
			label = ""
			markeredgecolor = 'none'

		ax[plotnum][0].plot(spt[index],resid[index],i,color=color,zorder = zorder,alpha= alpha,markersize = markersize,label=label,mec=markeredgecolor)
	x = [60,70,80,90]
	minorLocator = AutoMinorLocator()
	band_annot = ['Y', 'J' , 'H', 'K1', 'K2']


	fig.canvas.draw()
	labels = [item.get_text() for item in ax[plotnum][0].get_xticklabels()]
	#x = range(60,91)
	#labels = ['F0', 'G0', 'K0', 'M0', 'L0', 'T0', 'Y0']
	labels = ['M0', 'L0', 'T0', 'Y0']
	ax[plotnum][0].set_xticks(x)
	#ax[plotnum][0].set_xticks(minor_ticks,minor=True)
	ax[plotnum][0].set_xticklabels(labels)
	ax[plotnum][0].xaxis.set_minor_locator(minorLocator)

	#ax.set_title( lib_band , fontsize = 20)
	if plotnum == 4 and lib_band != 'all':
		ax[plotnum][0].set_xlabel('Spectral Type')
	if plotnum == 0 and lib_band != 'all':
		lgnd = ax[plotnum][0].legend(numpoints=1,fontsize=9,ncol=3,labelspacing=-1,bbox_to_anchor=(0.10, .95,4,3), loc=3) #was -0.019
		lgnd.legendHandles[0]._legmarker.set_markersize(9)
		lgnd.legendHandles[1]._legmarker.set_markersize(6)
		lgnd.legendHandles[2]._legmarker.set_markersize(6)
		for i in range(5):
			ax[i][0].annotate(band_annot[i],xy=(62,100))


	if plotnum == 0 and lib_band == 'all':
		ax[1][0].set_xlabel('Spectral Type')
		#lgnd = ax[plotnum][0].legend(numpoints=1,fontsize=8,ncol=3,labelspacing=-1,prop={'size':10.02},bbox_to_anchor=(-0.019, .95,4,3), loc=3)
		#lgnd.legendHandles[0]._legmarker.set_markersize(9)
		#lgnd.legendHandles[1]._legmarker.set_markersize(6)
		#lgnd.legendHandles[2]._legmarker.set_markersize(6)

	ax[plotnum][0].set_ylabel(r'$\chi^{2}_{\nu}$')
	ax[plotnum][0].set_yscale('log')
	ax[plotnum][0].set_xlim(60,90)



	#plt.tight_layout()

	"""spt_indx = sorted(spt[np.where(spt>=60)])
	spt_val = []
	for i in spt_indx:
		if i not in spt_val:
			spt_val.append(i)

	mean_spt =[]
	for i in spt_val:
		meanspt = np.nanmean(resid[np.where(spt==i)])
		mean_spt.append(meanspt)
	spt_val = np.array(spt_val)
	mean_spt = np.array(mean_spt)
	print(np.shape(spt_val),'sptvalshape')
	print(np.shape(mean_spt),'mean shape')

	fit = np.polyfit(spt_val,mean_spt, 3)
	yfit = fit[0]*spt_val**3 + fit[1]*spt_val**2 + fit[2]*spt_val + fit[3]
	#ax[plotnum][0].set_ylim(min(spt)*2e-2,max(fit)+10)	
	
	#ax[plotnum][0].plot(spt_val,yfit,'--r',markersize =20)
	yannot_loc = []
	xannot_loc = []
	name_loc =[]
	#print(sorted(resid)[0:10])
	#for k in sorted(resid)[0:10]:
	#	print(name[np.where(resid == k)])

	for i in range (len(yfit)):
		nameindx = name[np.where(spt==spt_val[i])]
		resid_loc = resid[np.where(spt==spt_val[i])]
		for j in resid_loc:
			if j > 2.*yfit[i] and j > 19:
				yannot_loc.append(j)
				xannot_loc.append(spt_val[i])
				name_loc.append(nameindx[np.where(resid_loc == j)])

			elif j < 0.25*yfit[i]:
				yannot_loc.append(j)
				xannot_loc.append(spt_val[i])
				name_loc.append(nameindx[np.where(resid_loc == j)])
	anotate_loc = np.vstack((np.array(xannot_loc),np.array(yannot_loc))).T"""
	#print(sorted(resid)[:5],'SORTED STUFF')

	#ax[plotnum][0].set_ylim(0.3,400.) # this is the ylim for the 5 panels
	#ax[plotnum][0].set_ylim(0.8,200) # this is the y lim for the 2 panels
	resid_copy = np.copy(resid)
	resid_copy[np.isnan(resid_copy)] = 1e1000
	res_sort = sorted(resid_copy)[:5]
	u = 0
	for i in res_sort:
		#print(str(i))
		#print(np.shape(res_sort))
		#print(u,'first five min chi**2 of each plot')
		if str(i) != 'nan':
	
			y_indx = resid[np.where(resid == i)]
			x_indx = spt[np.where(resid == i)]
			name_loc = name[np.where(resid == i)]
			#print(x_indx, y_indx)
			print(name_loc)
			print(i)
			#ax[plotnum][0].annotate(name_loc[0],xy=(x_indx[0],y_indx[0]),fontsize=2)
		#u+=1
	#print(t)



	#for i in range(len(name_loc)):

	#	ax[plotnum][0].annotate(name_loc[i],xy=(anotate_loc[:,0][i],anotate_loc[:,1][i]),fontsize=3)


def new_folder(object_name,band,image=False):
	if image == True:
		newpath = str(object_name)+'_'+str(band) + '_image'
		if not os.path.exists(newpath):
			os.makedirs(newpath)

def best_fit(ax):
	global lib_path
	name = read_sav(lib_path)["simbad_name"]
	flux = read_sav(lib_path)['gpi_fl']

	lib_band = ['Yband','Jband','Hband','K1band','K2band']

	name_of_min_chi2 = [[b'2MASS J01262109+1428057'], [b'2MASSI J0536199-192039'], [b'2MASS J04062677-3812102'], [b'2MASS J03552337+1133437'], [b'2MUCD 11538']]

	#for compare_object in name_of_min_chi2:
	#	print(np.where(name == i ))
	#for i in range(0,5):
	i = 0
	#plt.figure(figsize=(12,4))
	#fig,ax = plt.subplots(2,sharex=True,figsize =(12,4),squeeze=False)


	for path in lib_band:

		object_wavelength, object_flux, object_error = read_txt('newbetapic_'+path+'_Spectrum.txt')
		gpi_flux = np.transpose(flux[:,i,np.where(name == b'2MASS J01262109+1428057')])
		resid = np.array([])				
		for j in gpi_flux:
			if np.nansum(j) == 0.:
				#This should never happen
				resid = np.append(resid,np.nan)
			else:
				guess = np.nanmean(object_flux)/np.nanmean(j[0])
				index = np.isfinite(object_flux) & np.isfinite(j[0])

				if lib_band != 'all':
					#If fitting only one band, we don't need to run opt.minimize, since we can simply differentiate the chi2 formula
					this_dof = np.sum(index) - 2 #n_dof = no. measurements - no. fitted parameters - 1 
					alpha = np.nansum((object_flux[index] * j[0][index])/(object_error[index]**2.0)) / np.nansum((j[0][index]**2.0)/(object_error[index]**2.0))
					chi2 = np.nansum(((object_flux[index] - (alpha*j[0][index]))**2.0)/(object_error[index]**2.0)) / this_dof
				else:
					#If fitting all bands, we will have to optimize as we need to fit more than one parameter...
					this_dof = np.sum(index) - 2 #This will change once we add the adjustment parameters
					result = opt.minimize(fit_alpha, guess, args=(j[0][index], object_flux[index], object_error[index]), bounds=((1e-20, 1e20),))
					alpha = result['x'][0]
					chi2 = result['fun']/this_dof
				resid = np.append(resid,chi2)

		spectrum_image(True,object_wavelength,object_flux,alpha,j[0],i,lib_band,'empty',object_error,ax)
		#	plt.show()
		#spectrum_image(image,object_wave,object_flux,alpha,library_fluxi,n,lib_band,newpath,object_error):
		#spectrum_image(image,goi28_wave,goi28_fl,alpha[0],i,n,lib_band,newpath,goi28_error)		
		#


		i +=1
	ax.plot(object_wavelength/(1e4),(gpi_flux[0][0]), label='2MASS J01262109+1428057 (L4$\gamma$) ',color= 'red',linewidth=3,zorder=-10,markersize = -1)
	ax.errorbar(object_wavelength/(1e4),(object_flux/alpha), label = r'$\beta$ Pic b (unrestricted fit)',yerr=(object_error/alpha), color = 'k',fmt='.',capsize=0,markersize=4)
	ax.legend(loc='upper right',numpoints=1,fontsize=7.5)
	#plt.show()
	#plt.savefig('best_fit_field_object.pdf',bbox_inches='tight')

def best_fit2():
	bands = ['Yband','Jband','Hband', 'K1band', 'K2band']
	path_object_bands = 'newbetapic_full_Spectrum.txt'
	flux = read_sav(lib_path)['gpi_fl']
	plt.figure(1)
	fig,ax = plt.subplots(2,sharex=True,figsize =(12,4.5),squeeze=False)
	fig.subplots_adjust(wspace=0, hspace=0)
	#ax = ax[0]
	print(np.shape(ax))
	#print(ax[0])
	#print(t)
	
	best_fit(ax[1][0])
	ax[1][0].set_ylim(.1,1.3)
	ax[1][0].set_xlabel('Wavelength ($\mu$m)')



	#New line to save full spectrum
	full_wl, full_fl, full_err = read_txt('newbetapic_full_Spectrum.txt')

	i = 0
	#data =
	image=False
	band = 'all'
	
	#spt,resid,symbol,name, n_dof, adjust, all_alpha = optimize(i,band,image)
	data = optimize(path_object_bands,band,image)
	#print(np.shape(data))
	spt,resid,symbol,name, n_dof, adjust, all_alpha = data[0] , data[1] , data[2] , data[3] , data[4] , data[5] , data[6]
	#return data
	for band in bands:
		index = np.where(name == b'2MASS J04062677-3812102')[0][0]
		gpi_flux = np.transpose(flux[:,i,index])

		#We don't need to read in the individual bands anymore

		#object_wavelength, object_flux, object_error = read_txt('newbetapic_'+band+'_Spectrum.txt')
		#new lines:
		if band == 'Yband':
			min_index, max_index = 0, 37
		if band == 'Jband':
			min_index, max_index = 37, 74
		if band == 'Hband':
			min_index, max_index = 74, 111
		if band == 'K1band':
			min_index, max_index = 111, 148
		if band == 'K2band':
			min_index, max_index = 148, 185

		object_wavelength = full_wl[min_index:max_index]
		object_flux = full_fl[min_index:max_index]
		object_error = full_err[min_index:max_index]




		spectrum_image(True,object_wavelength,object_flux,all_alpha[index]/adjust[i][index],gpi_flux,i,bands,'empty',object_error,ax[0][0])
		#spectrum_image(True,object_wavelength,object_flux,alpha,j[0],i,lib_band,'empty',object_error)
		# j i the comparison objectflux
		ax[0][0].set_ylim(.2,1.4)
		ax[0][0].set_xlabel('Wavelength ($\mu$m)')
		print(adjust[i][index])
		i +=1
	ax[0][0].plot(object_wavelength/(1e4),gpi_flux, label='2MASS J04062677-3812102 (L2$\gamma$)',color= 'red',linewidth=3,zorder=-10,markersize = -1)
	ax[0][0].errorbar(object_wavelength/(1e4),(object_flux/(all_alpha[index]/adjust[i-1][index])), label = r'$\beta$ Pic b (restricted fit) ',yerr=(object_error/(all_alpha[index]/adjust[i-1][index])), color = 'k',fmt='.',capsize=0,ms=4)
	ax[0][0].legend(loc='upper right',numpoints=1,fontsize=7.7)
	#plt.show()
	plt.savefig('flux_vs_wave.pdf',format='pdf',bbox_inches='tight')



def mult_best_fit():
	comparison_objs = [b'2MASS J22134491-2136079',b'2MASSI J0518461-275645',b'2MASSI J0536199-192039',b'2MASSW J2208136+292121',b'2MASS J15515237+0941148',b'2MASS J22443167+2043433']
	spt_objs = ['L0', 'L1', 'L2', 'L3', 'L4', 'L6'][::-1]
	bands = ['Yband','Jband','Hband', 'K1band', 'K2band']
	path_object_bands = 'newbetapic_full_Spectrum.txt'
	flux = read_sav(lib_path)['gpi_fl']

	fig,ax = plt.subplots(1,sharex=True,figsize =(6,8),squeeze=False)
	fig.subplots_adjust(wspace=0, hspace=0)
	jet = plt.get_cmap('jet')
	cnorm = colors.Normalize(vmin=0, vmax=6)
	scalarmap = cmx.ScalarMappable(norm=cnorm, cmap = jet)
	#print(scalarmap)
	print(scalarmap.get_clim())


	
	print(np.shape(ax), 'ax shape')
	print(ax)
	
	ax[0][0].set_xlabel('Wavelength ($\mu$m)')



	#New line to save full spectrum
	full_wl, full_fl, full_err = read_txt('newbetapic_full_Spectrum.txt')

	shift = 0
	image=False
	band = 'all'
	
	data = optimize(path_object_bands,band,image)
	spt,resid,symbol,name, n_dof, adjust, all_alpha = data[0] , data[1] , data[2] , data[3] , data[4] , data[5] , data[6]
	for objct in comparison_objs[::-1]: # reversing the order of the list of comparison objects.
		i = 0
		colorval = scalarmap.to_rgba(shift)
		for band in bands:
			index = np.where(name == objct)[0][0]
			gpi_flux = np.transpose(flux[:,i,index])
	
			#We don't need to read in the individual bands anymore
	
			#object_wavelength, object_flux, object_error = read_txt('newbetapic_'+band+'_Spectrum.txt')
			#new lines:
			if band == 'Yband':
				min_index, max_index = 0, 37
			if band == 'Jband':
				min_index, max_index = 37, 74
			if band == 'Hband':
				min_index, max_index = 74, 111
			if band == 'K1band':
				min_index, max_index = 111, 148
			if band == 'K2band':
				min_index, max_index = 148, 185
	
			object_wavelength = full_wl[min_index:max_index]
			object_flux = full_fl[min_index:max_index]
			object_error = full_err[min_index:max_index]
	
	
	
	
			spectrum_image(True,object_wavelength,object_flux,(all_alpha[index]/adjust[i][index]),gpi_flux,shift,bands,'empty',object_error ,ax[0][0],shift,color=colorval)
			#spectrum_image(True,object_wavelength,object_flux,alpha,j[0],i,lib_band,'empty',object_error)
			# j i the comparison objectflux
			#ax[0][0].set_ylim(.2,1.3)
			ax[0][0].set_xlabel('Wavelength ($\mu$m)')

			print(adjust[i][index])
			i +=1
		ax[0][0].annotate(str(objct)[2:-1]+ ' ('+  spt_objs[shift]+r'$\gamma$)',(1.4,shift + 1.15))
		shift +=1
	#ax[1][0].plot(object_wavelength/(1e4),gpi_flux, label='2MASS J01262109+1428057 (L4$\gamma$)',color= 'red',linewidth=3,zorder=-10,markersize = -1)
	#ax[1][0].errorbar(object_wavelength/(1e4),(object_flux/(all_alpha[index]/adjust[i-1][index])), label = r'$\beta$Pic (unrestricted fit)',yerr=(object_error/(all_alpha[index]/adjust[i-1][index])), color = 'k',fmt='.')
	#ax[1][0].legend(loc='lower right',numpoints=1,fontsize=7.5)
	#plt.show()
	ax[0][0].set_ylim(.2,6.4)
	plt.savefig('six_flux_vs_wave.pdf',format='pdf',bbox_inches='tight')











def main():
	opto_loop('newbetapic')
	#best_fit2()
	#mult_best_fit()
	#lib = read_sav(lib_path)
	#print(lib.keys())
	#print(np.shape(lib['gpi_fl']))
	#wv = lib['gpi_wl']
	#firstband = sorted(wv[0:37])
	#print(firstband[0], firstband[-1])
 


if __name__ == '__main__':
    main()






	


#x = [[ 64.5        ,  1.51709359], [ 65.         ,  2.35534251], [ 66.         ,  1.79108853], [ 66.         ,  0.87947826], [ 66.5        ,  0.77684135], [ 66.5        ,  0.69964333], [ 67.         ,  0.3706375 ], [ 67.         ,  1.91654344], [ 68.         ,  0.29918653], [ 69.         ,  1.10067909], [ 69.         ,  1.08245593], [ 69.         ,  0.93905376], [ 69.         ,  0.89801884], [ 69.         ,  0.57302615], [ 69.5        ,  0.60884016], [ 70.         ,  0.59034063], [ 70.         ,  0.38358821], [ 70.         ,  0.43170642], [ 70.         ,  0.42206791]]








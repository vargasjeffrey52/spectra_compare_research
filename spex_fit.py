import numpy as np
from scipy.io import readsav
import scipy.optimize as opt
import multiprocessing as mp
import matplotlib.pyplot as plt


#=====================library/object_data=====================
lib_path = "SpeX-IRTF-GPI-Resolution.sav"
Yband_object = "betapic_Yband_Spectrum.txt"
Jband_object = "betapic_Jband_Spectrum.txt"
Hband_object = "betapic_Hband_Spectrum.txt"
K1band_object = "betapic_K1band_Spectrum.txt"
K2band_object = "betapic_K2band_Spectrum.txt"
# ============================================================

def optimize_all(library, data, object_band, ax_chi, ax_best, restricted=False):

    #If restricted == True, use adjustment factor

    object_wl = np.copy(data[:,0])
    object_fl = np.copy(data[:,1])
    object_err = np.copy(data[:,2])

    #        Y  J  H K1 K2
    bands = [0, 1, 2, 3, 4]
    used_bands = np.unique(object_band)

    spt1 = library["ir_spt"]
    spt2 = library["optical_spt"]
    spt3 = library["simbad_spt"]
    grav1 = library["grav_a"]
    grav2 = library["grav_b"]
    grav3 = library["grav_c"]
    name = library["adopted_name"]
    lum = library["simbad_lum"]

    comparison_flux = np.transpose(library['gpi_fl'], axes = (1, 0, 2))
    s = np.shape(comparison_flux)
    n_comparison = s[2]

    symbol = np.zeros(n_comparison, dtype=str)
    spt_final = np.zeros(n_comparison, dtype = np.float64)
    for i in xrange(0, n_comparison):
        spt_final[i], symbol[i] = spt_symbol(spt1[i], spt2[i], spt3[i], grav1[i], grav2[i], grav3[i])

    if restricted is True:

        chi2 = np.zeros(n_comparison, dtype=np.float64) * np.nan
        n_dof = np.zeros(n_comparison, dtype=np.float64) * np.nan
        adjust = np.zeros((len(used_bands), n_comparison), dtype=np.float64) * np.nan
        alpha = np.zeros(n_comparison, dtype=np.float64) * np.nan

        cores = mp.cpu_count()
        chunks = int(np.ceil(float(n_comparison) / float(cores))) #Submit this many jobs per core

        pool = mp.Pool()
        result = [pool.apply_async(optimize_all_mp, args = (start, object_fl, object_err, object_band, used_bands, comparison_flux, chunks, n_comparison)) for start in xrange(0, n_comparison, chunks)]
        output = [p.get() for p in result]
       
        s = np.shape(output)
        for i in xrange(0, s[0]):
            index = output[i][0]
            alpha[index] = output[i][1]
            n_dof[index] = output[i][2]
            chi2[index] = output[i][3]
            for j in xrange(0, len(index)):
                adjust[:, index[j]] = output[i][4][j]

        pool.close()
        pool.join()

        for i in ["*", "D", "s", "o"]:
            index = np.where((symbol==i) & np.isfinite(spt) & np.isfinite(resid))
        
            if i == "*" :
                zorder = 10
                color = "yellow"
                alp = 1
                markersize = 9
                label = r'$\gamma$ / $\delta$ / VL-G'
                markeredgecolor = 'black'
            
            elif i == "D":
                zorder = 9
                color = "green"
                alp = 1
                markersize = 5
                label = r"$\beta$ / INT-G"
                markeredgecolor = 'black'
            
            elif i == "s":
                zorder = 8
                color = "red"
                alp = 1
                markersize = 5
                label = r"$\alpha$ / FLD-G"
                markeredgecolor = 'black'
            
            else:
                zorder = 7
                color = "blue"
                alp = .2
                markersize = 5
                label = ""
                markeredgecolor = 'none'

            ax_chi.plot(spt_final[index],chi2[index],i,color=color,zorder = zorder,alpha= alp,markersize = markersize,label=label,mec=markeredgecolor) 

        ax_chi.set_ylim([np.nanmin(chi2*0.8), np.nanmax(chi2*1.2)])

        #Now to plot the best fit
        chi2[np.where(chi2 == 0.0)] = 1e10
        top_ind = np.argsort(chi2)[:5] #Contains top 5

        ax_best.plot(object_wl, np.concatenate(comparison_flux[np.ndarray.tolist(used_bands), :, top_ind[0]]), color='red', linewidth=3, zorder = 10)

        min_y = []
        max_y = []
        i = 0
        for this_band in used_bands:
            ind = np.where(object_band == this_band)
            ax_best.errorbar(object_wl[ind], object_fl[ind]/(alpha[top_ind[0]]/adjust[i,top_ind[0]]), yerr=object_err[ind]/(alpha[top_ind[0]]/adjust[i,top_ind[0]]), fmt='.', color='k', zorder = 50, capsize=0, ms=4)
            min_y += [object_fl[ind]/(alpha[top_ind[0]]/adjust[i,top_ind[0]]) - object_err[ind]/(alpha[top_ind[0]]/adjust[i,top_ind[0]])]
            max_y += [object_fl[ind]/(alpha[top_ind[0]]/adjust[i,top_ind[0]]) + object_err[ind]/(alpha[top_ind[0]]/adjust[i,top_ind[0]])]
            i+=1

        min_perc = np.nanpercentile(min_y, 5)
        max_perc = np.nanpercentile(max_y, 95)
        print (min_perc, max_perc)

        ax_best.set_ylim([min_perc*0.75, max_perc*1.25])
        ax_best.set_xlim([np.nanmin(object_wl)-0.05, np.nanmax(object_wl)+0.05])
        ax_best.annotate(name[top_ind[0]] + ' (' + num_to_alpha(spt_final[top_ind[0]]) + symbol_to_greek(symbol[top_ind[0]]) + ') '+r'$\chi^2_{\nu}$ = '+('%.2f' % chi2[top_ind[0]]), xy=(0.45, 0.875), xycoords='axes fraction')


    print (chi2[np.where(name == 'NAME CFBDSIR J214947.2-040308.9')])
    return name[top_ind], spt_final[top_ind], symbol[top_ind], chi2[top_ind]

def optimize_all_mp(start, object_fl, object_err, object_band, used_bands, comparison_flux, chunks, n_comparison):

    if start > (n_comparison - chunks):
        end = n_comparison
    else:
        end = start + chunks

    n_used = len(used_bands)
    
    this_i = np.arange(start, end)
    n = len(this_i)
    this_alpha = np.zeros(n) * np.nan
    this_n_dof = np.zeros(n) * np.nan
    this_chi2 = np.zeros(n) * np.nan
    this_adjust = np.zeros((n, n_used)) * np.nan

    for i in xrange(start, end):

        this_comparison_flux = np.concatenate(comparison_flux[np.ndarray.tolist(used_bands), :, i])

        if np.nansum(this_comparison_flux) != 0.:
            guess = np.nanmean(object_fl)/np.nanmean(this_comparison_flux)
            index = np.isfinite(object_fl) & np.isfinite(this_comparison_flux)

            n_dof = np.sum(index) - (2 + n_used)

            adjust = np.zeros(n_used)
            result = opt.minimize(fit_alpha2, [guess]+[1 for x in xrange(n_used)], args=(this_comparison_flux, object_fl, object_err, object_band, used_bands), bounds=[(1e-20, 1e20)]+[(0.5, 2.0) for x in xrange(n_used)])
            alpha = result['x'][0]
            chi2 = result['fun'] / n_dof

            for j in xrange(0, n_used):
                adjust[j] = result['x'][j+1]

        this_alpha[i-start] = alpha
        this_n_dof[i-start] = n_dof
        this_chi2[i-start] = chi2
        this_adjust[i-start,:] = adjust

    return this_i, this_alpha, this_n_dof, this_chi2, this_adjust

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

def num_to_alpha(num):
    if np.isnan(num):
        return 'nan'
    else:
        base = int(np.floor(num/10.))
        remainder = num - float(10*base)
        if base == 1:
            alpha = 'B'
        if base == 2:
            alpha = 'A'
        if base == 3:
            alpha = 'F'
        if base == 4:
            alpha = 'G'
        if base == 5:
            alpha = 'K'
        if base == 6:
            alpha = 'M'
        if base == 7:
            alpha = 'L'
        if base == 8:
            alpha = 'T'
        if base == 9:
            alpha = 'Y'

        if (num % 1) == 0:
            alpha = alpha + str(int(remainder))
        else:
            alpha = alpha + ('%.1f' % remainder)

        return alpha

def symbol_to_greek(sym):

    if sym == '*':
        greek = r'$\gamma$'
    elif sym == 'D':
        greek = r'$\beta$'
    elif sym == 's':
        greek = r'$\alpha$'
    else:
        greek = ''

    return greek


def spt_symbol(spt1, spt2, spt3, grav1, grav2, grav3):

    if spt1.decode("utf-8") == "NULL":
        if spt2.decode("utf-8") == "NULL":
            if spt3.decode("utf-8") == "NULL":
                spt_final = np.nan
            else:
                spt_final = alpha_to_num(spt3)
        else:
            spt_final = alpha_to_num(spt2)
    else:
        spt_final = alpha_to_num(spt1)

    if grav1.decode("utf-8") == 'gamma (opt)' or grav2.decode("utf-8") == "gamma (opt)" or grav3.decode("utf-8") == "gamma (opt)" or \
       grav1.decode("utf-8") == 'delta (opt)' or grav2.decode("utf-8") == "delta (opt)" or grav3.decode("utf-8") == "delta (opt)" or \
       grav1.decode("utf-8") == 'VL-G' or grav2.decode("utf-8") == "VL-G" or grav3.decode("utf-8") == "VL-G" or \
       grav1.decode("utf-8") == 'youngT' or grav2.decode("utf-8") == 'youngT' or grav3.decode("utf-8") == 'youngT':
        
        symbol = "*"


    elif grav1.decode("utf-8") == 'beta (opt)' or grav2.decode("utf-8") == "beta (opt)" or grav3.decode("utf-8") == "beta (opt)" or \
       grav1.decode("utf-8") == 'INT-G' or grav2.decode("utf-8") == "INT-G" or grav3.decode("utf-8") == "INT-G":
        
        symbol = "D"

    elif grav1.decode("utf-8") == 'alpha (opt)' or grav2.decode("utf-8") == "alpha (opt)" or grav3.decode("utf-8") == "alpha (opt)" or \
       grav1.decode("utf-8") == 'FLD-G' or grav2.decode("utf-8") == "FLD-G" or grav3.decode("utf-8") == "FLD-G":
        
        symbol = "s"

    else:
        symbol = "o"

    return spt_final, symbol

def fit_alpha2(p, model, obs, error, bands, used_bands):
    #p[0] contains the normal alpha parameter to be applied to all bands
    #p[1] Yband adjustment
    #p[2] Jband adjustment
    #p[3] Hband adjustment
    #p[4] K1band adjustment
    #p[5] K2band adjustment

    #make a copy so original isn't modified, not sure if this is necessary!
    this_obs = np.copy(obs)

    count = 1
    lens = np.zeros(len(used_bands), dtype=np.float64)
    for i in xrange(0, 5):
        #Try all bands
        index = np.where(bands == i)
        if len(index[0]) == 37:
            this_obs[index] *= p[count]
            lens[count-1] = float(np.sum(np.isfinite(this_obs[index] * model[index])))
            count+=1


    chi2 = np.nansum((((model*p[0])-this_obs)/error)**2.0)
    

    #we take the logarithm of the adjustment because we are comparing to the uncertainties in the Maire et al. 2014 GPI paper which are in magnitudes
    #Using mag_a - mag_b = -2.5*log10(flux_a/flux_b), if you set flux_b = 1 (i.e. no adjustment to the flux in that band) 
    #then the adjustment in magnitudes is mag_a - mag_b = -2.5xlog10(flux_a), which is your observed-expected (mag_a - mag_b) term on the top of the chi2 function.
    #this is then divided by the uncertainty on the spot ratio in each band, which is hard-coded below (0.05mag in Y, 0.03mag in J...)
    #it is then multiplied by the number of non-nans so it properly weights this constraint (otherwise it would only cause a very small change in the final chi2). 

    spot_errors = np.array([0.05, 0.03, 0.06, 0.07, 0.21])
    chi2 += np.nansum((((-2.5*np.log10(p[1:])) / spot_errors[used_bands])**2.0) * lens)

    return chi2


def optimize_one(library, data, object_band, ax_chi ):
    #         Y  J  H K1 K2
    #bands=  [0, 1, 2, 3, 4]
    

    object_wl = np.copy(data[:,0])
    object_fl = np.copy(data[:,1])
    object_err = np.copy(data[:,2])

    #        Y  J  H K1 K2
    BANDS = {'Y': 0, 'J': 1, 'H': 2, 'K1': 3, 'K2': 4}
    bands = [0, 1, 2, 3, 4]
    used_bands = np.unique(object_band)

    spt1 = library["ir_spt"]
    spt2 = library["optical_spt"]
    spt3 = library["simbad_spt"]
    grav1 = library["grav_a"]
    grav2 = library["grav_b"]
    grav3 = library["grav_c"]
    #name = library["adopted_name"]
    lum = library["simbad_lum"]

    comparison_flux = np.transpose(library['gpi_fl'], axes = (1, 0, 2))
    s = np.shape(comparison_flux)
    n_comparison = s[2]
    comparison_object = np.transpose(comparison_flux)[:,:,BANDS[object_band]]
    #print(np.shape(comparison_object))

    #print(np.shape(comparison_flux)[object_band,:,:])
    #print(np.shape(comparison_flux[BANDS[object_band],:,:]))
    #print(k)

    resid = np.array([])
    n_dof = np.array([])
    all_alpha = np.array([])

    symbol = np.zeros(n_comparison, dtype=str)
    spt_final = np.zeros(n_comparison, dtype = np.float64)
    for i in range(0, n_comparison):
        spt_final[i], symbol[i] = spt_symbol(spt1[i], spt2[i], spt3[i], grav1[i], grav2[i], grav3[i])


    for objct in comparison_object:
        if np.nansum(objct) == 0.:
            #If there is no flux for the comparison object
            resid = np.append(resid,np.nan)
            n_dof = np.append(n_dof, np.nan)

        else:
            guess = np.nanmean(object_fl)/np.nanmean(objct)
            index = np.isfinite(object_fl) & np.isfinite(objct)
            #If fitting only one band, we don't need to run opt.minimize, since we can simply differentiate the chi2 formula
            this_dof = np.sum(index) - 2 #n_dof = no. measurements - no. fitted parameters - 1 
            alpha = np.nansum((object_fl[index] * objct[index])/(object_err[index]**2.0)) / np.nansum((objct[index]**2.0)/(object_err[index]**2.0))
            chi2 = np.nansum(((object_fl[index] - (alpha*objct[index]))**2.0)/(object_err[index]**2.0)) / this_dof

        resid = np.append(resid,chi2)
        n_dof = np.append(n_dof, this_dof)
        #all_alpha +=  [alpha]
        all_alpha = np.append(all_alpha, alpha)
    #plt.plot(resid)
    #plt.show()




    for i in ["*", "D", "s", "o"]:
            index = np.where((symbol==i)) # & np.isfinite(spt) & np.isfinite(resid))
        
            if i == "*" :
                zorder = 10
                color = "yellow"
                alp = 1
                markersize = 9
                label = r'$\gamma$ / $\delta$ / VL-G'
                markeredgecolor = 'black'
            
            elif i == "D":
                zorder = 9
                color = "green"
                alp = 1
                markersize = 5
                label = r"$\beta$ / INT-G"
                markeredgecolor = 'black'
            
            elif i == "s":
                zorder = 8
                color = "red"
                alp = 1
                markersize = 5
                label = r"$\alpha$ / FLD-G"
                markeredgecolor = 'black'
            
            else:
                zorder = 7
                color = "blue"
                alp = .2
                markersize = 5
                label = ""
                markeredgecolor = 'none'

            plt.plot(spt_final[index],resid[index],i,color=color,zorder = zorder,alpha= alp,markersize = markersize,label=label,mec=markeredgecolor) 

    plt.show()


    #Now to plot the best fit
    #chi2[np.where(chi2 == 0.0)] = 1e10
    top_ind = np.argsort(resid)[:5] #Contains top 5








    return spt_final[top_ind], symbol[top_ind], resid[top_ind]
    #return name[top_ind], spt_final[top_ind], symbol[top_ind], chi2[top_ind]
@app.route('/sed_fit_plot/<data>')
@login_required
def sed_fit_plot(data = 'None'):

    import spex_fit
    from scipy.io import readsav

    text = data.replace('\r\n', '\n')
    text = text.replace('\r', '\n')
    text = text.encode('ascii')


    text = text.split('\n')


    #First remove any blank lines
    text = [x for x in text if x!= '']
    n_lines = len(text)
    
    #Now convert to numpy arrays (3, n_lines)
    data = np.zeros((n_lines, 3), dtype=np.float64)
    band = np.zeros((n_lines), dtype = int) - 1
    i = 0
    for line in text:
        sub_str = re.split('\t| +', line)
        for j in xrange(0,3):
            data[i,j] = float(sub_str[j])
        i+=1

    min_wl = [0.9444, 1.1108, 1.4904, 1.8818, 2.1034]
    max_wl = [1.1448, 1.3530, 1.8016, 2.1994, 2.4004]
    gpi_wl = np.zeros((5, 37))
    for i in xrange(0, 5):
        gpi_wl[i,:] = np.linspace(min_wl[i], max_wl[i], num=37)


    #Now to work out which bands have actually been sent to the routine.
    #Do a chi2 minimization between gpi wl array and the input array
    for i in xrange(0, n_lines, 37):
        test_wl = data[i:i+37,0]
        foo = np.zeros(5) * np.nan #variable to store sum of differences
        j = 0
        for wl in gpi_wl:
            foo[j] = np.nansum(np.abs(test_wl - wl))
            j+=1

        band[i:i+37] = np.argmin(foo)

    scale = np.nanmean(data[:,1])
    data[:,1] /= scale
    data[:,2] /= scale

    print band
    n_y = len(np.where(band == 0)[0])
    n_j = len(np.where(band == 1)[0])
    n_h = len(np.where(band == 2)[0])
    n_k1 = len(np.where(band == 3)[0])
    n_k2 = len(np.where(band == 4)[0])

    if n_y == 37:
        print 'doing Y'
    if n_j == 37:
        print 'doing J'
    if n_h == 37:
        print 'doing H'
    if n_k1 == 37:
        print 'doing K1'
    if n_k2 == 37:
        print 'doing K2'

    rc('font', **{'family':'serif','serif':'cm'})
    rc('text', usetex=True)

    fig = Figure(figsize=[12.0,10.0], dpi = 200)
    
    #Define the per-band chi2 plots here
    bottom = [0.95-0.18, 0.95-(0.18*2), 0.95-(0.18*3), 0.95-(0.18*4), 0.95-(0.18*5)]
    ax5 = fig.add_axes([0.06, bottom[4], 0.30, 0.18])
    ax1 = fig.add_axes([0.06, bottom[0], 0.30, 0.18])
    ax2 = fig.add_axes([0.06, bottom[1], 0.30, 0.18])
    ax3 = fig.add_axes([0.06, bottom[2], 0.30, 0.18])
    ax4 = fig.add_axes([0.06, bottom[3], 0.30, 0.18])

    ax7 = fig.add_axes([0.425, 0.5, 0.30, 0.225])
    ax6 = fig.add_axes([0.425, 0.725, 0.30, 0.225])

    ax9 = fig.add_axes([0.425, 0.05, 0.55, 0.20])
    ax8 = fig.add_axes([0.425, 0.25, 0.55, 0.20])

    ax1.get_xaxis().set_ticklabels([])
    ax2.get_xaxis().set_ticklabels([])
    ax3.get_xaxis().set_ticklabels([])
    ax4.get_xaxis().set_ticklabels([])
    ax6.get_xaxis().set_ticklabels([])
    ax8.get_xaxis().set_ticklabels([])

    ax1.annotate('ax1 - Y chi2', xy=(0.5, 0.8), xycoords='axes fraction')
    ax2.annotate('ax2 - J chi2', xy=(0.5, 0.8), xycoords='axes fraction')
    ax3.annotate('ax3 - H chi2', xy=(0.5, 0.8), xycoords='axes fraction')
    ax4.annotate('ax4 - K1 chi2', xy=(0.5, 0.8), xycoords='axes fraction')
    ax5.annotate('ax5 - K2 chi2', xy=(0.5, 0.8), xycoords='axes fraction')
    ax6.annotate('ax6 - chi2 free', xy=(0.5, 0.8), xycoords='axes fraction')
    ax7.annotate('ax7 - chi2 restricted', xy=(0.5, 0.8), xycoords='axes fraction')

    ax8.annotate('ax8 - best fit free', xy=(0.5, 0.8), xycoords='axes fraction')
    ax9.annotate('ax9 - best restricted', xy=(0.5, 0.8), xycoords='axes fraction')

    ax1.annotate('best fits list:', xy=(0.74,0.91), xycoords = 'figure fraction')

    ax1.annotate('object 1 (chi2 = 10)', xy=(0.75, 0.8875-0.000), xycoords='figure fraction', fontsize=10)
    ax1.annotate('object 2 (chi2 = 10)', xy=(0.75, 0.8875-0.025), xycoords='figure fraction', fontsize=10)
    ax1.annotate('object 3 (chi2 = 15)', xy=(0.75, 0.8875-0.050), xycoords='figure fraction', fontsize=10)
    ax1.annotate('object 4 (chi2 = 20)', xy=(0.75, 0.8875-0.075), xycoords='figure fraction', fontsize=10)
    ax1.annotate('object 5 (chi2 = 25)', xy=(0.75, 0.8875-0.100), xycoords='figure fraction', fontsize=10)

    ax1.set_ylabel(r'$\chi^2_{\nu}$ (Y band)')
    ax2.set_ylabel(r'$\chi^2_{\nu}$ (J band)')
    ax3.set_ylabel(r'$\chi^2_{\nu}$ (H band)')
    ax4.set_ylabel(r'$\chi^2_{\nu}$ (K1 band)')
    ax5.set_ylabel(r'$\chi^2_{\nu}$ (K2 band)')

    spt_val = [60, 70, 80, 90]
    for ax in (ax1, ax2, ax3, ax4, ax5, ax6, ax7):
        ax.set_xlim(60, 90)
        ax.set_xticks(spt_val)
        ax.set_xticklabels([])
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.set_yscale('log')

    spt_labels = ['M0', 'L0', 'T0', 'Y0']
    for ax in (ax5, ax7):
        ax.set_xlabel(r'Spectral Type')
        ax.set_xticklabels(spt_labels)
    

    ax6.set_ylabel(r'$\chi^2_{\nu}$')
    ax7.set_ylabel(r'$\chi^2_{\nu}$ (restricted fit)')


    ax9.set_xlabel('Wavelength ($\mu$m)')
    for ax in (ax8, ax9):
        ax.set_ylabel('Flux')
        ax.set_xlim(0.95, 2.45)

    lib_path = '/Users/derosa/Dropbox (Personal)/Jeffrey_Spectra/Interpolation/SpeX-IRTF-GPI-Resolution.sav'
    library = readsav(lib_path)


    #Do calculation for restricted fit
    if n_lines == 37:
        #This means we only have one so use optimize_one function. Will need an optional keyword argument to pass a ax to plot the best fit spectrum
        bar = 0

    else:
        best_name, best_spt, best_symbol, best_chi2 = spex_fit.optimize_all(library, data, band, ax7, ax9, restricted = True)
        ax1.annotate('Best fits (restricted fit):', xy=(0.74,0.69), xycoords = 'figure fraction')
        for i in xrange(0, 5):
            ax1.annotate(best_name[i]+ ' ('+spex_fit.num_to_alpha(best_spt[i]) + spex_fit.symbol_to_greek(best_symbol[i])+') '+r'$\chi^2_{\nu}$ = '+('%.2f' % best_chi2[i]), xy=(0.75, 0.6675-(0.025*i)), xycoords='figure fraction', fontsize=10)

    canvas = FigureCanvas(fig)
    output = StringIO.StringIO()
    canvas.print_png(output)
    response = make_response(output.getvalue())
    response.mimetype = 'image/png'
    #To stop the file having a ridiculous file name
    response.headers['Content-Disposition'] = 'attachment;filename=sed_fit.png'

    return response
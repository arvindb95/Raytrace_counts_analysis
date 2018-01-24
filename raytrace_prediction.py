import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
import glob
from scipy.integrate import simps
from scipy.integrate import quad
import astropy.units as u
from astropy.stats import sigma_clipped_stats
# Functions for Raytrace prediction analysis

def band(E, alpha=-1.0, beta=-2.5, E_peak=250.0*u.keV, norm=1.0):
    """
    Returns PHOTONS/s/cm^2/keV from band function with given parameters

    Required inputs:
    E = energy array in keV
    Optional inputs:
    alpha = first power law index of the band function, default = -1.0
    beta = second powerlaw index of the band function, default = -2.5
    E_peak = characteristic energy in keV, default = 250 (keV)
    norm = normalisation constant for the band function, default = 1   
    
    Returns:
    Total number of PHOTONS/s/cm^2/keV
    """
    #if (alpha - beta)*E_peak >= E:
    #    return norm*(E/100)**alpha*np.exp(-E/E_peak)
    #elif (alpha - beta)*E_peak < E:
    #    return norm*((alpha - beta)*E_peak/100)**(alpha - beta)*np.exp(beta - alpha)*(E/100)**beta

    E_below = (E < E_peak)
    band_out = np.zeros(len(E))
    band_out[E_below] = norm*(E/100.0)**alpha*np.exp(-E/E_peak)
    band_out[~E_below] = norm*((alpha - beta)*E_peak/100)**(alpha - beta)*np.exp(beta - alpha)*(E/100)**beta
    
    return band_out

def powerlaw(E, alpha=-1.0,E_peak=250.0*u.keV,norm=1.0):
    """
    Returns PHOTONS/s/cm^2/keV from powerlaw using given parameters

    Required inputs:
    E = energy array in keV
    Optional inputs:
    alpha = power law index , default = -1.0
    norm = normalisation constant for the powerlaw , default = 1
    
    Returns:
    Total number of PHOTONS/s/cm^2/keV
    """
    return norm*E**alpha*np.exp(-E/E_peak)

def model(E, alpha=-1.0, beta=-2.5, E_peak=250.0*u.keV, norm=1.0,typ="band"):
    """
    Returns PHOTONS/s/cm^2/keV from band or powerlaw based on input typ

    Required inputs:
    E = energy array (keV)
    Optional inputs:
    alpha = first power law index of the band function, default = -1.0
    beta = second powerlaw index of the band function, default = -2.5
    E_peak = characteristic energy in keV, default = 250  (keV)
    norm = normalisation constant for the band function, default = 1 
    typ = string to set the function to be used band or powerlaw, default = "band" 

    Returns:
    PHOTONS/s/cm^2/keV
    """
    if (typ=="powerlaw"):
        return powerlaw(E,alpha,norm)
    else:
        return band(E,alpha,beta,E_peak,norm)

def flux(E,alpha=-1.0, beta=-2.5, E_peak=250.0*u.keV, norm=1.0,typ="band"):
    """
    Returns flux in erg/cm^2/s/keV given a band function at a given energy

    Required inputs:
    E = energy array (keV)
    Optional inputs:
    alpha = first power law index of the band function, default = -1.0
    beta = second powerlaw index of the band function, default = -2.5
    E_peak = characteristic energy in keV, default = 250  (keV)
    norm = normalisation constant for the band function, default = 1 
    typ = string to set the function to be used band or powerlaw, default = "band" 

    Returns:
    Flux in erg/cm^2/s/keV
    """
    return model(E, alpha, beta, E_peak, norm,typ)*E*u.keV.to(u.erg)

def get_area_E(files):
    """
    Returns the total effective area array at evry energy in a given array of energies cm^2

    Required inputs:
    files = the output files from raytrace for different energies

    Returns:
    Effective area at each of the defined energy points in cm^2
    """
    area_E = np.zeros(len(files))
    for i in range(len(files)):
        t = Table.read(files[i],names=["quad","detx","dety","eff_a"],format='ascii')
        area_E[i] = np.sum(t['eff_a'])
    return area_E

def get_norm(flux,emin,emax,fluence,t_src,alpha=-1.0,beta=-2.5,E_peak=250.0*u.keV,typ="band"):
    """
    Returns the value of the normalisation constant (PHOTONS/cm^2/s/keV) calculated from the given spectral 
    type and parameters over a range of energies for which fluence is known

    Required inputs:
    flux = function that calculates flux from spectra ie. E*model
    emin = lower limit of energy for integration in keV
    emax = upper limit of energy for integration in keV
    fluence = fluence in ergs/cm^2
    t_src = time interval over which the fluence is observed in s
    Optional inputs:
    alpha = first power law index of the band function, default = -1.0
    beta = second powerlaw index of the band function, default = -2.5
    E_peak = characteristic energy in keV, default = 250  (keV)
    typ = string to set the function to be used band or powerlaw, default = "band" 
                        
    Returns norm in PHOTONS/cm^2/s/keV
    """
    unscaled_fluence = quad(flux,emin,emax,args=(alpha,beta,E_peak,1,typ))[0]*t_src # assuming norm is 1 we calculate the fluence
    norm = fluence/unscaled_fluence 
    
    return norm,unscaled_fluence

def get_tot_counts(E,eff_area,t_src,alpha=-1.0,beta=-2.5,E_peak=250.0*u.keV,norm=1.0,typ="band"):
    """
    Returns the total counts detected in all energies given the spectral parameters, array 
    of effective areas at the energies E and the timespan of source detection

    Required inputs:
    E = energy array in keV
    eff_area = effective area array at the above energies in cm^2
    t_src = time span of detection of source in s
    Optional inputs:
    alpha = first power law index of the band function, default = -1.0
    beta = second powerlaw index of the band function, default = -2.5
    E_peak = characteristic energy in keV, default = 250  (keV)
    norm = normalisation constant for the band function, default = 1 
    typ = string to set the function to be used band or powerlaw, default = "band" 
    
    Returns total counts (PHOTONS) 
    """
    counts_per_kev = eff_area*model(E,alpha,beta,E_peak,norm,typ)*t_src
    final_counts = simps(counts_per_kev,E)

    return final_counts

# New function: complete processing
# given model name, energies, areas, fermi fluence, model params, return czti photons

#----------------------------------------------------------------------

if __name__ == '__main__':
    files = glob.glob("*.out")
    E = np.arange(20.0,201.0,10.0) # Energy range over which the simulation data is taken

    # Initialising all required arrays

    eff_area = get_area_E(files)

    num_sim = 10000
    alpha = np.random.normal(-0.88,0.44,num_sim)    ##
    beta = np.repeat(-2.5,num_sim)           ##
    E_peak = np.random.normal(128.0,48.7,num_sim)     ## Parametrs obtained from Fermi GCN
    fluence = np.repeat(2.2e-7,num_sim)      ##
    t_src = 2.0                               ##
    typ = "powerlaw"
    emin = 10 # lower limit for calculating the norm from 
    emax = 1000 # upper limit for obtaining norm
    
    norm = np.zeros(num_sim)
    unscaled_fluence = np.zeros(num_sim)
    final_counts = np.zeros(num_sim)
#------------------------------------------------------------------------
# Calculating the total counts for each set of parameters
    for param in range(num_sim):
        norm[param],unscaled_fluence[param] = get_norm(flux,emin,emax,fluence[param],t_src,alpha[param],beta[param],E_peak[param],typ)
        
        final_counts[param] = get_tot_counts(E,eff_area,t_src,alpha[param],beta[param],E_peak[param],norm[param],typ)

#------------------------------------------------------------------------        
    # Plot a histogram of counts
    
    print "Total counts for each set of parameter values: ",final_counts
    print "--------------------------------------------------------------------"
    print "Average effective area: ",np.mean(eff_area)
    bins = np.linspace(np.nanmin(final_counts),np.nanmax(final_counts),50000)
    plt.hist(final_counts,bins,color="skyblue",ec="blue")
    plt.xlim(0,4000)
    #plt.title("Mean counts = {m:0.7f}".format(m=sigma_clipped_stats(final_counts)[0]) + ", Std. Dev = {s:0.2f}".format(s=sigma_clipped_stats(final_counts)[2]))
    
    
    # THe cutoff rates calculated 
    cutoff_counts = 115.18 + 112.17 + 101.17 + 98.28
    print "Total cutoff :",cutoff_counts 
    plt.axvline(cutoff_counts, linestyle = "dashed",color="k")
    plt.xlabel("Counts/sec")
    plt.text(cutoff_counts+10,400,"Cutoff (counts/sec)",rotation=90)

    print "The mean counts : ",sigma_clipped_stats(final_counts)[0]
    print "Median of counts : ",sigma_clipped_stats(final_counts)[1]
    print "Std. dev of counts : ",sigma_clipped_stats(final_counts)[2]

    final_counts_1 = final_counts[:(len(final_counts)/2)]
    final_counts_2 = final_counts[(len(final_counts)/2):]

    # No of sims with counts above cutoff
    no_sim_counts_above_1 = len(np.where(final_counts_1 > cutoff_counts)[0])
    percentage_1 = (no_sim_counts_above_1/float(num_sim/2.0)) *100.0
    no_sim_counts_above_2= len(np.where(final_counts_2 > cutoff_counts)[0])
    percentage_2 = (no_sim_counts_above_2/float(num_sim/2.0)) *100.0

    print "Percentage of simulations in which we got counts greater than cutoff",percentage_1
    print "Percentage of simulations in which we got counts greater than cutoff",percentage_2
    average_percentage = (percentage_1 + percentage_2)/2.0
    #plt.text(2500,30,r"{s:0.2f} $\%$".format(s=average_percentage))
    plt.show()


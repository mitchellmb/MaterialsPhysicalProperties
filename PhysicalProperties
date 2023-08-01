# Curie-Weiss (magnetism) and specific heat (thermal vibrations) fitting with scipy

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad


class CurieWeiss():
    ''' 
    A class to fit magnetic susceptibility data to a Curie-Weiss model, including a chi0 term.
    
    Attributes
    ----------
    x : temperature data, in K
    y : susceptibility, in emu/mol/Oe
    x_trim : (optional) trimmed temperature range for fitting
    y_trim : (optional) trimmed susceptibility concurrent with x_trim
    params : Curie constant (C), Curie Weiss temperature (theta), and chi0 offset
    fitparams : fit values of params after calling CurieWeiss.fit()
    fitcov : covariance matrix corresponding to fitparams
    
    Methods
    -------
    data_trim() : select a smaller temperature range to perform fitting over
    CurieWeissFormula() : formula used in fitting and plotting
    fit() : scipy curve_fit of x_trim, y_trim data with initial guesses stored in params
    plotter() : quick plot to check input data vs. fit parameters
    
    Example
    -------
    # Common use pipeline

    b = CurieWeiss(x, y, param_test)
    yvals = b.CurieWeissFormula(b.x, 1, -10, 0.0001)
    b.data_trim(0,500)
    b.fit()
    
    # initialize (x, y) data with guesses C = 1, theta = -10, and chi0 = 0.0001
    cw = HeatCapacityDebye(x, y, 1, -10, 0.0001) 
    
    # trim the data to a reasonable temperature range
    cw.data_trim(200, 300) 
    
    # fit, plot, and check fit values
    cw.fit()
    cw.plotter()
    cw.fitparams
    cw.fitcov
    '''
    
    def __init__(self, x_data, y_data, params): 
        self.x = x_data
        self.y = y_data
        self.x_trim = self.x
        self.y_trim = self.y
        self.params = params
        
        self.x_trim_label_low = int(np.min(self.x))
        self.x_trim_label_high = int(np.max(self.x))
        
    def data_trim(self, x_low, x_high):        
        trimmed_data = [(x,y) for (x,y) in zip(self.x, self.y) if (x >= x_low) & (x <= x_high)]
        self.x_trim = np.array([trimmed_data[i][0] for i in range(0, len(trimmed_data))])
        self.y_trim = np.array([trimmed_data[i][1] for i in range(0, len(trimmed_data))])
        
        self.x_trim_label_low = str(x_low)
        self.x_trim_label_high = str(x_high)
        
    def CurieWeissFormula(self, T, C, theta, chi):
        return C / (T - theta) + chi

    def fit(self):
            fitparams, fitcov = curve_fit(self.CurieWeissFormula, self.x_trim, self.y_trim, p0=self.params)
            self.fitcov = fitcov
            self.fitparams = fitparams
            
    def plotter(self, inverted = True, label = 'data', markersize = 5, fontsize = 15, linewidth = 2.5, 
                        color_data = 'black', color_fit = '#f28b1d'):
        ms = markersize
        fs = fontsize
        lw = linewidth
        
        if inverted:
            Y = 1./self.y
            Y_fit = 1./self.CurieWeissFormula(self.x_trim, *self.fitparams)
            y_label = r'$\frac{1}{\chi}$ (mol Oe emu$^{-1}$)'
            
        else:
            Y = self.y
            Y_fit = self.CurieWeissFormula(self.x_trim, *self.fitparams)
            y_label = r'$\chi$ (emu mol$^{-1}$ Oe$^{-1}$)'
        
        f = plt.figure()
        
        plt.plot(self.x, Y, 
                 color = color_data, linestyle = '', marker='o', markersize = ms, label = label)
        plt.plot(self.x_trim, Y_fit, 
                 color = color_fit, linestyle = '--', linewidth = lw, label = r'Curie-Weiss fit')
        
        plt.xlabel(r'$T$ (K)',fontsize = fs)
        plt.ylabel(y_label, fontsize = fs)
        plt.xlim(0,)
        plt.ylim(0,)
        
        plt.tick_params(axis='x',pad=10)
        plt.legend(loc='best', fontsize=fs, framealpha=0)
        fig = plt.gcf()
        fig.set_size_inches(6,6)
        
        ax = plt.gca()
        plt.text(1.01, 0.95, r'$\chi = C/(T-\Theta_{cw})+\chi_{o}$',
                 transform = ax.transAxes, fontsize=fs)
        plt.text(1.01,0.87,r'C: %4.4f emu K mol$^{-1}$' % self.fitparams[0], 
                 transform = ax.transAxes, fontsize=fs)
        plt.text(1.01,0.79,r'$\Theta_{cw}$: %4.4f K' % self.fitparams[1], 
                 transform = ax.transAxes, fontsize=fs)
        plt.text(1.01,0.71,r'$\chi_{o}$: %4.6f emu mol$^{-1}$' % self.fitparams[2], 
                 transform = ax.transAxes, fontsize=fs)
        plt.text(1.01,0.63,r'$\mu_{eff}$: %4.4f $\mu_B$' % np.sqrt(8.0*self.fitparams[0]),
                 transform = ax.transAxes, fontsize=fs)
        plt.text(1.01, 0.1, r'Temperature fit range:',
                 transform = ax.transAxes, fontsize=fs)
        plt.text(1.01, 0.02, f'{self.x_trim_label_low} K to {self.x_trim_label_high} K',
                 transform = ax.transAxes, fontsize=fs)
        
        return f
 

class HeatCapacityDebye():   
    ''' 
    A class to fit specific heat data to a Debye model. One Debye temperature or two 
    Debye temperatures can be provided.
    
    Attributes
    ----------
    x : temperature data, in K
    y : specific heat data, in J/K/mol
    x_trim : (optional) trimmed temperature range for fitting
    y_trim : (optional) trimmed specific heat concurrent with x_trim
    params : Debye temperature 1, scaling prefactor for Debye temperature 1, 
             optional : Debye temperature 2, scaling prefactor for Debye temperature 2
    fitparams : fit values of params after calling HeatCapacityDebye.fit()
    fitcov : covariance matrix corresponding to fitparams
    
    Methods
    -------
    data_trim() : select a smaller temperature range to perform fitting over
    debye_function() : Debye formula used in fitting and plotting
    fit() : scipy curve_fit of x_trim, y_trim data with initial guesses stored in params
    plotter() : quick plot to check input data vs. fit parameters
    
    Example
    -------
    # Common use pipeline
    
    # initialize (x, y) data with guesses [100, 10, 150, 10]
    cp = HeatCapacityDebye(x, y, [100, 10, 150, 10]) 
    
    # trim the data to a reasonable temperature range
    cp.data_trim(10, 200) 
    
    # fit, plot, and check fit values
    cp.fit()
    cp.plotter()
    cp.fitparams
    cp.fitcov
    '''
    
    def __init__(self, x_data, y_data, params):
        self.x = x_data
        self.y = y_data
        self.x_trim = self.x
        self.y_trim = self.y
        self.params = params
        
        self.x_trim_label_low = int(np.min(self.x))
        self.x_trim_label_high = int(np.max(self.x))
        
    def data_trim(self, x_low, x_high):        
        trimmed_data = [(x,y) for (x,y) in zip(self.x, self.y) if (x >= x_low) & (x <= x_high)]
        self.x_trim = np.array([trimmed_data[i][0] for i in range(0, len(trimmed_data))])
        self.y_trim = np.array([trimmed_data[i][1] for i in range(0, len(trimmed_data))])
        
        self.x_trim_label_low = str(x_low)
        self.x_trim_label_high = str(x_high)
        
    def debye_function(self, T, debye_temperature1, prefactor1, debye_temperature2 = None, prefactor2 = None):
        
        def debye_integrand(x):
            return np.power(x,4)*np.power(np.e,x)/np.power((np.power(np.e,x)-1),2)
        
        R = 8.3144598 #J/K/mol
        heat_capacity1 = [prefactor1*9.*R*(T[i]/debye_temperature1)**3.*
                          quad(debye_integrand, 0, debye_temperature1/T[i])[0] 
                          for i in range(0,len(T))]
        
        if debye_temperature2 != None:
            heat_capacity2 = [prefactor2*9.*R*(T[i]/debye_temperature2)**3.*
                              quad(debye_integrand, 0, debye_temperature2/T[i])[0] 
                              for i in range(0,len(T))]
            return np.array(heat_capacity1) + np.array(heat_capacity2)
        else:
            return np.array(heat_capacity1)
        
    def fit(self):
            fitparams, fitcov = curve_fit(self.debye_function, self.x_trim, self.y_trim, p0=self.params)
            self.fitcov = fitcov
            self.fitparams = fitparams
            
    def plotter(self, label = 'data', markersize = 5, fontsize = 15, linewidth = 2.5, 
                        color_data = 'black', color_fit = '#f28b1d'):
        ms = markersize
        fs = fontsize
        lw = linewidth
        
        Y = np.divide(self.y, self.x)
        Y_fit = np.divide(self.debye_function(self.x_trim, *self.fitparams), self.x)

        f = plt.figure()
        plt.plot(self.x, Y, 
                 color = color_data, linestyle = '', marker='o', markersize = ms, label = label)
        plt.plot(self.x_trim, Y_fit, 
                 color = color_fit, linestyle = '--', linewidth = lw, label = r'Debye model fit')
        
        plt.xlabel(r'$T$ (K)',fontsize = fs)
        plt.ylabel(r'$C_p/T$ (J K$^{-2}$ mol$^{-1}$)', fontsize = fs)
        plt.xlim(0,)
        plt.ylim(0,)
        
        plt.tick_params(axis='x',pad=10)
        plt.legend(loc='best', fontsize=fs, framealpha=0)
        fig = plt.gcf()
        fig.set_size_inches(6,6)
        
        ax = plt.gca()
        plt.text(1.01,0.95,r'Debye temperature: %4.2f K' % self.fitparams[0], 
                 transform = ax.transAxes, fontsize=fs)
        plt.text(1.01,0.87,r'Scaling prefactor: %4.2f' % self.fitparams[1], 
                 transform = ax.transAxes, fontsize=fs)
        plt.text(1.01, 0.1, r'Temperature fit range:',
                 transform = ax.transAxes, fontsize=fs)
        plt.text(1.01, 0.02, f'{self.x_trim_label_low} K to {self.x_trim_label_high} K',
                 transform = ax.transAxes, fontsize=fs)
        
        if self.fitparams[2]:
            plt.text(1.01,0.79,r'Debye temperature 2: %4.2f K' % self.fitparams[2], 
                     transform = ax.transAxes, fontsize=fs)
            plt.text(1.01,0.71,r'Scaling prefactor 2: %4.2f' % self.fitparams[3],
                     transform = ax.transAxes, fontsize=fs)
        
        return f
 

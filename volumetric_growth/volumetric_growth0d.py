import numpy as np
import matplotlib.pyplot as plt

#global constants
f_ff_max    = 0.3
f_f         = 150   
s_l50       = 0.06
F_ff50      = 1.35
f_l_slope   = 40
f_cc_max    = 0.1
c_f         = 75
s_t50       = 0.07
F_cc50      = 1.28
c_th_slope  = 60


def k_growth(F_g_cum, slope, F_50):
    return 1 / (1 + np.exp(slope * (F_g_cum - F_50)))


def incr_fiber_growth(s_l, dt, F_l_cum):
    if s_l >= 0:
        k_ff = k_growth(F_l_cum, f_l_slope, F_ff50)
        frac = f_ff_max * dt / (1 + np.exp(-f_f * (s_l - s_l50)))
        return k_ff * frac + 1
    
    else:
        frac = -f_ff_max * dt / (1 + np.exp(f_f * (s_l + s_l50)))
        return frac + 1
    
def incr_trans_growth(s_t, dt, F_c_cum):
    if s_t >= 0:
        k_cc = k_growth(F_c_cum, c_th_slope, F_cc50)
        frac = f_cc_max * dt / (1 + np.exp(-c_f * (s_t - s_t50)))
        return np.sqrt(k_cc * frac +1)
    else:
        frac = -f_cc_max * dt / (1 + np.exp(c_f * (s_t + s_t50)))
        return np.sqrt(frac + 1)
    


def plot_growth_limits():
    """
    Plot the growth limits k_ff, k_cc as 
    functions of cumulative growth.
    """
    F_g_cum = np.linspace(0.5,1.5,101)
    k_ff = k_growth(F_g_cum, f_l_slope, F_ff50)
    k_cc = k_growth(F_g_cum, c_th_slope, F_cc50)
    plt.plot(F_g_cum, k_ff,label = r"$k_{ff}$")
    plt.plot(F_g_cum, k_cc,label = r"$k_{cc}$")
    plt.title("Growth limiting factors $k_{ff},k_{cc}$")
    plt.legend()
    plt.xlabel("Cumulative growth")
    plt.ylabel("Growth limiting factor")

    plt.savefig("k_functions.pdf")
    plt.show()

def plot_incr_growth():
    """
    Plot incremental growth tensor components as
    functions of growth stimuli. 
    """
    stim = np.linspace(-0.2,0.2,101)
    F_g_cum = 1.0
    dt = 1.0

    incr_fiber_growth = np.vectorize(incr_fiber_growth)
    incr_trans_growth = np.vectorize(incr_trans_growth)

    F_g_i_ff = incr_fiber_growth(stim, dt, F_g_cum) 
    F_g_i_cc = incr_trans_growth(stim, dt, F_g_cum) 

    plt.plot(stim, F_g_i_ff, label = r"$F_{g,i,ff}$")
    plt.plot(stim, F_g_i_cc, label = r"$F_{g,i,cc}$")
    plt.title("Incremental growth tensor components")
    plt.legend()
    plt.xlabel("Growth stimulus ($s_l,s_t$)")
    plt.ylabel("Incremental growth [1/day]")
    plt.savefig("incr_growth_tensor.pdf")    

    plt.show()


def grow_unit_cube(lmbda, T, N, E_f_set = 0, E_c_set = 0):
    """"
    Plot the growth of a unit cube over time, resulting from a 
    constant stretch lmbda
    """
    #time measured in days, N steps
    time = np.linspace(0, T, N+1)
    dt_growth = T/N

    #cumulative growth tensor components:
    F_g_f_tot = np.ones_like(time)
    F_g_c_tot = np.ones_like(time)

    #initial elastic strains
    E_f = 0.5 * (lmbda**2 - 1)
    lmbda_c = (1.0 / np.sqrt(lmbda))
    E_c = 0.5 * (lmbda_c**2 - 1)

    for i in range(N):
        print("Step ", i)
        #growth stimuli:
        sl = E_f - E_f_set
        st = E_c - E_c_set
        print(st) 
        
        #incremental and cumulative growth tensors:
        F_g_i_f = incr_fiber_growth(sl, dt_growth, F_g_f_tot[i])
        F_g_i_c = incr_trans_growth(st, dt_growth, F_g_c_tot[i])
        print(F_g_i_f, F_g_i_c)

        F_g_f_tot[i + 1] = F_g_f_tot[i] * F_g_i_f
        F_g_c_tot[i + 1] = F_g_c_tot[i] * F_g_i_c

        #update elastic strains E_f and E_c:
        lmbda_e_f = lmbda / F_g_f_tot[i+1]
        lmbda_e_c = np.sqrt(1 / lmbda_e_f) 

        E_f = 0.5 * (lmbda_e_f**2 - 1)
        E_c = 0.5 * (lmbda_e_c**2 - 1)
    

    plt.plot(time, F_g_f_tot, label = 'Fiber')
    plt.plot(time, F_g_c_tot, label = 'Cross-fiber')
    plt.plot(time, np.ones_like(time) * lmbda, ':')
    plt.plot(time, np.ones_like(time) * 1/np.sqrt(lmbda), ':')
    
    plt.title(f'Uniaxial stretch, $\lambda$ = {lmbda}')
    plt.xlabel('Time [days]')
    plt.ylabel('Cumulative growth tensor components')
    plt.legend()
    plt.savefig('cumulative_growth.pdf')
    plt.show()


    
grow_unit_cube(lmbda = 1.1 , T = 100, N = 5000, E_f_set = 0, E_c_set = 0)



    

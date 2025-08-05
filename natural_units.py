# Written 2025-07-14 by Jack Beda (jack.beda.ca).
###############################################################

"""
See the function units_help() for most explanations. Our simulation is based off of a particular set of base units described
there, and all quantities are expressed in these derived units unless said otherwise.

For example, we call the unit of energy in our system κ, which is 1 amu·μm²/μs². The unit of temperature is called 
θ and is equal to 1 κ / k_B. For example, if you see in the code something labeled temperature, this will be stored in units of 
θ. Occasionally, when we desire things in other units, we will specify precicely. For example, temperature_mK will contain a 
quantity measured in milli Kelvin. 

"""

hbar = 0.063507799295889 # Reduced Plank's constant(in units of amu·μm²·μs⁻¹)
kB = 1. # Boltzmann constant (in units of kB)
k = 1.38935378902e5 # Coulomb constant (in units of e²·μm³·μs⁻²·amu⁻¹)
epsilon_0 = 5.7276607423e-7 # Vacuum permitivity (in units of amu·μs²·μm⁻³·e⁻²)
electron_charge = 1. # Electron charge (in units of e)

def theta_to_mK(theta):
    return theta * 0.120272422607

def mK_to_theta(mK):
    return mK / 0.120272422607

def kappa_to_neV(kappa):
    return kappa * 10.3642537043

def neV_to_kappa(neV):
    return neV / 10.3642537043

def Newton_to_force_units(Newton):
    return Newton / 1.66054e-21

def force_units_to_Newton(force_units):
    return force_units * 1.66054e-21

def units_help():
    return """
        Units for simulation:
        ---------------------
        Base units (fixed by assumption):
        - Length: 1 micrometer (μm)
        - Time: 1 microsecond (μs)
        - Mass: 1 atomic mass unit (amu)
        - Charge: 1 elementary charge (e)
        - Boltzmann constant: kB = 1 kB

        Derived units:
        - Energy (κ):               κ = amu·μm²·μs⁻² ≈ 1.66054 × 10⁻²⁷ J ≈ 10.362537043 neV
        - Temperature (θ):          θ = κ·kB⁻¹ ≈ 0.12027 mK
        - Frequency:                MHz (μs⁻¹)
        - Force:                    amu·μm·μs⁻² ≈ 1.66054 × 10⁻²¹ N
        
        Physical constants:
        - Vacuum permittivity:      ε₀ ≈ 5.7276607423 × 10⁻⁷ amu·μs²·μm⁻³·e⁻²
        - Coulomb constant:         k = 1 / (4π ε₀) ≈ 1.38935378902e5 × 10⁵ e²·μm³·μs⁻²·amu⁻¹
        - Reduced Plank's constant: hbar = 0.063507799295889 amu·μm²·μs⁻¹

        Notes:
        - All physical quantities are expressed in these natural units.
        """

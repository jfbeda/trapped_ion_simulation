# Written 2025-07-14

hbar = 0.063507799295889 # (amu)(μm2)(μs−1)
kB = 1. # 1 in natural units
k = 1.38935378902e5 # Coulomb constant in units of e²·μm³·μs⁻²·amu⁻¹
epsilon_0 = 5.7276607423e-7 # amu·μs² / (μm³·e²)
electron_charge = 1. # 1 in natural units

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
        - Boltzmann constant: k_B = 1

        Derived units:
        - Energy (κ):     1 amu·μm²/μs² ≈ 1.66054 × 10⁻²⁷ J ≈ 10.36 neV
        - Temperature (θ): κ / k_B = 0.00012027 K ≈ 0.12027 mK
        - Frequency:       1 MHz (μs⁻¹)
        - Force:           1 amu·μm/μs² ≈ 1.66054 × 10⁻²¹ N
        - Vacuum permittivity:
          ε₀ ≈ 5.7276607423 × 10⁻⁷ amu·μs² / (μm³·e²)
        - Coulomb constant:
          k ≈ 1 / (4π ε₀) ≈ 1.389 × 10⁵ e²·μm³ / (amu·μs²)

        Notes:
        - All physical quantities are expressed in these natural units.
        - Temperatures, energies, and forces are dimensionless in code but represent physical values via these units.
        """

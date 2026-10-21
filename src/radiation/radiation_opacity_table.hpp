#ifndef RADIATION_RADIATION_OPACITY_TABLE_HPP_
#define RADIATION_RADIATION_OPACITY_TABLE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_opacity_table.hpp
//! \brief data strcuture to store opacity table, so that device can access them

#include <math.h>

#include "athena.hpp"

//----------------------------------------------------------------------------------------

// struct OpacityTable {
//   // Size of table
//   int n_rho;
//   int n_temp;

//   // Tabulated opacity? store in a Kokkos::View
//   Kokkos::View<Real*> rho_grid;
//   Kokkos::View<Real*> temp_grid;
//   Kokkos::View<Real**> kappa_ross;
//   Kokkos::View<Real**> kappa_planck;
// };
  
struct OpacityData{

  // Size of table
  int n_rho;
  int n_temp;

  // Tabulated opacity? store in a Kokkos::View
  Kokkos::View<Real*> rho_grid;
  Kokkos::View<Real*> temp_grid;
  Kokkos::View<Real**> kappa_ross; // NOTE: stored as log10(kappa_ross)
  Kokkos::View<Real**> kappa_planck; // NOTE: stored as log10(kappa_planck)

  // Precomputed log-space grid parameters (uniform spacing)
  Real log_tmin;
  Real log_rhomin;
  Real inv_dlogT;
  Real inv_dlogrho;

  // OpacityTable table_host; //this is the table live on host
  // Kokkos::View<OpacityTable*> table_device; //this is the table live on device
  
  // Singleton instance
  static OpacityData& GetInstance() {

    //write it as a pointer instace of actual instance, so it will not be destructed
    //this seem to avoid the error of some array is destructed after Kokkos::finalize()
    static OpacityData *instance = new OpacityData();
    return *instance;
  }

private:
  OpacityData() {}  // Private constructor
};


// Interpolation function, free-free opacity, input cgs output cgs 
KOKKOS_INLINE_FUNCTION
Real kappa_ff_nu(Real nu, Real temp, Real rho){

  Real h_planck = 6.626196e-27 ;
  Real evtohz = 2.41838e14;
  Real rho_cgs = rho;
  Real temp_cgs =  temp;
  Real m_p = m_p = 1.6726e-24;
  Real k_B = 1.3807e-16;
  
  Real  gff = 1.0;
  Real  z = 1.0;

  Real  he_adbund = 0.04;
  Real  nh = rho_cgs/m_p/(1.0 + 4.0*he_adbund);
  Real  nhe = nh*he_adbund;
  Real  ne = nh + 2.0*nhe;
  Real  n_rho = rho_cgs/m_p/0.62;

  Real  e_ff = 3.7e8 * pow(temp_cgs, -0.5) * pow(z, 2) * pow(n_rho, 2) * pow(nu, -3) * (1.0 - exp(-h_planck*nu/k_B/temp_cgs)) * gff;

  return e_ff/rho_cgs;
}

// Interpolation function, planck mean free free absorption
KOKKOS_INLINE_FUNCTION
Real kappa_ff_planck(Real temp, Real rho){
  Real rho_cgs = rho;
  Real temp_cgs =  temp;
  Real kappa_cgs = 2.86e-5*(rho_cgs/1.0e-8)*pow(temp_cgs/1.0e6, -3.5);

  return kappa_cgs;
}

// Interpolation function, rosseland mean free free absorption
KOKKOS_INLINE_FUNCTION
Real kappa_ff_ross(Real temp, Real rho){
  Real rho_cgs = rho;
  Real temp_cgs =  temp;
  Real kappa_cgs = 7.73e-7*(rho_cgs/1.0e-8)*pow(temp_cgs/1.0e6, -3.5);

  return kappa_cgs;
}

//Klein-Nishina suppression of the electron-scattering opacity. Fit by Paczynski (1983) to the Compton-scattering opacities of  Buchler & Yueh (1976)
KOKKOS_INLINE_FUNCTION
Real kappa_es_kn(Real k_s, Real temp_cgs) {
  return k_s / (1.0 + pow(temp_cgs/4.5e8, 0.86));
}

// Interpolation function, directly read views instead of OpacityData structure
KOKKOS_INLINE_FUNCTION
void InterpolateKappa(int n_rho, int n_temp,
                      const Kokkos::View<Real*>& rho_grid,
                      const Kokkos::View<Real*>& temp_grid,
                      const Kokkos::View<Real**>& log_kappa_ross_tab,
                      const Kokkos::View<Real**>& log_kappa_planck_tab,
                      Real log_tmin, Real log_rhomin,
                      Real inv_dlogT, Real inv_dlogrho,
                      Real rho, Real tgas, Real k_s,
                      Real &kappa_ross, Real &kappa_planck){

//STEP 1: fractional-index positions in log space
  Real logT   = log10(fmax(tgas,1.0e-10));
  Real logrho = log10(fmax(rho,1.0e-32));
  Real log_tmax   = log_tmin   + (Real)(n_temp - 1) / inv_dlogT;
  Real log_rhomax = log_rhomin + (Real)(n_rho  - 1) / inv_dlogrho;

  Real posT   = (logT   - log_tmin  ) * inv_dlogT;
  Real posrho = (logrho - log_rhomin) * inv_dlogrho;

  //STEP 2: base/upper indices and in-cell fractions, with edge clamping
  // (clamp below: use first grid point; clamp above: use last grid point)
  int nt1, nt2;
  Real fT;
  if (posT <= 0.0) {
    nt1 = 0;           nt2 = 0;           fT = 0.0;
  } else if (posT >= (Real)(n_temp - 1)) {
    nt1 = n_temp - 1;  nt2 = n_temp - 1;  fT = 0.0;
  } else {
    nt1 = (int)floor(posT);  nt2 = nt1 + 1;  fT = posT - (Real)nt1;
  }

  int nrho1, nrho2;
  Real frho;
  if (posrho <= 0.0) {
    nrho1 = 0;          nrho2 = 0;          frho = 0.0;
  } else if (posrho >= (Real)(n_rho - 1)) {
    nrho1 = n_rho - 1;  nrho2 = n_rho - 1;  frho = 0.0;
  } else {
    nrho1 = (int)floor(posrho);  nrho2 = nrho1 + 1;  frho = posrho - (Real)nrho1;
  }

  //STEP 3: read 4 corners of log(kappa) tables
  Real lkr_11 = log_kappa_ross_tab(nt1,nrho1);
  Real lkr_12 = log_kappa_ross_tab(nt1,nrho2);
  Real lkr_21 = log_kappa_ross_tab(nt2,nrho1);
  Real lkr_22 = log_kappa_ross_tab(nt2,nrho2);
  Real lkp_11 = log_kappa_planck_tab(nt1,nrho1);
  Real lkp_12 = log_kappa_planck_tab(nt1,nrho2);
  Real lkp_21 = log_kappa_planck_tab(nt2,nrho1);
  Real lkp_22 = log_kappa_planck_tab(nt2,nrho2);

  //STEP 4: extrapolation above Tmax.
  //  Planck: pure absorption, Kramers T^-3.5.  Verified against this table --
  //    d log kappa_P / d log T at the top is -3.509 at every one of the 100
  //    density columns, independent of how many rows the fit uses.
  //  Rosseland: at Tmax the table is PURE SCATTERING 
  //   so continue it by the KN RATIO.
  //  NOTE: the pure-scattering claim is a property of THIS table. If the table
  //    is replaced, re-check that its top row is density-independent before
  //    trusting the ratio form.
  //  NOTE 2: the KN ratio applied below no longer reaches the solver.
  //  Scattering is overwritten with the Thomson value
  //  because in the TDE problem Trad << max(Tgas)
  //  but KN in TOPS' Rosseland assumes Tgas~Trad (diffusion limit)
  //  The scaling is kept because it correctly describes the table though.
  //  If Trad ever reaches KN temperatures, change radiation_opacities.hpp
  if (logT > log_tmax) {
    Real logscale = -3.5 * (logT - log_tmax);
    Real log_knr  = log10(kappa_es_kn(k_s, tgas)
                        / kappa_es_kn(k_s, pow(10.0, log_tmax)));

    lkr_11 += log_knr;
    lkr_12 += log_knr;
    lkr_21 += log_knr;
    lkr_22 += log_knr;

    lkp_11 += logscale;
    lkp_12 += logscale;
    lkp_21 += logscale;
    lkp_22 += logscale;
  }

  //UPDATE: STEP 4b REMOVED.  There is no longer any rho < rho_min extrapolation.
  //  UserOpacityFunction() in radiation_opacities.hpp now floors every cell
  //  below the table's rho_min, at all temperatures, so nothing reaching this
  //  function can have logrho < log_rhomin and the branch was unreachable.
  //  Do not restore this without checking at which temperature in a given table  //  the slope is actually Kramers. For the table tested it was above ~4e6K.
  //  If table is to be extended, extrapolate with edge slope at each T row.
  
  /*//STEP 4b: Kramers rho-linear extrapolation for rho < rho_min and T >= T_ion
  //   Below the table's rho_min we extrapolate the absorption
  //  piece linearly in rho; scattering (k_s) stays fixed.
  // Gated by T >= 1e4 K where Kramers applies,
  // if T<1e4K fall through with clamp.
  Real log_tion_cgs = 4.0;
  if (logrho < log_rhomin && logT >= log_tion_cgs) {
    Real log_rho_offset = logrho - log_rhomin; 
    Real rho_scale      = pow(10.0, log_rho_offset);

    // Rosseland: scale absorption piece only, k_s fixed
    Real kr_11 = pow(10.0, lkr_11);
    Real kr_12 = pow(10.0, lkr_12);
    Real kr_21 = pow(10.0, lkr_21);
    Real kr_22 = pow(10.0, lkr_22);
    Real ka_11 = fmax(kr_11 - k_s, 0.0);
    Real ka_12 = fmax(kr_12 - k_s, 0.0);
    Real ka_21 = fmax(kr_21 - k_s, 0.0);
    Real ka_22 = fmax(kr_22 - k_s, 0.0);
    lkr_11 = log10(ka_11 * rho_scale + k_s);
    lkr_12 = log10(ka_12 * rho_scale + k_s);
    lkr_21 = log10(ka_21 * rho_scale + k_s);
    lkr_22 = log10(ka_22 * rho_scale + k_s);

    // Planck
    lkp_11 += log_rho_offset;
    lkp_12 += log_rho_offset;
    lkp_21 += log_rho_offset;
    lkp_22 += log_rho_offset;
  }*/

  //STEP 5: log-space bilinear
  Real w11 = (1.0 - fT) * (1.0 - frho);
  Real w12 = (1.0 - fT) * frho;
  Real w21 = fT         * (1.0 - frho);
  Real w22 = fT         * frho;

  Real log_kappa_ross   = w11*lkr_11 + w12*lkr_12 + w21*lkr_21 + w22*lkr_22;
  Real log_kappa_planck = w11*lkp_11 + w12*lkp_12 + w21*lkp_21 + w22*lkp_22;

  kappa_ross   = pow(10.0, log_kappa_ross);
  kappa_planck = pow(10.0, log_kappa_planck);

}

#endif

#ifndef RADIATION_RADIATION_OPACITIES_HPP_
#define RADIATION_RADIATION_OPACITIES_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_opacities.hpp
//! \brief implements functions for computing opacities

#include <math.h>

#include "athena.hpp"

#include "radiation_opacity_table.hpp"
//----------------------------------------------------------------------------------------
//! \fn void OpacityFunction
//! \brief sets sigma_a, sigma_s, sigma_p in the comoving frame

KOKKOS_INLINE_FUNCTION
void OpacityFunction(// density and density scale
                     const Real dens, const Real density_scale,
                     // temperature and temperature scale
                     const Real temp, const Real temperature_scale,
                     // length scale, adiabatic index minus one, mean molecular weight
                     const Real length_scale, const Real gm1, const Real mu,
                     // power law opacities
                     const bool pow_opacity,
                     const Real rosseland_coef, const Real planck_minus_rosseland_coef,
                     // spatially and temporally constant opacities
                     const Real k_a, const Real k_s, const Real k_p,
                     // output sigma
                     Real& sigma_a, Real& sigma_s, Real& sigma_p) {
  if (pow_opacity) {  // power law opacity (accounting for diff b/w Ross & Planck)
    Real power_law = (dens*density_scale)*pow(gm1*mu/(temp*temperature_scale), 3.5);
    Real k_a_r = rosseland_coef * power_law;
    Real k_a_p = planck_minus_rosseland_coef * power_law;
    sigma_a = dens*k_a_r*density_scale*length_scale;
    sigma_p = dens*k_a_p*density_scale*length_scale;
    sigma_s = dens*k_s  *density_scale*length_scale;
  } else {  // spatially and temporally constant opacity
    sigma_a = dens*k_a*density_scale*length_scale;
    sigma_p = dens*k_p*density_scale*length_scale;
    sigma_s = dens*k_s*density_scale*length_scale;
  }
  return;
}

//! \fn void UserOpacityFunction
//! \brief sets sigma_a, sigma_s, sigma_p in the comoving frame
KOKKOS_INLINE_FUNCTION
void UserOpacityFunction(// density and density scale
                     const Real dens, const Real density_scale,
                     // temperature and temperature scale
                     const Real temp, const Real temperature_scale,
                     // length scale, adiabatic index minus one, mean molecular weight
                     const Real length_scale, const Real gm1, const Real mu,
                     // power law opacities
                     const bool pow_opacity,
                     const Real rosseland_coef, const Real planck_minus_rosseland_coef,
                     // spatially and temporally constant opacities
                     const Real k_a, const Real k_s, const Real k_p,
                     // output sigma
                     Real& sigma_a, Real& sigma_s, Real& sigma_p,
                     // pass the opacity table variables
                     const int n_rho, const int n_temp,
                     const Kokkos::View<Real*>& rho_grid,
                     const Kokkos::View<Real*>& temp_grid,
                     const Kokkos::View<Real**>& kappa_ross,
                     const Kokkos::View<Real**>& kappa_planck,
                     const Real log_tmin, const Real log_rhomin,
                     const Real inv_dlogT, const Real inv_dlogrho,
                     const Real low_rho_threshold_cgs,
                     const Real low_temp_threshold_cgs){
  

  //allocate individual views for the Opacity Data structure
  //const OpacityTable tab = opacity_dev(0);
  Real kappa_ross_interp, kappa_planck_interp;

  Real dens_cgs = dens * density_scale;
  Real temp_cgs = temp * temperature_scale;
  Real temp_recomb_cgs = 3.0e3; //below this H is neutral, no scattering

  //Everything below the table's lowest density is grid filler, not modelled
  // gas, and is made transparent. 
  //Apply opacity floors for background gas (e.g. 10d_amb)
  Real rho_floor_cgs = fmax(low_rho_threshold_cgs, pow(10.0, log_rhomin));
  (void)low_temp_threshold_cgs;  // retained in the signature, no longer used

  if (dens_cgs < rho_floor_cgs){
    Real k_floor = 1.0e-6;
    
    sigma_a = dens*k_floor*density_scale*length_scale;
    sigma_p = 0.0;
    sigma_s = 0.0;
    return;
  }
  
  InterpolateKappa(n_rho, n_temp, rho_grid, temp_grid, kappa_ross, kappa_planck,
		   log_tmin, log_rhomin, inv_dlogT, inv_dlogrho,
                   dens_cgs, temp_cgs, k_s, kappa_ross_interp, kappa_planck_interp);
  
  //if (dens_cgs>1.0e-14){
  //    printf("current density: %g, temperature: %g, kappa_ross: %g, kappa_planck: %g\n", dens_cgs, temp_cgs, kappa_ross_interp, kappa_planck_interp);
  //}
  //OPAL/TOPs Rosseland mean opacity is the total absorption, including scatter
  Real kappa_ross_cgs = 0.0;
  Real kappa_sct_cgs = 0.0;
  Real temp_ion_cgs = 1.0e4;

  //k_s is in c.g.s
  if (kappa_ross_interp >= k_s){
    if (temp_cgs > temp_recomb_cgs){
      kappa_ross_cgs = fmax(kappa_ross_interp - k_s, 0.0);
      kappa_sct_cgs = k_s;
    }else{//cold dense gas past H recombination, pure absorption
      kappa_ross_cgs = kappa_ross_interp;
      kappa_sct_cgs = 0.0;
    }
  }else{ //if tabulated rosseland mean < scatter  
    if (temp_cgs < temp_ion_cgs){//below ionization temperature, no scatter opacity
      kappa_ross_cgs = kappa_ross_interp;
      kappa_sct_cgs = 0.0;
    }else{
      // Klein-Nishina in TOPS Rosseland mean applies in the diffusion limit Tgas~Trad
      // Our TDE problem has T_rad<<max(T_gas) so it doesnt apply here.
      // If a run ever puts T_gas > 1e8 K at tau_eff >> 1, the tabulated KN
      // value is the correct one and this must be gated on the local
      // radiation temperature instead of overridden unconditionally.
      kappa_sct_cgs = fmax(kappa_ross_interp, k_s);
      kappa_ross_cgs = 0.0;
    }
  }
  
  //in code, kappa_planck is difference between kappa_planck and kappa_ross
  Real kappa_planck_cgs = kappa_planck_interp - kappa_ross_cgs;
  
  //assign to cell
  sigma_a = dens*kappa_ross_cgs*density_scale*length_scale;
  sigma_p = dens*kappa_planck_cgs*density_scale*length_scale;
  sigma_s = dens*kappa_sct_cgs*density_scale*length_scale;
  
  return;
  
}

#endif // RADIATION_RADIATION_OPACITIES_HPP_

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
                     Real& sigma_a, Real& sigma_s, Real& sigma_p){
  
  const auto& tab = OpacityData::GetInstance().table;
  Real kappa_ross_tab, kappa_planck_tab;

  // Real rho_test = 1.0e-10;
  // Real temp_test = 3.4e5;
  // InterpolateKappa(tab, rho_test, temp_test, kappa_ross_tab, kappa_planck_tab);

  // printf("testing for density %g and temperature %g: kappa_ross=%g, kappa_planck=%g\n",
  // 	 rho_test, temp_test, kappa_ross_tab, kappa_planck_tab);

  Real dens_cgs = dens * density_scale;
  Real temp_cgs = temp * temperature_scale;
  InterpolateKappa(tab, dens_cgs, temp_cgs, kappa_ross_tab, kappa_planck_tab);
  //printf("current density: %g, temperature: %g, kappa_ross: %g, kappa_planck: %g\n", dens_cgs, temp_cgs, kappa_ross_tab, kappa_planck_tab);

  //OPAL/TOPs Rosseland mean opacity is the total absorption, including scatter
  Real kappa_ross_cgs = 0.0;
  Real kappa_sct_cgs = 0.0;
  Real temp_ion_cgs = 1.0e4;
  Real temp_ion = temp_ion_cgs/temperature_scale;

  //k_s is in c.g.s
  if (kappa_ross_tab > k_s){
    kappa_ross_cgs = kappa_ross_tab - k_s;
    kappa_sct_cgs = k_s;
  }else{ //if tabulated rosseland mean < scatter  
    if (temp_cgs < temp_ion_cgs){//below ionization temperature, no scatter opacity
      kappa_ross_cgs = kappa_ross_tab;
      kappa_sct_cgs = 0.0;
    }else{
      kappa_sct_cgs = kappa_ross_tab;
      kappa_ross_cgs = 0.0;
    }
  }
  
  Real kappa_planck_cgs = kappa_planck_tab;
  
  //assign to cell
  sigma_a = dens*kappa_ross_cgs*density_scale*length_scale;
  sigma_p = dens*kappa_planck_cgs*density_scale*length_scale;
  sigma_s = dens*kappa_sct_cgs*density_scale*length_scale;
  
  return;
  
}

#endif // RADIATION_RADIATION_OPACITIES_HPP_

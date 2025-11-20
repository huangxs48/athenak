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

struct OpacityTable {
  // Size of table
  int n_rho;
  int n_temp;

  // Tabulated opacity? store in a Kokkos::View
  Kokkos::View<Real*> rho_grid;
  Kokkos::View<Real*> temp_grid;
  Kokkos::View<Real**> kappa_ross;
  Kokkos::View<Real**> kappa_planck;
};
  
struct OpacityData{
  OpacityTable table;
  
  // Singleton instance
  static OpacityData& GetInstance() {
    //static OpacityData instance;
    //return instance;
    
    //chatGPT suggested rewrite as a pointer, so it will not be destructed
    //this seem to avoid the error of some array is destructed after Kokkos::finalize()
    static OpacityData *instance = new OpacityData();
    return *instance;
  }

private:
  OpacityData() {}  // Private constructor
};

// Interpolation function, read in an opacity table instance, 
KOKKOS_INLINE_FUNCTION
void InterpolateKappa(const OpacityTable& tab, Real rho, Real tgas, Real &kappa_ross, Real &kappa_planck){
  //std::cout<<"InterpolateKappa called!"<<std::endl;

  // // quick check the readings
  // for (int i=90; i<100; ++i){
  //   std::cout<<"tab.kappa_ross(30, "<<i<<") = "<< tab.kappa_ross(30,i)<<", tab.kappa_planck(30, "<<i<<")="<< tab.kappa_planck(30,i)<<std::endl;
  // }

  int n_temp = tab.n_temp;
  int n_rho = tab.n_rho;

  //STEP1: find index of temperature and density range
  //index searching segment in rho grid
  int nrho1 = 0;
  int nrho2 = 0;

  while((nrho2 < n_rho-1) && (rho > tab.rho_grid(nrho2)) ){
    nrho1 = nrho2;
    nrho2++;
  }
  //if hits the end of table, set two index equal
  if((rho > tab.rho_grid(nrho2)) && (nrho2==n_rho-1)){
    nrho1=nrho2;
  }


  //index searching segments in temperature grid
  int nt1 = 0;
  int nt2 = 0;
  while((tgas > tab.temp_grid(nt2)) && (nt2 < n_temp-1)){
    nt1 = nt2;
    nt2++;
  }
  //if hits the end of table, set two index equal
  if(nt2==n_temp-1 && (tgas > tab.temp_grid(nt2))){
    nt1=nt2;
  }

  std::cout<<"n_temp:"<<n_temp<<" n_rho:"<<n_rho<<std::endl;
  std::cout<<"input rho:"<<rho<<" tgas:"<<tgas<<std::endl;
  std::cout<<"nrho1:"<<nrho1<<" nrho2:"<<nrho2<<" nt1:"<<nt1<<" nt2:"<<nt2<<std::endl;
  
  //STEP2: read the templated opacities, get ready for interpolation
  
  Real kappa_t1_rho1_gray=tab.kappa_ross(nt1,nrho1);
  Real kappa_t1_rho2_gray=tab.kappa_ross(nt1,nrho2);
  Real kappa_t2_rho1_gray=tab.kappa_ross(nt2,nrho1);
  Real kappa_t2_rho2_gray=tab.kappa_ross(nt2,nrho2);

  Real planck_t1_rho1_gray=tab.kappa_planck(nt1,nrho1);
  Real planck_t1_rho2_gray=tab.kappa_planck(nt1,nrho2);
  Real planck_t2_rho1_gray=tab.kappa_planck(nt2,nrho1);
  Real planck_t2_rho2_gray=tab.kappa_planck(nt2,nrho2);

  //in the case the temperature is out of range, extrapolate planck mean opacity by T^-3.5
  Real logt = log10(tgas);
  Real logtlim_table = log10(tab.temp_grid(n_temp-1));
  if(nt2 == n_temp-1 && (logt > logtlim_table)){
    Real scaling = pow(10.0, -3.5*(logt - logtlim_table));
    planck_t1_rho1_gray *= scaling;
    planck_t1_rho2_gray *= scaling;
    planck_t2_rho1_gray *= scaling;
    planck_t2_rho2_gray *= scaling;
  }

  //Note that if density is below the tabulated value, will use the lowest temperature in table

  Real rho_1 = tab.rho_grid(nrho1);
  Real rho_2 = tab.rho_grid(nrho2);

  Real t_1 = tab.temp_grid(nt1);
  Real t_2 = tab.temp_grid(nt2);

  //SPEP 3: Rossland opacity interpolation
  if (nrho1 == nrho2){ //if density both on lower or upper end of table 
    if (nt1 == nt2){ //if temperature also on lower or upper end of table
      kappa_ross = kappa_t1_rho1_gray; //use the only value, don't interpolate
    }else{ //interpolate only on temperature
      kappa_ross = kappa_t1_rho1_gray + (kappa_t2_rho1_gray - kappa_t1_rho1_gray) 
	          * (tgas - t_1)/(t_2 - t_1);
    }
  }else{ //if two densitites are different
    if(nt1 == nt2){ //if temperature index are the same, only interpolate density
      kappa_ross = kappa_t1_rho1_gray + (kappa_t1_rho2_gray - kappa_t1_rho1_gray) 
                                * (rho - rho_1)/(rho_2 - rho_1);
    }else{ //interpolate both density and temperature

      kappa_ross = kappa_t1_rho1_gray * (t_2 - tgas) * (rho_2 - rho)	
	                         /((t_2 - t_1) * (rho_2 - rho_1))
	         + kappa_t2_rho1_gray * (tgas - t_1) * (rho_2 - rho)
                                /((t_2 - t_1) * (rho_2 - rho_1))
	         + kappa_t1_rho2_gray * (t_2 - tgas) * (rho - rho_1)
                                /((t_2 - t_1) * (rho_2 - rho_1))
	         + kappa_t2_rho2_gray * (tgas - t_1) * (rho - rho_1)
                		 /((t_2 - t_1) * (rho_2 - rho_1));
    }
  }


  //STEP4: Planck opacity interpolation
  if (nrho1 == nrho2){ //if density both on lower or upper end of table 
    if (nt1 == nt2){ //if temperature also on lower or upper end of table
      kappa_planck = planck_t1_rho1_gray;
    }else{ //interpolate only on temperature
      kappa_planck = planck_t1_rho1_gray + (planck_t2_rho1_gray - planck_t1_rho1_gray)*(tgas - t_1)/(t_2 - t_1);
    }
  }else{//if two densitites are different
    if (nt1 == nt2){
      kappa_planck = planck_t1_rho1_gray + (planck_t1_rho2_gray - planck_t1_rho1_gray)*(rho - rho_1)/(rho_2 - rho_1);
    }else{ //interpolate both density and temperature
      kappa_planck = planck_t1_rho1_gray * (t_2 - tgas) * (rho_2 - rho)
	                         /((t_2 - t_1) * (rho_2 - rho_1))
                   + planck_t2_rho1_gray * (tgas - t_1) * (rho_2 - rho)
	                         /((t_2 - t_1) * (rho_2 - rho_1))
	           + planck_t1_rho2_gray * (t_2 - tgas) * (rho - rho_1)
	                         /((t_2 - t_1) * (rho_2 - rho_1))
	           + planck_t2_rho2_gray * (tgas - t_1) * (rho - rho_1)
                                 /((t_2 - t_1) * (rho_2 - rho_1));
      }
    }

}


#endif

//========================================================================================
// Athena++ astrophysical MHD code, Kokkos version
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gr_flat.cpp
//! \brief Problem generator for Cartesian flat spacetime
//!

#include <algorithm>
#include <cmath>
#include <sstream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "geodesic-grid/spherical_grid.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "radiation/radiation.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "globals.hpp"

//opacity table
#include "radiation/radiation_opacity_table.hpp"
#include <fstream>

namespace{
  KOKKOS_INLINE_FUNCTION
  static void GetBoyerLindquistCoordinates(struct tde_pgen pgen,
                                         Real x1, Real x2, Real x3,
                                         Real *pr, Real *ptheta, Real *pphi);
// KOKKOS_INLINE_FUNCTION
// static void ComputePrimitiveSingle(Real x1v, Real x2v, Real x3v, CoordData coord,
//                                    struct bondi_pgen pgen,
//                                    Real& rho, Real& pgas,
//                                    Real& uu1, Real& uu2, Real& uu3);

struct tde_pgen{
  Real spin;                // black hole spin
  Real dexcise, pexcise;    // excision parameters
  Real arad;                // radiation constant

  Real d_amb;               // initial ambient density
  Real p_amb;               // initial ambient pressure

  //injection location, velocity and threshhold
  Real x1_inj, x2_inj, x3_inj;
  Real vx1_inj, vx2_inj, vx3_inj;
  Real r_inj_thresh_coarse;
  Real local_dens, local_temp;

  //opacity table
  int n_rho;
  int n_temp;
  std::string read_rho_grid_name;
  std::string read_temp_grid_name;
  std::string read_kappa_name;
  std::string read_kappa_ross_name;
  std::string read_kappa_planck_name;

};
 tde_pgen tde;

}//namesapce

// prototypes for user-defined BCs and error functions
void FixedStreamInflow(Mesh *pm);

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for spherical tde stream problem

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  // Is radiation enabled?
  const bool is_radiation_enabled = (pmbp->prad != nullptr);

  // Get spin of black hole
  tde.spin = pmbp->pcoord->coord_data.bh_spin;

  // Get excision parameters
  tde.dexcise = pmbp->pcoord->coord_data.dexcise;
  tde.pexcise = pmbp->pcoord->coord_data.pexcise;

  // ambient gas 
  tde.p_amb   = pin->GetOrAddReal("problem", "p_amb", 1.0e-10);
  tde.d_amb   = pin->GetOrAddReal("problem", "d_amb", 1.0e-10);

  //injection point
  tde.x1_inj = pin->GetReal("problem", "x1_inj");
  tde.x2_inj = pin->GetReal("problem", "x2_inj");
  tde.x3_inj = pin->GetReal("problem", "x3_inj");
  tde.vx1_inj = pin->GetReal("problem", "vx1_inj");
  tde.vx2_inj = pin->GetReal("problem", "vx2_inj");
  tde.vx3_inj = pin->GetReal("problem", "vx3_inj");
  tde.local_dens = pin->GetReal("problem", "local_dens");
  tde.local_temp = pin->GetReal("problem", "local_temp");
  tde.r_inj_thresh_coarse = pin->GetReal("problem", "r_inj_thresh_coarse");

  //if radiation
  if (pmbp->prad != nullptr){
    tde.arad = pmbp->prad->arad;
    tde.n_rho = pin->GetInteger("problem","n_rho");
    tde.n_temp = pin->GetInteger("problem","n_temp");
    tde.read_rho_grid_name = pin->GetOrAddString("problem","rho_grid_name","");
    tde.read_temp_grid_name = pin->GetOrAddString("problem","temp_grid_name","");
    tde.read_kappa_name = pin->GetOrAddString("problem","kappa_name","");
    tde.read_kappa_ross_name = pin->GetOrAddString("problem","kappa_ross_name","");
    tde.read_kappa_planck_name = pin->GetOrAddString("problem","kappa_planck_name","");
  }

  ////check MPI tag size bound
  //int ub, flag;
  //MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &ub, &flag);
  //if(flag) printf("MPI_TAG_UB = %d\n", ub);

  // set user boundary bondition, effective is diode in three directions now
  user_bcs_func = FixedStreamInflow;

  //-------------------------------
  // load opacity table, write them into an opacitydata instance, so all devices can access to them
  OpacityData& data = OpacityData::GetInstance();
  OpacityTable& tab = data.table;

  int n_rho = tde.n_rho;
  int n_temp = tde.n_temp;
  
  //assign shape to opacity table instance
  tab.n_rho = n_rho;
  tab.n_temp = n_temp;
  
  //prepare buffer arrays to store read-in data
  HostArray1D<Real> combine_temp_grid("combine_temp_grid", n_temp);
  HostArray1D<Real> combine_rho_grid("combine_rho_grid", n_rho);
  HostArray2D<Real> combine_ross_table("combine_ross_table", n_temp, n_rho);
  HostArray2D<Real> combine_planck_table("combine_planck_table", n_temp, n_rho);
  
  // Read file into std::vector<Real> rho_vec, temp_vec, kappa_vec
  if (!tde.read_kappa_name.empty()){

    std::ifstream fin(tde.read_kappa_name);
    if (!fin.is_open()) {
      std::cerr << "Error opening opacity file" << tde.read_kappa_name<<std::endl;
      return;
    }

    // skip first two integers
    int n_rho_read, n_temp_read;
    fin >> n_temp_read >> n_rho_read;
 
    // quick sanity check
    if (n_rho_read!=n_rho){
      std::cerr<<"opacity file n_rho is:"<<n_rho_read<<", but input file n_rho is:"<<n_rho<<std::endl;
      return;
    }
    if (n_temp_read!=n_temp){
      std::cerr<<"opacity file n_temp is:"<<n_temp_read<<", but input file n_temp is:"<<n_temp<<std::endl;
      return;
    }

    // load temperature grid
    for (int i = 0; i < n_temp; ++i) {
      fin >> combine_temp_grid(i);
    }

    // load density grid
    for (int i = 0; i < n_rho; ++i) {
      fin >> combine_rho_grid(i);
    }

    // Rosseland opacity table
    for (int j = 0; j < n_temp; ++j) {
      for (int i = 0; i < n_rho; ++i) {
	fin >> combine_ross_table(j, i);
      }
    }

    // Planck opacity table
    for (int j = 0; j < n_temp; ++j) {
      for (int i = 0; i < n_rho; ++i) {
	fin >> combine_planck_table(j, i);
      }
    }

    fin.close();

  }//if kappa_name_is_empty

  // Allocatedebice  Kokkos views
  tab.rho_grid  = Kokkos::View<Real*>("rho_grid", n_rho);
  tab.temp_grid = Kokkos::View<Real*>("temp_grid", n_temp);
  tab.kappa_ross = Kokkos::View<Real**>("kappa_ross", n_temp, n_rho);
  tab.kappa_planck = Kokkos::View<Real**>("kappa_planck", n_temp, n_rho);

  // Create host mirrors and deep_copy 
  auto rho_host  = Kokkos::create_mirror_view(tab.rho_grid);
  auto temp_host = Kokkos::create_mirror_view(tab.temp_grid);
  auto kappa_ross_host  = Kokkos::create_mirror_view(tab.kappa_ross);
  auto kappa_planck_host  = Kokkos::create_mirror_view(tab.kappa_planck);

  // fill in the rho_host, temp_host, kappa_ross_host and kappa_planck_host
  for (int i = 0; i < n_temp; ++i) {
    temp_host(i) = combine_temp_grid(i);
  }
  for (int i = 0; i < n_rho; ++i) {
    rho_host(i) = combine_rho_grid(i);
  }
  for (int j = 0; j < n_temp; ++j) {
    for (int i = 0; i < n_rho; ++i) {
      kappa_ross_host(j,i) = combine_ross_table(j, i);
    }
  }
  for (int j = 0; j < n_temp; ++j) {
    for (int i = 0; i < n_rho; ++i) {
      kappa_planck_host(j,i) = combine_planck_table(j, i);
    }
  }
  
  // copy to instance
  Kokkos::deep_copy(tab.rho_grid,  rho_host);
  Kokkos::deep_copy(tab.temp_grid, temp_host);
  Kokkos::deep_copy(tab.kappa_ross, kappa_ross_host);
  Kokkos::deep_copy(tab.kappa_planck, kappa_planck_host);

  // // quick check the readings
  // for (int i=90; i<100; ++i){
  //   std::cout<<"kappa_ross_host(30, "<<i<<") = "<< kappa_ross_host(30,i)<<", kappa_planck_host(30, "<<i<<")="<< kappa_planck_host(30,i)<<std::endl;
  // }
  
  //-------------------------------------

  // return if restart
  if (restart) return;

  // capture variables for the kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &coord = pmbp->pcoord->coord_data;

  auto &size = pmbp->pmb->mb_size;

  //extract radiation parameters
  int nangles_;
  int nang1;
  DualArray2D<Real> nh_c_;
  DvceArray6D<Real> norm_to_tet_, tet_c_, tetcov_c_;
  DvceArray5D<Real> i0_;
  if (is_radiation_enabled){
    nang1 = (pmbp->prad->prgeo->nangles-1);
    nangles_ = pmbp->prad->prgeo->nangles;
    nh_c_ = pmbp->prad->nh_c;
    norm_to_tet_ = pmbp->prad->norm_to_tet;
    tet_c_ = pmbp->prad->tet_c;
    tetcov_c_ = pmbp->prad->tetcov_c;
    i0_ = pmbp->prad->i0;
  }
  
  // initialize Hydro variables ----------------------------------------------------------
  if (pmbp->phydro != nullptr) {
    auto &w0_ = pmbp->phydro->w0;
    auto tde_ = tde;
    Real g_gamma = pmbp->phydro->peos->eos_data.gamma;

    par_for("pgen_tde",DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m,int k,int j,int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      int nx3 = indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

      // Extract metric and inverse
      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski, coord.bh_spin,
			      glower, gupper);

      Real rad = std::sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
      // Calculate Boyer-Lindquist coordinates of cell
      //Real r, theta, phi;
      //GetBoyerLindquistCoordinates(tde_, x1v, x2v, x3v, &r, &theta, &phi);
      //Real sin_theta = sin(theta);
      //Real cos_theta = cos(theta);
      //Real sin_phi = sin(phi);
      //Real cos_phi = cos(phi);

      Real den = tde_.d_amb;
      Real pres = tde_.p_amb;
      
      //xs: does this need the check of excising inside horizon as gr_torus.cpp L362?
      if (rad < 1.0) {
        den = tde_.dexcise;
	pres = tde_.pexcise;
      }
      w0_(m,IDN,k,j,i) = den;
      w0_(m,IVX,k,j,i) = 0.0;
      w0_(m,IVY,k,j,i) = 0.0;
      w0_(m,IVZ,k,j,i) = 0.0;
      w0_(m,IEN,k,j,i) = pres/(g_gamma-1.0);

      
      if (is_radiation_enabled){//copied from gr_torus.cpp, initialize radiation intensity
	Real temp_init = pres/den;
	Real urad = tde_.arad * SQR(SQR(temp_init));

	//no initial velocity
	Real uu1 = 0.0;
	Real uu2 = 0.0;
	Real uu3 = 0.0;

	Real q = glower[1][1]*uu1*uu1 + 2.0*glower[1][2]*uu1*uu2 + 2.0*glower[1][3]*uu1*uu3
	  + glower[2][2]*uu2*uu2 + 2.0*glower[2][3]*uu2*uu3
	  + glower[3][3]*uu3*uu3;
	Real uu0 = sqrt(1.0 + q);
	Real u_tet_[4]; //velocity in tetrad frame
	u_tet_[0] = (norm_to_tet_(m,0,0,k,j,i)*uu0 + norm_to_tet_(m,0,1,k,j,i)*uu1 +
		     norm_to_tet_(m,0,2,k,j,i)*uu2 + norm_to_tet_(m,0,3,k,j,i)*uu3);
	u_tet_[1] = (norm_to_tet_(m,1,0,k,j,i)*uu0 + norm_to_tet_(m,1,1,k,j,i)*uu1 +
		     norm_to_tet_(m,1,2,k,j,i)*uu2 + norm_to_tet_(m,1,3,k,j,i)*uu3);
	u_tet_[2] = (norm_to_tet_(m,2,0,k,j,i)*uu0 + norm_to_tet_(m,2,1,k,j,i)*uu1 +
		     norm_to_tet_(m,2,2,k,j,i)*uu2 + norm_to_tet_(m,2,3,k,j,i)*uu3);
	u_tet_[3] = (norm_to_tet_(m,3,0,k,j,i)*uu0 + norm_to_tet_(m,3,1,k,j,i)*uu1 +
		     norm_to_tet_(m,3,2,k,j,i)*uu2 + norm_to_tet_(m,3,3,k,j,i)*uu3);

	
	// Go through each angle
	for (int n=0; n<nangles_; ++n) {
	  // Calculate direction in fluid frame
	  Real un_t = (u_tet_[1]*nh_c_.d_view(n,1) + u_tet_[2]*nh_c_.d_view(n,2) +
		       u_tet_[3]*nh_c_.d_view(n,3)); //nh_c_ 
	  Real n0_f = u_tet_[0]*nh_c_.d_view(n,0) - un_t; //fluid 
           
	  //// Calculate intensity in tetrad frame
	  Real n0 = tet_c_(m,0,0,k,j,i); 
	  Real n_0 = 0.0;
	  for (int d=0; d<4; ++d) {  
	    n_0 += tetcov_c_(m,d,0,k,j,i)*nh_c_.d_view(n,d);  
	  }
	  //printf("m:%d, n:%d, k:%d, j:%d, i:%d, n0:%g, n_0:%g, n0_f:%g, urad:%g, \n", m, n , k, j,i, n0, n_0, n0_f, urad); 
	  i0_(m,n,k,j,i) = n0*n_0*(urad/(4.0*M_PI))/SQR(SQR(n0_f));//cons
	}
	
      }
      
      }//ijk
     );//par for


    // Convert primitives to conserved
    pmbp->phydro->peos->PrimToCons(w0_, pmbp->phydro->u0, is, ie, js, je, ks, ke);
  }  // End initialization Hydro variables

  // initialize MHD variables ------------------------------------------------------------
  // if (pmbp->pmhd != nullptr) {
  // }

  return;
}


//----------------------------------------------------------------------------------------
//! \fn FixedStreamInflow
//  \brief Sets boundary condition on surfaces of computational domain

void FixedStreamInflow(Mesh *pm) {
  auto &indcs = pm->mb_indcs;
  auto &size = pm->pmb_pack->pmb->mb_size;
  auto &coord = pm->pmb_pack->pcoord->coord_data;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  auto &mb_bcs = pm->pmb_pack->pmb->mb_bcs;
  auto tde_ = tde;

  int nmb = pm->pmb_pack->nmb_thispack;
  auto u0_ = pm->pmb_pack->phydro->u0;
  auto w0_ = pm->pmb_pack->phydro->w0;
  Real g_gamma = pm->pmb_pack->phydro->peos->eos_data.gamma;

  // intensity array and n_angle if radiation is enabled
  const bool is_radiation_enabled = (pm->pmb_pack->prad != nullptr);
  DvceArray5D<Real> i0_; int nang1;
  if (is_radiation_enabled) {
    i0_ = pm->pmb_pack->prad->i0;
    nang1 = pm->pmb_pack->prad->prgeo->nangles - 1;
  }

  pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,is-ng,is-1,0,(n2-1),0,(n3-1));
  pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,ie+1,ie+ng,0,(n2-1),0,(n3-1));
  par_for("fixed_x1", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,(ng-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // inner x1 boundary
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    //Real rho, pgas, uu1, uu2, uu3;
    if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
      //ComputePrimitiveSingle(x1v,x2v,x3v,coord,bondi_,rho,pgas,uu1,uu2,uu3);
      w0_(m,IDN,k,j,is-i-1) = w0_(m,IDN,k,j,is);
      w0_(m,IEN,k,j,is-i-1) = w0_(m,IEN,k,j,is);
      w0_(m,IM1,k,j,is-i-1) = fmin(0.0, w0_(m,IM1,k,j,is));
      w0_(m,IM2,k,j,is-i-1) = w0_(m,IM2,k,j,is);
      w0_(m,IM3,k,j,is-i-1) = w0_(m,IM3,k,j,is);
    }

    // outer x1 boundary
    x1v = CellCenterX((ie+i+1)-is, indcs.nx1, x1min, x1max);

    if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
      Real r_now = std::sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
      Real dr_now = std::sqrt(SQR(x1v - tde_.x1_inj) + SQR(x2v - tde_.x2_inj) + SQR(x3v - tde_.x3_inj));
      if (dr_now <= tde_.r_inj_thresh_coarse){
	printf("x1v:%g, x2v:%g, x3v:%g, x1inj:%g, x2inj:%g, x3inj:%g, rnow:%g, dr_now:%g\n", x1v, x2v, x3v, tde_.x1_inj, tde_.x2_inj, tde_.x3_inj, r_now, dr_now);
	w0_(m,IDN,k,j,(ie+i+1)) = tde_.local_dens;
	w0_(m,IEN,k,j,(ie+i+1)) = tde_.local_dens * tde_.local_temp * (g_gamma-1.0);
	w0_(m,IM1,k,j,(ie+i+1)) = tde_.vx1_inj;
	w0_(m,IM2,k,j,(ie+i+1)) = tde_.vx2_inj;
	w0_(m,IM3,k,j,(ie+i+1)) = tde_.vx3_inj;
	printf("i:%d, local density:%g, temp:%g, ein:%g\n", ie+i+1, w0_(m,IDN,k,j,(ie+i+1)), w0_(m,IEN,k,j,(ie+i+1))/w0_(m,IDN,k,j,(ie+i+1))/(g_gamma-1.0), w0_(m,IEN,k,j,(ie+i+1)));
      }else{
	//ComputePrimitiveSingle(x1v,x2v,x3v,coord,bondi_, rho,pgas,uu1,uu2,uu3);
	w0_(m,IDN,k,j,(ie+i+1)) = w0_(m,IDN,k,j,ie);
	w0_(m,IEN,k,j,(ie+i+1)) = w0_(m,IEN,k,j,ie);
	w0_(m,IM1,k,j,(ie+i+1)) = fmax(0.0, w0_(m,IM1,k,j,ie));
	w0_(m,IM2,k,j,(ie+i+1)) = w0_(m,IM2,k,j,ie);
	w0_(m,IM3,k,j,(ie+i+1)) = w0_(m,IM3,k,j,ie);
      }
    }
  });

  if (is_radiation_enabled) {
    // Set X1-BCs on i0 if Meshblock face is at the edge of computational domain
    par_for("outflow_rad_x1", DevExeSpace(),0,(nmb-1),0,nang1,0,(n3-1),0,(n2-1),
    KOKKOS_LAMBDA(int m, int n, int k, int j) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          i0_(m,n,k,j,is-i) = i0_(m,n,k,j,is);
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          i0_(m,n,k,j,ie+i) = i0_(m,n,k,j,ie);
        }
      }
    });
  }

  // PrimToCons on X1 physical boundary ghost zones
  pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,is-ng,is-1,0,(n2-1),0,(n3-1));
  pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,ie+1,ie+ng,0,(n2-1),0,(n3-1));

  pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),js-ng,js-1,0,(n3-1));
  pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),je+1,je+ng,0,(n3-1));

  par_for("fixed_x2", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(ng-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // inner x2 boundary
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    //Real rho, pgas, uu1, uu2, uu3;
    if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
      //ComputePrimitiveSingle(x1v,x2v,x3v,coord,bondi_,rho,pgas,uu1,uu2,uu3);
      w0_(m,IDN,k,js-j-1,i) = w0_(m,IDN,k,js,i);
      w0_(m,IEN,k,js-j-1,i) = w0_(m,IEN,k,js,i);
      w0_(m,IM1,k,js-j-1,i) = w0_(m,IM1,k,js,i);
      w0_(m,IM2,k,js-j-1,i) = fmin(0.0, w0_(m,IM2,k,js,i));
      w0_(m,IM3,k,js-j-1,i) = w0_(m,IM3,k,js,i);
    }

    // outer x2 boundary
    x2v = CellCenterX((je+j+1)-js, indcs.nx2, x2min, x2max);

    if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
      
      w0_(m,IDN,k,(je+j+1),i) = w0_(m,IDN,k,je,i);
      w0_(m,IEN,k,(je+j+1),i) = w0_(m,IEN,k,je,i);
      w0_(m,IM1,k,(je+j+1),i) = w0_(m,IM1,k,je,i);
      w0_(m,IM2,k,(je+j+1),i) = fmax(0.0, w0_(m,IM2,k,je,i));
      w0_(m,IM3,k,(je+j+1),i) = w0_(m,IM3,k,je,i);
    }
  });
  if (is_radiation_enabled) {
    // Set X2-BCs on i0 if Meshblock face is at the edge of computational domain
    par_for("outflow_rad_x2", DevExeSpace(),0,(nmb-1),0,nang1,0,(n3-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int n, int k, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          i0_(m,n,k,js-j-1,i) = i0_(m,n,k,js,i);
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          i0_(m,n,k,je+j+1,i) = i0_(m,n,k,je,i);
        }
      }
    });
  }//radiation

  pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),js-ng,js-1,0,(n3-1));
  pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),je+1,je+ng,0,(n3-1));

  pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),0,(n2-1),ks-ng,ks-1);
  pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),0,(n2-1),ke+1,ke+ng);
  par_for("fixed_ix3", DevExeSpace(),0,(nmb-1),0,(ng-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // inner x3 boundary
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    //Real rho, pgas, uu1, uu2, uu3;
    if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
      //ComputePrimitiveSingle(x1v,x2v,x3v,coord,bondi_,rho,pgas,uu1,uu2,uu3);
      w0_(m,IDN,ks-k-1,j,i) = w0_(m,IDN,ks,j,i);
      w0_(m,IEN,ks-k-1,j,i) = w0_(m,IEN,ks,j,i);
      w0_(m,IM1,ks-k-1,j,i) = w0_(m,IM1,ks,j,i);
      w0_(m,IM2,ks-k-1,j,i) = w0_(m,IM2,ks,j,i);
      w0_(m,IM3,ks-k-1,j,i) = fmin(0.0, w0_(m,IM3,ks,j,i));
    }

    // outer x3 boundary
    x3v = CellCenterX((ke+k+1)-ks, indcs.nx3, x3min, x3max);

    if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
      //ComputePrimitiveSingle(x1v,x2v,x3v,coord,bondi_,rho,pgas,uu1,uu2,uu3);
      w0_(m,IDN,(ke+k+1),j,i) = w0_(m,IDN,ke,j,i);
      w0_(m,IEN,(ke+k+1),j,i) = w0_(m,IEN,ke,j,i);
      w0_(m,IM1,(ke+k+1),j,i) = w0_(m,IM1,ke,j,i);
      w0_(m,IM2,(ke+k+1),j,i) = w0_(m,IM2,ke,j,i);
      w0_(m,IM3,(ke+k+1),j,i) = fmax(0.0, w0_(m,IM3,ke,j,i));
    }
  });

  if (is_radiation_enabled) {
    // Set X3-BCs on i0 if Meshblock face is at the edge of computational domain
    par_for("outflow_rad_x3", DevExeSpace(),0,(nmb-1),0,nang1,0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int n, int j, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          i0_(m,n,ks-k-1,j,i) = i0_(m,n,ks,j,i);
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          i0_(m,n,ke+k+1,j,i) = i0_(m,n,ke,j,i);
        }
      }
    });
  }

  pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),0,(n2-1),ks-ng,ks-1);
  pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),0,(n2-1),ke+1,ke+ng);

  return;
}




namespace {
//----------------------------------------------------------------------------------------
// Function for returning corresponding Boyer-Lindquist coordinates of point
// Inputs:
//   x1,x2,x3: global coordinates to be converted
// Outputs:
//   pr,ptheta,pphi: variables pointed to set to Boyer-Lindquist coordinates

KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(struct tde_pgen pgen,
                                         Real x1, Real x2, Real x3,
                                         Real *pr, Real *ptheta, Real *pphi) {
  Real rad = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  Real r = fmax((sqrt( SQR(rad) - SQR(pgen.spin) + sqrt(SQR(SQR(rad)-SQR(pgen.spin))
                      + 4.0*SQR(pgen.spin)*SQR(x3)) ) / sqrt(2.0)), 1.0);
  *pr = r;
  *ptheta = (fabs(x3/r) < 1.0) ? acos(x3/r) : acos(copysign(1.0, x3));
  *pphi = atan2(r*x2-pgen.spin*x1, pgen.spin*x2+r*x1) -
          pgen.spin*r/(SQR(r)-2.0*r+SQR(pgen.spin));
  return;
}

}//namespace

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
#include "radiation/radiation_opacity.hpp" //added to enroll user opacity
#include "dyn_grmhd/dyn_grmhd.hpp"


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

  //binary file read-in datas
  std::string bin_file; //name of the binary file to intialize domain, 
  int binary_nbin1; //binary file size nx1
  int binary_nbin2; //binary file size nx2
  int binary_nbin3; //binary file size nx3
  int binary_n_vars; //number of variable in binary file
  int binary_n_mb; //number of meshblock in binary file
  Real binary_rmax; //binary file maxium radius (physical size of square box)
  Real binary_x0; //binary file center
  Real binary_y0; //binary file center
  Real binary_z0; //binary file center
  //Real binary_ibc_fac;

};
tde_pgen tde;
 
//function to interpolate a grid point in the domain locates at x,y,z
//using values from a 5D data_array with shape (N_mb, N_var, nx, ny, nz)
KOKKOS_INLINE_FUNCTION Real TrilinearInterpolate(
    const DvceArray5D<Real>& data_array, int var_idx,
    Real x, Real y, Real z, Real rmax, Real x0, Real y0, Real z0, 
    int nx, int ny, int nz);

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

  //binary file
  tde.bin_file = pin->GetOrAddString("problem","bin_file","");
  tde.binary_nbin1 = pin->GetOrAddInteger("problem", "binary_nbin1", 64);
  tde.binary_nbin2 = pin->GetOrAddInteger("problem", "binary_nbin2", 64);
  tde.binary_nbin3 = pin->GetOrAddInteger("problem", "binary_nbin3", 64);
  tde.binary_rmax = pin->GetOrAddReal("problem", "binary_rmax", 50.0); //in rg for GR run
  tde.binary_x0 = pin->GetOrAddReal("problem", "binary_x0", 0.0); //in rg for GR run
  tde.binary_y0 = pin->GetOrAddReal("problem", "binary_y0", 0.0); //in rg for GR run
  tde.binary_z0 = pin->GetOrAddReal("problem", "binary_z0", 0.0); //in rg for GR run
  tde.binary_n_vars = pin->GetOrAddInteger("problem", "binary_n_vars", 1);
  tde.binary_n_mb = pin->GetOrAddInteger("problem", "binary_n_mb", 1);
  //tde.binary_ibc_fac = pin->GetOrAddReal("problem", "binary_ibc_fac", 0.9);

  //if radiation
  if (pmbp->prad != nullptr){
    tde.arad = pmbp->prad->arad;
  }

  // // set user boundary bondition, 
  // user_bcs_func = FixedStreamInflow;

  // return if restart
  if (restart) return;

  // capture variables for the kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  auto &coord = pmbp->pcoord->coord_data;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie, nx1 = indcs.nx1;
  int &js = indcs.js; int &je = indcs.je, nx2 = indcs.nx2;
  int &ks = indcs.ks; int &ke = indcs.ke, nx3 = indcs.nx3;
  int &ng = indcs.ng;
  int nmb = pmbp->nmb_thispack;

  // Get ready for read in array -------------------------
  // Static variables to ensure we only read the binary file once and keep data persistent
  // See note later, issue when free memory at end of the run
  static bool binary_read = false;
  static DvceArray5D<Real> binary_data_device("binary_data_device", 1, 1, 1, 1, 1);
  static int binary_nmb = tde.binary_n_mb; // Number of MeshBlocks in binary data
  static int binary_n_vars = tde.binary_n_vars; //Number of variables in binary data
  int &binary_nbin1_ = tde.binary_nbin1;
  int &binary_nbin2_ = tde.binary_nbin2;
  int &binary_nbin3_ = tde.binary_nbin3;
  Real &binary_rmax_ = tde.binary_rmax;
  Real &binary_x0_ = tde.binary_x0;
  Real &binary_y0_ = tde.binary_y0;
  Real &binary_z0_ = tde.binary_z0;
  //Real &binary_ibc_fac_ = tde.binary_ibc_fac;

  //std::cout<<"!binary_read "<<!binary_read<<" tde.bin_file.empty() "<<tde.bin_file.empty()<<"bin file name: "<<tde.bin_file<<std::endl;
  if (!binary_read && !tde.bin_file.empty()){
    // Open binary file using standard C I/O
    FILE* fp = std::fopen(tde.bin_file.c_str(), "rb");
    if (fp == nullptr) {
      if (global_variable::my_rank == 0) {
        std::cout << "Warning: Binary file '" << tde.bin_file 
                  << "' could not be opened for inner boundary condition" << std::endl;
      }
      binary_read = true; // Mark as read to avoid repeated attempts
      return;
    }

    //Allocate storage for binary file on host
    int cells = binary_nbin1_ * binary_nbin2_ * binary_nbin3_;
    std::size_t data_size = (cells * binary_n_vars) * sizeof(Real);
    HostArray5D<Real> binary_data_host("binary_data_host", binary_nmb, binary_n_vars, 
                                      binary_nbin3_, binary_nbin2_, binary_nbin1_);

    // Read data from binary file (assume we want the first MeshBlock)
    // Skip any header - for simplicity, assume data starts immediately
    auto mbptr = Kokkos::subview(binary_data_host, 0, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    std::size_t elements_to_read = mbptr.size();
    std::size_t elements_read = std::fread(mbptr.data(), sizeof(Real), elements_to_read, fp);

    // // Simple debug to check the elements are read in correctly
    // if (global_variable::my_rank == 0) {
    //   // std::cout << "Sample host values: "
    //   // 		<< binary_data_host(0, 0, 33, 32, 32) << ", "
    //   // 		<< binary_data_host(0, 0, 32, 33, 32) << ", "
    //   // 		<< binary_data_host(0, 0, 32, 32, 33) << std::endl;
    //   for (int v = 0; v < binary_n_vars; v++) {
    // 	std::cout << "var " << v << " value: "
    //           << binary_data_host(0, v, 32, 32, 32) << std::endl;
    //   }

    // }
    // //check binary size
    // std::fseek(fp, 0, SEEK_END);
    // std::cout<<"File size (bytes): "<<std::ftell(fp)<<std::endl;
    // std::rewind(fp);
    // std::cout<<"elements_to_read*size of Real number: "<<elements_to_read * sizeof(Real)<<std::endl;

    if (elements_read != elements_to_read) {
      if (global_variable::my_rank == 0) {
        std::cout << "Warning: Read " << elements_read << " elements, expected " 
                  << elements_to_read << " from binary file '" << tde.bin_file << "'" << std::endl;
      }
    }

    // Reallocate device array with correct dimensions and copy data
    // The code tries to free binary_data_device after kokkos:finalize() bc it's a static array, giving seg fault at end 
    Kokkos::realloc(binary_data_device, binary_nmb, binary_n_vars, binary_nbin3_, binary_nbin2_, binary_nbin1_);
    Kokkos::deep_copy(binary_data_device, binary_data_host);

    std::fclose(fp);
    binary_read = true;
    
    if (global_variable::my_rank == 0) {
      std::cout << "Binary data loaded from '" << tde.bin_file 
                << "' for initialize the in-domain data" << std::endl;
      std::cout << "Binary data dimensions: " << binary_nmb << " MeshBlocks, "
                << binary_n_vars << " variables, "
                << binary_nbin1_ << " x " << binary_nbin2_ << " x " << binary_nbin3_ 
                << " cells" << std::endl;
    }

  }//end bin_file not empty, loading binary file on host and copy on device
  // End read in array ---------------------------------------

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

    if (binary_read && !tde_.bin_file.empty()) {
      // Capture static variables for device kernel
      //can this loop inside of the par_for
      auto binary_data_d = binary_data_device;
      int binary_nbin1_d = binary_nbin1_;
      int binary_nbin2_d = binary_nbin2_;
      int binary_nbin3_d = binary_nbin3_;
      int binary_n_vars_d = binary_n_vars;
      Real binary_rmax_d = binary_rmax_;
      Real binary_x0_d = binary_x0_;
      Real binary_y0_d = binary_y0_;
      Real binary_z0_d = binary_z0_;
      //Real binary_ibc_fac_d = binary_ibc_fac_;
    
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
      Real den = tde_.d_amb;
      Real pres = tde_.p_amb;
      Real vx = 0.0;
      Real vy = 0.0;
      Real vz = 0.0;

      //if (binary_read && !tde_.bin_file.empty()) {
      den = TrilinearInterpolate(binary_data_d, 0, x1v, x2v, x3v, 
				 binary_rmax_d, binary_x0_d, binary_y0_d, binary_z0_d,
				 binary_nbin1_d, binary_nbin2_d, binary_nbin3_d);
      //std::cout<<"interpolating nmb="<<m<<", k="<<k<<", j="<<j<<", i="<<i<<" x1v:"<<x1v<<" x2v:"<<x2v<<" x3v:"<<x3v<<", dens="<<den<<std::endl;
	//}//end binary_read
      vx = TrilinearInterpolate(binary_data_d, 1, x1v, x2v, x3v, 
				 binary_rmax_d, binary_x0_d, binary_y0_d, binary_z0_d,
				 binary_nbin1_d, binary_nbin2_d, binary_nbin3_d);
      vy = TrilinearInterpolate(binary_data_d, 2, x1v, x2v, x3v, 
				 binary_rmax_d, binary_x0_d, binary_y0_d, binary_z0_d,
				 binary_nbin1_d, binary_nbin2_d, binary_nbin3_d);
      vz = TrilinearInterpolate(binary_data_d, 3, x1v, x2v, x3v, 
				 binary_rmax_d, binary_x0_d, binary_y0_d, binary_z0_d,
				 binary_nbin1_d, binary_nbin2_d, binary_nbin3_d);
      
      pres = den*tde_.local_temp;
      
      //xs: does this need the check of excising inside horizon as gr_torus.cpp L362?
      if (rad < 1.0) {
        den = tde_.dexcise;
	pres = tde_.pexcise;
      }
      w0_(m,IDN,k,j,i) = den;
      w0_(m,IVX,k,j,i) = vx;
      w0_(m,IVY,k,j,i) = vy;
      w0_(m,IVZ,k,j,i) = vz;
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

  }//end binary is not empty
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




//----------------------------------------------------------------------------------------
// Device function for trilinear interpolation, ChatGPT made this figure
        // z ↑
        //   |
        //   |     c110 -------- c111
        //   |     /|            /|
        //   |    / |           / |
        //   | c100 -------- c101 |
        //   |   |  |         |  |
        //   |   |  c010 -----|--c011
        //   |   | /          | /
        //   |   |/           |/
        //   |  c000 -------- c001
        //   |
        //   +------------------------→ x
        //  /
        // y
  
KOKKOS_INLINE_FUNCTION Real TrilinearInterpolate(
    const DvceArray5D<Real>& data_array, int var_idx,
    Real x, Real y, Real z, Real rmax, Real x0, Real y0, Real z0, 
    int nx, int ny, int nz) {
  
  // Map physical coordinates to array indices (continuous)
  Real fx = (x - x0 + rmax) / (2.0 * rmax) * (nx - 1);
  Real fy = (y - y0 + rmax) / (2.0 * rmax) * (ny - 1);
  Real fz = (z - z0 + rmax) / (2.0 * rmax) * (nz - 1);
  
  // Check bounds
  if (fx < 0.0 || fx >= nx-1 || fy < 0.0 || fy >= ny-1 || fz < 0.0 || fz >= nz-1) {
    return 1.0e-20; // Return default value if outside bounds
  }
  
  // Get integer indices and fractional parts
  int i0 = static_cast<int>(fx);
  int j0 = static_cast<int>(fy);
  int k0 = static_cast<int>(fz);
  int i1 = std::min(i0 + 1, nx - 1);
  int j1 = std::min(j0 + 1, ny - 1);
  int k1 = std::min(k0 + 1, nz - 1);
  
  Real dx = fx - i0;
  Real dy = fy - j0;
  Real dz = fz - k0;
  
  // Trilinear interpolation
  Real c000 = data_array(0, var_idx, k0, j0, i0);
  Real c001 = data_array(0, var_idx, k0, j0, i1);
  Real c010 = data_array(0, var_idx, k0, j1, i0);
  Real c011 = data_array(0, var_idx, k0, j1, i1);
  Real c100 = data_array(0, var_idx, k1, j0, i0);
  Real c101 = data_array(0, var_idx, k1, j0, i1);
  Real c110 = data_array(0, var_idx, k1, j1, i0);
  Real c111 = data_array(0, var_idx, k1, j1, i1);
  
  Real c00 = c000 * (1.0 - dx) + c001 * dx;
  Real c01 = c010 * (1.0 - dx) + c011 * dx;
  Real c10 = c100 * (1.0 - dx) + c101 * dx;
  Real c11 = c110 * (1.0 - dx) + c111 * dx;
  
  Real c0 = c00 * (1.0 - dy) + c01 * dy;
  Real c1 = c10 * (1.0 - dy) + c11 * dy;

  //std::cout<<"k0: "<<k0<<" j0: "<<j0<<" i0: "<<i0<<" k1: "<<k1<<" j1: "<<j1<<" i1: "<<i1<<" c000: "<<c000<<" c001: "<<c001<<std::endl;
  
  return c0 * (1.0 - dz) + c1 * dz;
}


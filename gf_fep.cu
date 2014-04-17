#include "svt_utils.h"

__global__ void gf_fep_comb_GPU (evt_arrays* evt_dev, fep_arrays* fep_dev, int maxEvt) {

  int ie, ir;

  int nlyr; // The number of layers with a hit
  int ncomb; // The number of combinations

  ie = blockIdx.x; // events index
  ir = threadIdx.x; // roads index

  // la memoria __shared__ è accessibile solo a livello di blocco:
  // dato che ogni evento è gestito da un singolo blocco di thread,
  // possiamo copiare in memoria shared solo i dati riguardanti al
  // singolo evento.

  __shared__ int sh_fep_ncmb[MAXROAD];
  __shared__ int sh_fep_zid[MAXROAD];
  __shared__ int sh_fep_road[MAXROAD];
  __shared__ int sh_fep_sect[MAXROAD];
  __shared__ int sh_fep_cable_sect[MAXROAD];
  __shared__ int sh_fep_err[MAXROAD];

  __shared__ int sh_evt_err[MAXROAD];

  if ( ( ie < maxEvt ) &&
		  ( ir < evt_dev->evt_nroads[ie] ) ) {

	  sh_fep_ncmb[ir] = 0;
	  sh_fep_zid[ir] = 0;
	  sh_fep_road[ir] = 0;
	  sh_fep_sect[ir] = 0;
	  sh_fep_cable_sect[ir] = 0;
	  sh_fep_err[ir] = 0;
	  sh_evt_err[ir] = 0;

		ncomb = 1;
		nlyr = 0;

		int this_ncomb, this_nlyr;
		// At first, we calculate how many combinations are there
		for (int id=0; id<(XFT_LYR+1); id++) {
		  /*if (evt_dev->evt_nhits[ie][ir][id] != 0) {
			ncomb *= evt_dev->evt_nhits[ie][ir][id];
			nlyr++;
		  }*/

		  this_ncomb = ncomb;
		  this_nlyr = nlyr;
		  //evaluate condition
		  int t = (evt_dev->evt_nhits[ie][ir][id] != 0) ? ~0 : 0;
		  //calculate branches (in this case, just one)
		  this_ncomb *= evt_dev->evt_nhits[ie][ir][id];
		  this_nlyr++;
		  // mask and “blend”
		  ncomb = (t&this_ncomb) | (~t&ncomb);
		  nlyr = (t&this_nlyr) | (~t&nlyr);
		}

		if ( nlyr < MINHITS )
			sh_evt_err[ir] |= (1<<UFLOW_HIT_BIT);

		sh_fep_ncmb[ir] = ncomb;
		atomicOr(&evt_dev->evt_err_sum[ie], sh_evt_err[ir]);

		sh_fep_zid[ir] = (evt_dev->evt_zid[ie][ir] & gf_mask_GPU(GF_ZID_WIDTH));
		sh_fep_road[ir] = (evt_dev->evt_road[ie][ir] & gf_mask_GPU(SVT_ROAD_WIDTH));
		sh_fep_sect[ir] = (evt_dev->evt_sect[ie][ir] & gf_mask_GPU(SVT_SECT_WIDTH));
		sh_fep_cable_sect[ir] = (evt_dev->evt_cable_sect[ie][ir] & gf_mask_GPU(SVT_SECT_WIDTH));
		sh_fep_err[ir] = sh_evt_err[ir];

		evt_dev->evt_err[ie][ir] = sh_evt_err[ir];

		fep_dev->fep_ncmb[ie][ir] = sh_fep_ncmb[ir];
		fep_dev->fep_zid[ie][ir] = sh_fep_zid[ir];
		fep_dev->fep_road[ie][ir] = sh_fep_road[ir];
		fep_dev->fep_sect[ie][ir] = sh_fep_sect[ir];
		fep_dev->fep_cable_sect[ie][ir] = sh_fep_cable_sect[ir];
		fep_dev->fep_err[ie][ir] = sh_fep_err[ir];

		fep_dev->fep_nroads[ie]  = evt_dev->evt_nroads[ie];
		fep_dev->fep_ee_word[ie] = evt_dev->evt_ee_word[ie];
		fep_dev->fep_err_sum[ie] = evt_dev->evt_err_sum[ie];
  }
}

/*//STRIDE loop version
 __global__ void gf_fep_comb_GPU2 (evt_arrays* evt_dev, fep_arrays* fep_dev, int maxEvt) {

  int ie, ir;

  int nlyr; // The number of layers with a hit
  int ncomb; // The number of combinations

  //ie = blockIdx.x; // events index
  ir = threadIdx.x; // roads index

  __shared__ int sh_fep_ncmb[MAXROAD];
  __shared__ int sh_fep_zid[MAXROAD];
  __shared__ int sh_fep_road[MAXROAD];
  __shared__ int sh_fep_sect[MAXROAD];
  __shared__ int sh_fep_cable_sect[MAXROAD];
  __shared__ int sh_fep_err[MAXROAD];

  __shared__ int sh_evt_err[MAXROAD];


  for(ie = blockIdx.x * blockDim.x + threadIdx.x; ( ie < maxEvt ); ie += blockDim.x * gridDim.x){


	  sh_fep_ncmb[ir] = 0;
	  sh_fep_zid[ir] = 0;
	  sh_fep_road[ir] = 0;
	  sh_fep_sect[ir] = 0;
	  sh_fep_cable_sect[ir] = 0;
	  sh_fep_err[ir] = 0;

	  sh_evt_err[ir] = 0;

	  if ( //( ie < maxEvt ) &&
		  ( ir < evt_dev->evt_nroads[ie] ) ) {

		ncomb = 1;
		nlyr = 0;
		// At first, we calculate how many combinations are there
		for (int id=0; id<(XFT_LYR+1); id++) {
		  if (evt_dev->evt_nhits[ie][ir][id] != 0) {
			ncomb *= evt_dev->evt_nhits[ie][ir][id];
			nlyr++;
		  }

		}

		if ( nlyr < MINHITS )
			sh_evt_err[ir] |= (1<<UFLOW_HIT_BIT);

		sh_fep_ncmb[ir] = ncomb;
		atomicOr(&evt_dev->evt_err_sum[ie], sh_evt_err[ir]);

		sh_fep_zid[ir] = (evt_dev->evt_zid[ie][ir] & gf_mask_GPU(GF_ZID_WIDTH));
		sh_fep_road[ir] = (evt_dev->evt_road[ie][ir] & gf_mask_GPU(SVT_ROAD_WIDTH));
		sh_fep_sect[ir] = (evt_dev->evt_sect[ie][ir] & gf_mask_GPU(SVT_SECT_WIDTH));
		sh_fep_cable_sect[ir] = (evt_dev->evt_cable_sect[ie][ir] & gf_mask_GPU(SVT_SECT_WIDTH));
		sh_fep_err[ir] = sh_evt_err[ir];



	  }

	  evt_dev->evt_err[ie][ir] = sh_evt_err[ir];

	  fep_dev->fep_ncmb[ie][ir] = sh_fep_ncmb[ir];
	  fep_dev->fep_zid[ie][ir] = sh_fep_zid[ir];
	  fep_dev->fep_road[ie][ir] = sh_fep_road[ir];
	  fep_dev->fep_sect[ie][ir] = sh_fep_sect[ir];
	  fep_dev->fep_cable_sect[ie][ir] = sh_fep_cable_sect[ir];
	  fep_dev->fep_err[ie][ir] = sh_fep_err[ir];

	  fep_dev->fep_nroads[ie]  = evt_dev->evt_nroads[ie];
	  fep_dev->fep_ee_word[ie] = evt_dev->evt_ee_word[ie];
	  fep_dev->fep_err_sum[ie] = evt_dev->evt_err_sum[ie];
  }

}
*/


__global__ void gf_fep_set_GPU (evt_arrays* evt_dev, fep_arrays* fep_dev, int maxEvt) {

  int ie, ir, ic;
  int icomb; /* The number of combinations */

  ie = blockIdx.x; // events index

  ir = blockIdx.y * blockDim.y + threadIdx.y ; // roads index

  ic = threadIdx.x; // comb index

  // la memoria __shared__ è accessibile solo a livello di blocco:
  // dato che ogni evento è gestito da un singolo blocco di thread,
  // possiamo copiare in memoria shared solo i dati riguardanti al
  // singolo evento.

  __shared__ int sh_fep_hitmap[MAXCOMB];
  __shared__ int sh_evt_nhits[MAXROAD][NSVX_PLANE+1];

  if ( ( ie < maxEvt ) && ( ir < fep_dev->fep_nroads[ie] ) /*&& ( ic < fep_dev->fep_ncmb[ie][ir] )*/ ) {


	for (int id=0; id<NSVX_PLANE+1; id++) {
	  sh_evt_nhits[ir][id] = evt_dev->evt_nhits[ie][ir][id];
	}

	// first initialize fep arrays
	fep_dev->fep_lcl[ie][ir][ic] = 0;
	sh_fep_hitmap[ic] = 0;
	fep_dev->fep_phi[ie][ir][ic] = 0;
	fep_dev->fep_crv[ie][ir][ic] = 0;
	fep_dev->fep_lclforcut[ie][ir][ic] = 0;
	fep_dev->fep_ncomb5h[ie][ir][ic] = 0;
	fep_dev->fep_crv_sign[ie][ir][ic] = 0;

	for (int id=0; id<XFT_LYR; id++) {
	  fep_dev->fep_hit[ie][ir][ic][id] = 0;
	  fep_dev->fep_hitZ[ie][ir][ic][id] = 0;
	}

    icomb = ic;

    for (int id=0; id<XFT_LYR; id++) {

      if (sh_evt_nhits[ir][id] != 0) {
        fep_dev->fep_hit[ie][ir][ic][id] = evt_dev->evt_hit[ie][ir][id][icomb%sh_evt_nhits[ir][id]];
        fep_dev->fep_hitZ[ie][ir][ic][id] = evt_dev->evt_hitZ[ie][ir][id][icomb%sh_evt_nhits[ir][id]];
        fep_dev->fep_lcl[ie][ir][ic] |= ((evt_dev->evt_lcl[ie][ir][id][icomb%sh_evt_nhits[ir][id]] & gf_mask_GPU(1)) << id);
        fep_dev->fep_lclforcut[ie][ir][ic] |= ((evt_dev->evt_lclforcut[ie][ir][id][icomb%sh_evt_nhits[ir][id]] & gf_mask_GPU(1)) << id);
        icomb /= sh_evt_nhits[ir][id];
        sh_fep_hitmap[ic] |= (1<<id);
      } /* if (evt_dev->evt_nhits[ie][ir][id] |= 0)  */

      __syncthreads();

    } /* for (id=0; id<XFT_LYR; id++) */

    /* check if this is a 5/5 track */
    int t = (sh_fep_hitmap[ic] != 0x1f) ? ~0 : 0;
    fep_dev->fep_ncomb5h[ie][ir][ic] = (t&1) | (~t&5);

    if (sh_evt_nhits[ir][XFT_LYR] != 0) {
      fep_dev->fep_phi[ie][ir][ic] = (evt_dev->evt_phi[ie][ir][icomb%sh_evt_nhits[ir][XFT_LYR]] & gf_mask_GPU(SVT_PHI_WIDTH));
      fep_dev->fep_crv[ie][ir][ic] = (evt_dev->evt_crv[ie][ir][icomb%sh_evt_nhits[ir][XFT_LYR]] & gf_mask_GPU(SVT_CRV_WIDTH));
      fep_dev->fep_crv_sign[ie][ir][ic] = (evt_dev->evt_crv_sign[ie][ir][icomb%sh_evt_nhits[ir][XFT_LYR]]);
    }

  }


  fep_dev->fep_hitmap[ie][ir][ic] = sh_fep_hitmap[ic];

  for (int id=0; id<XFT_LYR; id++) {
  	  evt_dev->evt_nhits[ie][ir][id] = sh_evt_nhits[ir][id];
  }
  __syncthreads();
}

void gf_fep_GPU( evt_arrays* evt_dev, fep_arrays* fep_dev, int maxEvt , cudaStream_t stream ) {

	extern cudaDeviceProp deviceProp;

  gf_fep_comb_GPU<<<maxEvt, MAXROAD, 0, stream>>>(evt_dev, fep_dev, maxEvt);

  // --- Prendiamo le informazioni specifiche della GPU per la divisione del lavoro appropriata

    unsigned int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    // dividiamo adeguatamente il lavoro di ogni evento
    // in base al numero massimo di thread disponibili in un singolo thread-block
    unsigned int num_groupBlocks_per_evt = (MAXROAD*MAXCOMB) / maxThreadsPerBlock;
    // calcoliamo la dimensione dei blocchi: manteniamo una dimensione fissa (x) e scaliamo l'altra
    unsigned int block_y_dim = maxThreadsPerBlock / MAXCOMB;
    dim3 blocks(MAXCOMB, block_y_dim);
    // calcoliamo la dimensione della grid
    dim3 grid(maxEvt, num_groupBlocks_per_evt);

    gf_fep_set_GPU<<<grid, blocks, 0, stream>>>(evt_dev, fep_dev, maxEvt);
}



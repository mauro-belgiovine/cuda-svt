#include "svt_utils.h"


/* __global__ void gf_unpack(unsigned int start_word, int *d_ids, int *d_out1, int *d_out2, int *d_out3, struct evt_arrays *evt_dev){

	int evt = threadIdx.x;
	bool gf_xft = 0;
	int id_last = -1;

	int id = d_ids[start_word];

	if (id == XFT_LYR_2) { // compatibility - stp
		id = XFT_LYR;
	    gf_xft = 1;
	}

	int nroads = evt_dev->evt_nroads[evt];
	int nhits = evt_dev->evt_nhits[evt][nroads][id];

	// SVX Data <------------------------- DA QUI
	    if (id < XFT_LYR) {
	      int zid = d_out1[i];
	      int lcl = d_out2[i];
	      int hit = d_out3[i];

	      evt_dev->evt_hit[evt][nroads][id][nhits] = hit;
	      evt_dev->evt_hitZ[evt][nroads][id][nhits] = zid;
	      evt_dev->evt_lcl[evt][nroads][id][nhits] = lcl;
	      evt_dev->evt_lclforcut[evt][nroads][id][nhits] = lcl;
	      evt_dev->evt_layerZ[evt][nroads][id] = zid;

	      if (evt_dev->evt_zid[evt][nroads] == -1) {
	        evt_dev->evt_zid[evt][nroads] = zid & gf_mask(GF_SUBZ_WIDTH);
	      } else {
	        evt_dev->evt_zid[evt][nroads] = (((zid & gf_mask(GF_SUBZ_WIDTH)) << GF_SUBZ_WIDTH)
	                                + (evt_dev->evt_zid[evt][nroads] & gf_mask(GF_SUBZ_WIDTH)));
	      }

	      nhits = ++evt_dev->evt_nhits[evt][nroads][id];

	      // Error Checking
	      if (nhits == MAX_HIT) evt_dev->evt_err[evt][nroads] |= (1 << OFLOW_HIT_BIT);
	      if (id < id_last) evt_dev->evt_err[evt][nroads] |= (1 << OUTORDER_BIT);

	    } else if (id == XFT_LYR && gf_xft == 0) {
	      // we ignore - stp
	    } else if (id == XFT_LYR && gf_xft == 1) {

	      int crv = d_out1[i];
	      int crv_sign = d_out2[i];
	      int phi = d_out3[i];

	      evt_dev->evt_crv[evt][nroads][nhits] = crv;
	      evt_dev->evt_crv_sign[evt][nroads][nhits] = crv_sign;
	      evt_dev->evt_phi[evt][nroads][nhits] = phi;

	      nhits = ++evt_dev->evt_nhits[evt][nroads][id];

	      // Error Checking
	      if (nhits == MAX_HIT) evt_dev->evt_err[evt][nroads] |= (1 << OFLOW_HIT_BIT);
	      if (id < id_last) evt_dev->evt_err[evt][nroads] |= (1 << OUTORDER_BIT);

	    } else if (id == EP_LYR) {

	      int sector = d_out1[i];
	      int amroad = d_out2[i];

	      evt_dev->evt_cable_sect[evt][nroads] = sector;
	      evt_dev->evt_sect[evt][nroads] = sector;
	      evt_dev->evt_road[evt][nroads] = amroad;
	      evt_dev->evt_err_sum[evt] |= evt_dev->evt_err[evt][nroads];

	      nroads = ++evt_dev->evt_nroads[evt];

	      for (id = 0; id <= XFT_LYR; id++)
	        evt_dev->evt_nhits[evt][nroads][id] = 0;

	      evt_dev->evt_err[evt][nroads] = 0;
	      evt_dev->evt_zid[evt][nroads] = -1;

	      id = -1; id_last = -1;

	    } else if (id == EE_LYR) {

	      evt_dev->evt_ee_word[evt] = d_out[i];

	      atomicAdd(&tEvts, 1);

	      id = -1; id_last = -1;

	    } else {
	      evt_dev->evt_err[evt][nroads] |= (1 << INV_DATA_BIT);
	    }

	    id_last = id;

}*/

// Unpacking evts data and return number of events
unsigned int gf_unpack_cuda_GPU(int *ids, int *out1, int *out2, int *out3, int n_words, struct evt_arrays *evt_dev, int* d_tEvts, struct evt_arrays *evta, unsigned int *start_word, cudaStream_t stream, cudaEvent_t event ) {


  MY_CUDA_CHECK(cudaStreamSynchronize(stream));
  /*MY_CUDA_CHECK(cudaStreamWaitEvent(stream, event, 0));
	  cudaMemcpyAsync(evt_dev, evta, sizeof(struct evt_arrays), cudaMemcpyHostToDevice, stream);*/

  unsigned int tEvts=0;
  ///////////////// now fill evt (gf_fep_unpack)

  memset(evta->evt_nroads, 0, sizeof(evta->evt_nroads));
  memset(evta->evt_err_sum, 0, sizeof(evta->evt_err_sum));
  memset(evta->evt_layerZ, 0, sizeof(evta->evt_layerZ));
  memset(evta->evt_nhits, 0,  sizeof(evta->evt_nhits));
  memset(evta->evt_err,  0,   sizeof(evta->evt_err));
  memset(evta->evt_zid,  0,   sizeof(evta->evt_zid));



  for (int ie = 0; ie < NEVTS; ie++) {
    evta->evt_zid[ie][evta->evt_nroads[ie]] = -1; // because we set it to 0 for GPU version
  }


  int id_last = -1;
  int evt = EVT;
  int id;

  unsigned int i = 0;

  if((start_word != NULL) ){
	  if(*start_word < n_words){
		  i = *start_word;
	  }else{
		  printf("gf_unpack_cuda_GPU ERROR: *start_word is >= than n_words; starting from zero\n");
	  }
  }//start from zero if NULL



  do {
        
    id = ids[i];

    bool gf_xft = 0;
    if (id == XFT_LYR_2) { // compatibility - stp
      id = XFT_LYR;
      gf_xft = 1;
    }

    int nroads = evta->evt_nroads[evt];
    int nhits = evta->evt_nhits[evt][nroads][id];

    // SVX Data
    if (id < XFT_LYR) {
      int zid = out1[i];
      int lcl = out2[i];
      int hit = out3[i];

      evta->evt_hit[evt][nroads][id][nhits] = hit;
      evta->evt_hitZ[evt][nroads][id][nhits] = zid;
      evta->evt_lcl[evt][nroads][id][nhits] = lcl;
      evta->evt_lclforcut[evt][nroads][id][nhits] = lcl;
      evta->evt_layerZ[evt][nroads][id] = zid;

      if (evta->evt_zid[evt][nroads] == -1) {
        evta->evt_zid[evt][nroads] = zid & gf_mask(GF_SUBZ_WIDTH);
      } else {
        evta->evt_zid[evt][nroads] = (((zid & gf_mask(GF_SUBZ_WIDTH)) << GF_SUBZ_WIDTH)
                                + (evta->evt_zid[evt][nroads] & gf_mask(GF_SUBZ_WIDTH)));
      }

      nhits = ++evta->evt_nhits[evt][nroads][id];

      // Error Checking
      if (nhits == MAX_HIT) evta->evt_err[evt][nroads] |= (1 << OFLOW_HIT_BIT);
      if (id < id_last) evta->evt_err[evt][nroads] |= (1 << OUTORDER_BIT);
    } else if (id == XFT_LYR && gf_xft == 0) {
      // we ignore - stp
    } else if (id == XFT_LYR && gf_xft == 1) {
      int crv = out1[i];
      int crv_sign = out2[i];
      int phi = out3[i];

      evta->evt_crv[evt][nroads][nhits] = crv;
      evta->evt_crv_sign[evt][nroads][nhits] = crv_sign;
      evta->evt_phi[evt][nroads][nhits] = phi;

      nhits = ++evta->evt_nhits[evt][nroads][id];

      // Error Checking
      if (nhits == MAX_HIT) evta->evt_err[evt][nroads] |= (1 << OFLOW_HIT_BIT);
      if (id < id_last) evta->evt_err[evt][nroads] |= (1 << OUTORDER_BIT);
    } else if (id == EP_LYR) {
      int sector = out1[i];
      int amroad = out2[i];

      evta->evt_cable_sect[evt][nroads] = sector;
      evta->evt_sect[evt][nroads] = sector;
      evta->evt_road[evt][nroads] = amroad;
      evta->evt_err_sum[evt] |= evta->evt_err[evt][nroads];

      nroads = ++evta->evt_nroads[evt];

      if (nroads > MAXROAD) {
        printf("The limit on the number of roads fitted by the TF is %d\n",MAXROAD);
        printf("You reached that limit evt->nroads = %d\n",nroads);
      }

      for (id = 0; id <= XFT_LYR; id++)
        evta->evt_nhits[evt][nroads][id] = 0;

      evta->evt_err[evt][nroads] = 0;
      evta->evt_zid[evt][nroads] = -1;

      id = -1; id_last = -1;
    } else if (id == EE_LYR) {

      evta->evt_ee_word[evt] = out1[i];
      tEvts++;
      evt++;

      id = -1; id_last = -1;

    } else {
      printf("Error INV_DATA_BIT: layer = %u\n", id);
      evta->evt_err[evt][nroads] |= (1 << INV_DATA_BIT);
    }
    id_last = id;
    //increment words counter
    i++;

  }while((i < n_words) && (tEvts < NEVTS)); //end loop on input words when tEvts == NEVTS

  cudaMemcpyAsync(evt_dev, evta, sizeof(struct evt_arrays), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_tEvts, &tEvts, sizeof(int), cudaMemcpyHostToDevice, stream);

  //printf("tEvts after gf_unpack_cuda(): %d\n", tEvts);

  //returning counter of read words
  *start_word = i;

  return tEvts;

}


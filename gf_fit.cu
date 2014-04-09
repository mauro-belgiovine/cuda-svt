#include "svt_utils.h"

#define MAX_THREADS_PER_BLOCK 512
#define MIN_BLOCKS_PER_MP     4

__device__ int svtsim_whichFit_full_GPU(int layerMask, int lcMask) {

   switch (layerMask & 0x1f) {
   case 0x0f: /* 0123 */
     return 0;
   case 0x17: /* 0124 */
     return 1;
   case 0x1b: /* 0134 */
     return 2;
   case 0x1d: /* 0234 */
     return 3;
   case 0x1e: /* 1234 */
     return 4;
   case 0x1f: /* 01234 - this is the fun one to be careful with */
     if(lcMask == 0)
       return 2; /* use 0134 if we have no LC */
     else if (lcMask == 0x1)
       return 4;
     else if (lcMask == 0x2)
       return 3;
     else if (lcMask == 0x3)
       return 3;
     else if (lcMask == 0x4)
       return 2;
     else if (lcMask == 0x5)
       return 2;
     else if (lcMask == 0x6)
       return 2;
     else if (lcMask == 0x7)
       return 2;
     else if (lcMask == 0x8)
       return 1;
     else if (lcMask == 0x9)
       return 1;
     else if (lcMask == 0xa)
       return 1;
     else if (lcMask == 0xb)
       return 1;
     else if (lcMask == 0xc)
       return 2;
     else if (lcMask == 0xd)
       return 2;
     else if (lcMask == 0xe)
       return 2;
     else if (lcMask == 0xf)
       return 2;
     else  /* If we have LC on outer layer just use 0123 */
       return 0;
   default:
     return 0;

  }
}

__device__ int svtsim_whichFit_GPU(struct extra_data* edata_dev, int zin, int layerMask, int lcMask) {

   int which0 = 0, which = 0;
   if (zin<0 || zin>=SVTSIM_NBAR) zin = 0;
   which0 = svtsim_whichFit_full_GPU(layerMask, lcMask);
   which = edata_dev->whichFit[zin][which0];

   return which;
}


__device__ int  svtsim_get_gfMkAddr_GPU(struct extra_data* edata_dev, int *d, int nd, int d0) {

   /* 
      d0 = iaddr
      
   */
   int j;
   int md = 0x4000;
   int iz, lcl, hit;

   if (d0+nd>md) nd = md-d0;
   for (j = 0; j<nd; j++) {
     int i = j+d0;
     int word = 0xffff, intcp = 0, coeff = 0;
     int which;

     iz = i&7, lcl = i>>3 & 0x1f, hit = i>>8 & 0x3f;

     which = svtsim_whichFit_GPU(edata_dev, iz, hit, lcl);
     coeff = iz + which*6;  /* poor choice for illegal iz=6,7, but compatible */
     intcp = which;

     word = coeff<<3 | intcp;
     d[j] = word;
   }
   return nd;
}

__device__  int gf_mkaddr_GPU(struct extra_data* edata_dev, int hitmap, int lclmap, int zmap,
                          int *coe_addr, int *int_addr, int *addr, int *err) {

  int iaddr;
  unsigned int datum = 0;

  if ((hitmap<0) || (hitmap > gf_mask_GPU( NSVX_PLANE + 1 )) || /* + XFT_LYR */
       (lclmap<0) || (lclmap > gf_mask_GPU( NSVX_PLANE )) ||
       (zmap<0)   || (zmap   > gf_mask_GPU( GF_ZID_WIDTH )))
    *err |= ( 1 << SVTSIM_GF_MKADDR_INVALID );

  iaddr = ((zmap & gf_mask_GPU(GF_SUBZ_WIDTH)) + (lclmap<<MADDR_NCLS_LSB) + (hitmap<<MADDR_HITM_LSB));
#define MAXMKA 8192
  if ((iaddr < 0) || (iaddr >= MAXMKA)) return SVTSIM_GF_ERR;

  int ldat = 0;
  svtsim_get_gfMkAddr_GPU(edata_dev, &ldat, 1, iaddr);
  datum = ldat;
    
  *int_addr = datum & gf_mask_GPU(OFF_SUBA_WIDTH);
  *coe_addr = (datum >> OFF_SUBA_WIDTH) & gf_mask_GPU(PAR_ADDR_WIDTH);
  *addr = iaddr;

  return SVTSIM_GF_OK;

}

__device__  int gf_fit_proc_GPU(int hit[], int sign_crv, long long int coeff[], 
                            long long int intcp, long long int *result, int *err) {

  long long int temp = 0;
  int i = 0;

  *result = 0;
  *err = 0;
  for (i = 0; i < SVTNHITS; i++) {
    if (i < NSVX_PLANE) {
      temp += hit[i] * coeff[i];
    } else if (i == HIT_PHI) { /* XFT phi */
      hit[i] = (hit[i]&0x400) ? -((~hit[i]&0x3ff)+1) : (hit[i]&0x3ff);
      temp += hit[i] * coeff[i];
    } else if (i == HIT_CRV) { /* XFT curvature (curv already with sign in fep ) */
      if (sign_crv == 1) { /* if negative bit is set */
        temp -= hit[i] * coeff[i];
      } else {
        temp += hit[i] * coeff[i];
      }
    }
  }
  *result = *result + temp + intcp;
  *result = *result<0 ? -((-*result)>>17) : *result>>17;
  if (*result > 0)
    *result &= gf_mask3_GPU(FIT_DWIDTH);
  else
    *result = -(abs(*result)&gf_mask3_GPU(FIT_DWIDTH));
  return SVTSIM_GF_OK;
}


__device__ int gf_chi2_GPU(long long int chi[], int* trk_err, long long int *chi2) {

  long long int temp = 0;
  long long int chi2memdata = 0;

  *chi2 = 0;

  for (int i=0; i<NCHI; i++) {
    temp = abs(chi[i]);
    if (chi[i] < 0) temp++;

    chi2memdata = temp*temp;
    *chi2 += chi2memdata;

  }

  *chi2 = (*chi2 >> 2);

  if ((*chi2 >> 2) > gf_mask_GPU(CHI_DWIDTH)) {
    *chi2 = 0x7ff;
    *trk_err |= (1 << OFLOW_CHI_BIT);
  }

  return SVTSIM_GF_OK;

}

__device__ int gf_getq_GPU(int lyr_config) {

  int q = 0;

  switch (lyr_config) {
  case 0x01e : /* lcmap = 00000, hitmap = 11110 */
    q = 3;
    break;
  case 0x01d : /* lcmap = 00000, hitmap = 11101 */
    q = 2;
    break;
  case 0x01b : /* lcmap = 00000, hitmap = 11011 */
    q = 1;
    break;
  case 0x017 : /* lcmap = 00000, hitmap = 10111 */
    q = 2;
    break;
  case 0x00f : /* lcmap = 00000, hitmap = 01111 */
    q = 2;
    break;

  case 0x03e : /* lcmap = 00001, hitmap = 11110 */
    q = 2;
    break;
  case 0x03d : /* lcmap = 00001, hitmap = 11101 */
    q = 1;
    break;
  case 0x03b : /* lcmap = 00001, hitmap = 11011 */
    q = 1;
    break;
  case 0x037 : /* lcmap = 00001, hitmap = 10111 */
    q = 1;
    break;
  case 0x02f : /* lcmap = 00001, hitmap = 01111 */
    q = 1;
    break;

  case 0x05e : /* lcmap = 00010, hitmap = 11110 */
    q = 7;
    break;
  case 0x05d : /* lcmap = 00010, hitmap = 11101 */
    q = 1;
    break;
  case 0x05b : /* lcmap = 00010, hitmap = 11011 */
    q = 2;
    break;
  case 0x057 : /* lcmap = 00010, hitmap = 10111 */
    q = 2;
    break;
  case 0x04f : /* lcmap = 00010, hitmap = 01111 */
    q = 2;
    break;
  case 0x09e : /* lcmap = 00100, hitmap = 11110 */
    q = 7;
    break;
  case 0x09d : /* lcmap = 00100, hitmap = 11101 */
    q = 2;
    break;
  case 0x09b : /* lcmap = 00100, hitmap = 11011 */
    q = 1;
    break;
  case 0x097 : /* lcmap = 00100, hitmap = 10111 */
    q = 2;
    break;
  case 0x08f : /* lcmap = 00100, hitmap = 01111 */
    q = 3;
    break;

  case 0x11e : /* lcmap = 01000, hitmap = 11110 */
    q = 7;
    break;
  case 0x11d : /* lcmap = 01000, hitmap = 11101 */
    q = 2;
    break;
  case 0x11b : /* lcmap = 01000, hitmap = 11011 */
    q = 2;
    break;
  case 0x117 : /* lcmap = 01000, hitmap = 10111 */
    q = 1;
    break;
  case 0x10f : /* lcmap = 01000, hitmap = 01111 */
    q = 3;
    break;

  case 0x21e : /* lcmap = 10000, hitmap = 11110 */
    q = 7;
    break;
  case 0x21d : /* lcmap = 10000, hitmap = 11101 */
    q = 2;
    break;
  case 0x21b : /* lcmap = 10000, hitmap = 11011 */
    q = 2;
    break;
  case 0x217 : /* lcmap = 10000, hitmap = 10111 */
    q = 2;
    break;
  case 0x20f : /* lcmap = 10000, hitmap = 01111 */
    q = 1;
    break;

  case 0x0de : /* lcmap = 00110, hitmap = 11110 */
    q = 7;
    break;
  case 0x0dd : /* lcmap = 00110, hitmap = 11101 */
    q = 1;
    break;
  case 0x0db : /* lcmap = 00110, hitmap = 11011 */
    q = 2;
    break;
  case 0x0d7 : /* lcmap = 00110, hitmap = 10111 */
    q = 3;
    break;
  case 0x0cf : /* lcmap = 00110, hitmap = 01111 */
    q = 4;
    break;

  case 0x19e : /* lcmap = 01100, hitmap = 11110 */
    q = 7;
    break;
  case 0x19d : /* lcmap = 01100, hitmap = 11101 */
    q = 2;
    break;
  case 0x19b : /* lcmap = 01100, hitmap = 11011 */
    q = 1;
    break;
  case 0x197 : /* lcmap = 01100, hitmap = 10111 */
    q = 1;
    break;
  case 0x18f : /* lcmap = 01100, hitmap = 01111 */
    q = 3;
    break;


  case 0x31e : /* lcmap = 11000, hitmap = 11110 */
    q = 7;
    break;
  case 0x31d : /* lcmap = 11000, hitmap = 11101 */
    q = 3;
    break;
  case 0x31b : /* lcmap = 11000, hitmap = 11011 */
    q = 3;
    break;
  case 0x317 : /* lcmap = 11000, hitmap = 10111 */
    q = 1;
    break;
  case 0x30f : /* lcmap = 11000, hitmap = 01111 */
    q = 2;
    break;

  case 0x15e : /* lcmap = 01010, hitmap = 11110 */
    q = 7;
    break;
  case 0x15d : /* lcmap = 01010, hitmap = 11101 */
    q = 1;
    break;
  case 0x15b : /* lcmap = 01010, hitmap = 11011 */
    q = 3;
    q = 3;
    break;
  case 0x157 : /* lcmap = 01010, hitmap = 10111 */
    q = 2;
    break;
  case 0x14f : /* lcmap = 01010, hitmap = 01111 */
    q = 4;
    break;

  case 0x25e : /* lcmap = 10010, hitmap = 11110 */
    q = 7;
    break;
  case 0x25d : /* lcmap = 10010, hitmap = 11101 */
    q = 1;
    break;
  case 0x25b : /* lcmap = 10010, hitmap = 11011 */
    q = 2;
    break;
  case 0x257 : /* lcmap = 10010, hitmap = 10111 */
    q = 2;
    break;
  case 0x24f : /* lcmap = 10010, hitmap = 01111 */
    q = 1;
    break;

  case 0x29e : /* lcmap = 10100, hitmap = 11110 */
    q = 7;
    break;
  case 0x29d : /* lcmap = 10100, hitmap = 11101 */
    q = 2;
    break;
  case 0x29b : /* lcmap = 10100, hitmap = 11011 */
    q = 1;
    break;
  case 0x297 : /* lcmap = 10100, hitmap = 10111 */
    q = 2;
    break;
  case 0x28f : /* lcmap = 10100, hitmap = 01111 */
    q = 1;
    break;
  default:
    q = 7;
    break;
  }
  return q;
}

__device__ int gf_gfunc_GPU(int ncomb5h, int icomb5h, int hitmap, int lcmap, int chi2) {

  int lyr_config;
  int gvalue;
  int newhitmap;
  int newlcmap;
  int q = 0;

  if (ncomb5h == 1) {
    newhitmap = hitmap;
    newlcmap = lcmap;
  } else if (ncomb5h == 5) {
    switch (icomb5h) {
    case 0 :     /*  11110 */
      newhitmap = 0x1e;
      newlcmap  = (lcmap & 0x1e);
      break;
    case 1 :     /*  11101 */
      newhitmap = 0x1d;
      newlcmap  = lcmap & 0x1d;
      break;
    case 2 :     /*  11011 */
      newhitmap = 0x1b;
      newlcmap  = lcmap & 0x1b;
      break;
    case 3 :     /*  10111 */
      newhitmap = 0x17;
      newlcmap  = lcmap & 0x17;
      break;
    case 4 :     /*  01111 */
      newhitmap = 0x0f;
      newlcmap  = lcmap & 0x0f;
      break;
    }
  }
  lyr_config = newhitmap + (newlcmap << 5);
  q = gf_getq_GPU(lyr_config);
  gvalue = (q << 4) + ((chi2 & 0x3ff) >> 6);
  return gvalue;
}

__device__ int gf_stword_GPU(int id, int err) {
     /*
       Compose the GF status word in the 7th word from the GF 
       INPUT : err; error summary
       OUTPUT : return the gf_stword

       NOTE: Currently this code does not support the parity error and
             FIFO error.
     */

  int word;

  word = id;

  if ((err>>OFLOW_HIT_BIT)&gf_mask_GPU(1))
    word |= (1<<GFS_OFL_HIT);

  if ((err>>OFLOW_CHI_BIT)&gf_mask_GPU(1))
    word |= (1<<GFS_OFL_CHI);

  if (((err>>UFLOW_HIT_BIT)&gf_mask_GPU(1)) ||
       ((err>>OUTORDER_BIT)&gf_mask_GPU(1)))
    word |= (1<<GFS_INV_DATA);

  return word;

}

__device__ int cal_parity_GPU(int word) {

  int par = 0;

  for (int i=0; i<SVT_WORD_WIDTH; i++)
    par ^= ((word>>i) & gf_mask_GPU(1));

  return par;
}

__device__ int gf_formatter_err_GPU(int err, int cdfmsk, int svtmsk, int eoemsk,
                                    int *eoe, int *cdf, int *svt) {

    /*
       Simulate the board error conditions (CDF-ERR, SVT-ERR and EOE-ERR)
       INPUT: err; error summary.
       cdfmsk; Mask for the CDF-ERR.
       svtmsk; Mask for the SVT-ERR.
       eoemsk; Mask for the EOE-ERR.
       OUTPUT: *eoe; EOE error
       *cdf; CDF error
       *svt; SVT error
       */

  /* --------- Executable starts here ------------ */

  *cdf = 0; /* never turned ON except for the FIFO overflow */
  *svt = 0;
  *eoe = 0;

  for (int i=0; i<= FIT_RESULT_OFLOW_BIT; i++) {
    if ((err>>i)&gf_mask_GPU(1)) {
      if (((svtmsk>>i)&gf_mask_GPU(1)) == 0)
        *svt = 1;
  
      if (i == 0) {
        if (((eoemsk >> PARITY_ERR_BIT) & gf_mask_GPU(1)) == 0) {
          *eoe |= (1<<PARITY_ERR_BIT);
        }
      } else if ((i==2) || (i==3)) {
        if (((eoemsk>>INV_DATA_BIT)&gf_mask_GPU(1)) == 0) {
          *eoe |= (1<<INV_DATA_BIT);
        }
      } else {
        if (((eoemsk>>INT_OFLOW_BIT)&gf_mask_GPU(1)) == 0) {
          *eoe |= (1<<INT_OFLOW_BIT);
        }
      }
    } /* if ((err>>i)&gf_mask_GPU(1))  */

  } /* for (i=0; i<= FIT_RESULT_OFLOW_BIT; i++)  */

  return SVTSIM_GF_OK;


}


__device__ int gf_formatter_GPU(int ie, int ir, int ic, int ich, int chi2, 
                            struct fep_arrays* fep_dev, struct fit_arrays* fit_dev, struct evt_arrays* evt_dev,
                            struct fout_arrays* fout_dev) {

  int it, err;
  int hit_form[NSVX_PLANE];

  int z = 0; /* z should be 6 bits large */
  int gf_stat = 0;

  // atomicAdd returns the old value
  it = atomicAdd(&fout_dev->fout_ntrks[ie], 1);
  
  err = (fep_dev->fep_err[ie][ir] | fit_dev->fit_err[ie][ir][ic][ich]);

  for (int i=0; i<NSVX_PLANE; i++) {
    /* Hit coordinate */
    if (fep_dev->fep_ncomb5h[ie][ir][ic] == 5) {
      if (i != ich) {
        hit_form[i] = fep_dev->fep_hit[ie][ir][ic][i]&gf_mask_GPU(GF_HIT_WIDTH);
        /* Long Cluster bit */
        hit_form[i] += (((fep_dev->fep_hit[ie][ir][ic][i] & 0x4000) ? 1 : 0) << GF_HIT_WIDTH);
        /* Hit existence bit */
        hit_form[i] += (((fep_dev->fep_hitmap[ie][ir][ic]>>i)&gf_mask_GPU(1))<<(GF_HIT_WIDTH+1));
        hit_form[i] = (hit_form[i]&gf_mask_GPU(GF_HIT_WIDTH+2));
      } else 
        hit_form[i] = 0;
    } else {
      hit_form[i] = fep_dev->fep_hit[ie][ir][ic][i]&gf_mask_GPU(GF_HIT_WIDTH);
      /* Long Cluster bit */
      hit_form[i] += (((fep_dev->fep_hit[ie][ir][ic][i] & 0x4000) ? 1 : 0) << GF_HIT_WIDTH);
      /* Hit existence bit */
      hit_form[i] += (((fep_dev->fep_hitmap[ie][ir][ic]>>i)&gf_mask_GPU(1))<<(GF_HIT_WIDTH+1));
      hit_form[i] = (hit_form[i]&gf_mask_GPU(GF_HIT_WIDTH+2));
    }
  }

  if (1) {
    int presentmask;
    int newhitmap;

    if (fep_dev->fep_ncomb5h[ie][ir][ic] == 1) {
     presentmask = fep_dev->fep_hitmap[ie][ir][ic];
    } else if (fep_dev->fep_ncomb5h[ie][ir][ic] == 5) {
      switch (ich) {
      case 0 :     /*  11110 */
        newhitmap = 0x1e;
        break;
      case 1 :     /*  11101 */
        newhitmap = 0x1d;
        break;
      case 2 :     /*  11011 */
        newhitmap = 0x1b;
        break;
      case 3 :     /*  10111 */
        newhitmap = 0x17;
        break;
      case 4 :     /*  01111 */
        newhitmap = 0x0f;
        break;
      }
      presentmask = newhitmap;
    }
    {
      int longmask = presentmask & fep_dev->fep_lcl[ie][ir][ic];
      int goodmask = presentmask & ~longmask;
      int badmask = 0x1f & ~goodmask;
      int badmap[] = {
        0x0,    /* 00000: all layers good */
        0x5,    /* 10000: layer 0 bad */
        0x4,    /* 01000: layer 1 bad */
        0xe,    /* 11000: layers 0,1 bad  (changed from f to e) */
        0x3,    /* 00100: layer 2 bad */
        0xe,    /* 10100: layers 0,2 bad */
        0xb,    /* 01100: layers 1,2 bad */
        0xf,    /* 11100: >2 layers bad */
        0x2,    /* 00010: layer 3 bad */
        0xd,    /* 10010: layers 0,3 bad */
        0xa,    /* 01010: layers 1,3 bad */
        0xf,    /* 11010: >2 layers bad */
        0x8,    /* 00110: layers 2,3 bad */
        0xf,    /* 10110: >2 layers bad */
        0xf,    /* 01110: >2 layers bad */
        0xf,    /* 11110: >2 layers bad */
        0x1,    /* 00001: layer 4 bad */
        0xc,    /* 10001: layers 0,4 bad */
        0x8,    /* 01001: layers 1,4 bad  (oops: doc says 0x9 not 0x8) */
        0xf,    /* 11001: >2 layers bad */
        0x7,    /* 00101: layers 2,4 bad */
        0xf,    /* 10101: >2 layers bad */
        0xf,    /* 01101: >2 layers bad */
        0xf,    /* 11101: >2 layers bad */
        0x6,    /* 00011: layers 3,4 bad */
        0xf,    /* 10011: >2 layers bad */
        0xf,    /* 01011: >2 layers bad */
        0xf,    /* 11011: >2 layers bad */
        0xf,    /* 00111: >2 layers bad */
        0xf,    /* 10111: >2 layers bad */
        0xf,    /* 01111: >2 layers bad */
        0xf     /* 11111: all layers bad! */
      };
    gf_stat = badmap[badmask];
    }
  }
  gf_stat = gf_stword_GPU(gf_stat, err);

  /* output word (25 bits) (from CDFnote 5026)
    4-3-2-1-0-9-8-7-6-5-4-3-2-1-0-9-8-7-6-5-4-3-2-1-0                
  */
  /* 1st word 
    24-23-22-21- 20- 19- 18-17-16-15-14-13- 12-11-10-9-8-7-6-5-4-3-2-1-0 
    --------     1   -  z                  phi     
  */

  /* phi is already formatted by the fitter (13 bits) */
  if (fep_dev->fep_ncomb5h[ie][ir][ic] == 1) {
    z = fep_dev->fep_zid[ie][ir];
  } else if (fep_dev->fep_ncomb5h[ie][ir][ic] == 5) {
    if (ich == 0){
      z = ((fep_dev->fep_hitZ[ie][ir][ic][4]&gf_mask_GPU(GF_SUBZ_WIDTH))<<GF_SUBZ_WIDTH) + (fep_dev->fep_hitZ[ie][ir][ic][1]&gf_mask_GPU(GF_SUBZ_WIDTH));
    } else if (ich == 4){
      z = ((fep_dev->fep_hitZ[ie][ir][ic][3]&gf_mask_GPU(GF_SUBZ_WIDTH))<<GF_SUBZ_WIDTH) + (fep_dev->fep_hitZ[ie][ir][ic][0]&gf_mask_GPU(GF_SUBZ_WIDTH));
    } else {
      z = ((fep_dev->fep_hitZ[ie][ir][ic][4]&gf_mask_GPU(GF_SUBZ_WIDTH))<<GF_SUBZ_WIDTH) + (fep_dev->fep_hitZ[ie][ir][ic][0]&gf_mask_GPU(GF_SUBZ_WIDTH));
    }
  }
  fout_dev->fout_gfword[ie][it][0] = (fit_dev->fit_fit[ie][0][ir][ic][ich] & gf_mask_GPU(OPHI_WIDTH))
                                      + ((z & gf_mask_GPU(GF_ZID_WIDTH)) << OPHI_WIDTH)
                                      + (0 << OBP_ERR_BIT) // we follow the word structure in  http://www-cdf.fnal.gov/internal/upgrades/daq_trig/trigger/svt/BoardDocs/data_words/tracks_bits.html 
                                      + (1<<(OBP_ID_BIT));

  /* 2nd word 
     4-3-2-1-0-9-8   -7-6-5-4-3-2-1-0 -9   -8-7-6-5-4-3-2-1-0 
     24-23-22-21- 20- 19-  18-  17-16-15-14-13- 12-11-  10-9-8-7-6-5-4-3-2-1-0 
     ------------  rID      sign c                       d
     17mo bit di roadID -> 19
     18mo               -> 20
  */
  fout_dev->fout_gfword[ie][it][1] = fit_dev->fit_fit[ie][1][ir][ic][ich]
                                    + (fit_dev->fit_fit[ie][2][ir][ic][ich] << OCVR_LSB)
                                    + ((evt_dev->evt_road[ie][ir] & 0x60000) << 2);

  /* 3rd word 
     4-3-2-1-0-9-8-7 -6-5-4-3-2-1-0-9-8-7-6-5-4-3-2-1-0 
     --------sector   AM road id (17 LSB)
  */
  fout_dev->fout_gfword[ie][it][2] = (evt_dev->evt_road[ie][ir] & gf_mask_GPU(OAMROAD_WIDTH))
                                      + (( fep_dev->fep_cable_sect[ie][ir] & gf_mask_GPU(SVT_SECT_WIDTH)) << OSEC_LSB);

  /* 4th word 
     4-3-2-1-0-9-8-7-6-5-4-3-2-1-0 -9-8-7-6-5-4-3-2-1-0 
     ----------x1                   x0
     bit 21 = bit 19 del roadID
     hit = 8 bassi e 2 alti     
  */
  fout_dev->fout_gfword[ie][it][3] = hit_form[0] + (hit_form[1]<<OX1_LSB)
                                      + ((evt_dev->evt_road[ie][ir] & 0x80000) << 1);

  /* 5th word 
     4-3-2-1-0-9-8-7-6-5-4-3-2-1-0 -9-8-7-6-5-4-3-2-1-0 
     ----------x3                   x2
     bit 21 = road ID 20
  */

  fout_dev->fout_gfword[ie][it][4] = hit_form[2] + (hit_form[3]<<OX3_LSB)
                                      + ((evt_dev->evt_road[ie][ir] & 0x100000));

  /* 6th word 
     4-3-2-1-0-9-8-7-6-5-4-3-2-1-0 -9-8-7-6-5-4-3-2-1-0 
     ----------chisq                x4
  */

  fout_dev->fout_gfword[ie][it][5] = hit_form[4] + ((chi2 & gf_mask_GPU(CHI2SUM_WIDTH)) << OCHI2_LSB);

  /* 7th word 
     4-3-2-1 -0-9-8-7-6-5-4-3-2-1-0-9 -8-7-6-5-4-3-2-1-0 
     ------0  TrackFitter status       Track Number                
     Track Num = identificativo della traccia XFT
     phi - 3 bit meno significativi del phi della traccia XFT
  */
  fout_dev->fout_gfword[ie][it][6] = ((fep_dev->fep_phi[ie][ir][ic] >> SVT_TRKID_LSB)
                                      &gf_mask_GPU(SVT_TRKID_WIDTH))
                                      + ((gf_stat & gf_mask_GPU(GF_STAT_WIDTH))<<OSTAT_LSB)
                                      + (1<<SVT_EP_BIT);

  for (int i=0; i<NTFWORDS; i++)
    atomicXor(&fout_dev->fout_parity[ie], cal_parity_GPU(fout_dev->fout_gfword[ie][it][i]));

  return SVTSIM_GF_OK;

}

__global__ void gf_fit_format_GPU (struct fep_arrays* fep_dev, 
                                    struct fit_arrays* fit_dev, int maxEvt ) {

  int ie, ir, ic, ich;
  long long int temp = 0;

  ie = blockIdx.x; // events index
  ir = blockIdx.y; // roads index

  // combination indexes
  ic = threadIdx.x;
  ich = threadIdx.y;

  if ( ( ie < maxEvt ) && 
      ( ir < fep_dev->fep_nroads[ie] ) && 
      ( ic < fep_dev->fep_ncmb[ie][ir] ) && 
      ( ich < fep_dev->fep_ncomb5h[ie][ir][ic] ) ) {

    /* phi */
    temp = fit_dev->fit_fit[ie][0][ir][ic][ich];
    if ( temp > 0) {
      temp++;
      temp = temp >> 1;
    } else {
      temp--;
      temp = -((-temp) >> 1);
    }
    if (abs(temp) > gf_mask_GPU(OPHI_WIDTH)) {
      fit_dev->fit_err[ie][ir][ic][ich] |= (1<<FIT_RESULT_OFLOW_BIT);
    }

    temp = (temp < 0 ? -temp : temp) & gf_mask_GPU(OPHI_WIDTH);
    fit_dev->fit_fit[ie][0][ir][ic][ich] = temp;

    /* impact parameter */
    temp = fit_dev->fit_fit[ie][1][ir][ic][ich];
    if ( temp > 0) {
      temp++;
      temp = temp >> 1;
    } else {
      temp--;
      temp = -((-temp) >> 1);
    }
       /*overflow check */
    if (abs(temp) > gf_mask_GPU(OIMP_WIDTH)) {
      fit_dev->fit_err[ie][ir][ic][ich] |= (1<< FIT_RESULT_OFLOW_BIT);
    }

    temp = (temp < 0 ? -temp : temp) & gf_mask_GPU(OIMP_WIDTH);
    /* now add a bit for the sign  */
    if ( fit_dev->fit_fit[ie][1][ir][ic][ich] < 0) {
      temp += (1<<OIMP_SIGN);
    }
    fit_dev->fit_fit[ie][1][ir][ic][ich] = temp;

    /* curvature */
    temp = fit_dev->fit_fit[ie][2][ir][ic][ich];
    if (temp > 0) {
      temp++;
      temp = temp >> 1;
    } else {
      temp--;
      temp = -((-temp) >> 1);
    }
    /*overflow check */
    if (abs(temp) > gf_mask_GPU(OCVR_WIDTH)) {
      fit_dev->fit_err[ie][ir][ic][ich] |= (1<<FIT_RESULT_OFLOW_BIT);
    }
    temp = (temp < 0 ? -temp : temp) & gf_mask_GPU(OCVR_WIDTH);
    /*  now add a bit for the sign  */
    if (fit_dev->fit_fit[ie][2][ir][ic][ich] < 0) {
      temp += (1<<OCVR_SIGN);
    }
    fit_dev->fit_fit[ie][2][ir][ic][ich] = temp;

    /* chi 1,2,3 */

    /*
    for(ichi = 3; ichi < 6; ichi++) {
      temp = fit_fit[ie + ichi*NEVTS + ir*NEVTS*6 + ic*NEVTS*6*MAXROAD + ich*NEVTS*6*MAXROAD*MAXCOMB];
      fit_fit[ie][ichi][ir][ic][ich] = temp;
    }
    */

  } // end if


}


__device__ int getGlobalIdx_2D_2D()
{
  int blockId = blockIdx.x
        + blockIdx.y * gridDim.x;

  int threadId = blockId * (blockDim.x * blockDim.y)
         + (threadIdx.y * blockDim.x)
         + threadIdx.x;

  return threadId;
}


__global__ void /*__launch_bounds__( MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP )*/ kFit(struct fep_arrays* fep_dev, struct extra_data* edata_dev,
                     struct fit_arrays* fit_dev, int maxEvt) {

   int ir, ic, ip, ih;
   int hit[SVTNHITS];

   __shared__ long long int coeff[NFITTER][SVTNHITS];

   int coe_addr, int_addr; /* Address for coefficients and intercept */
   int mka_addr; /* Address for MKADDR memory */
   long long int theintcp = 0;
   int sign_crv = 0;
   int which, lwhich;
   int iz;
   int ie;

   /*__const__ int map[7][7] = {
     { 0, 1, 2, 3, -1, 4, 5 }, // 01235
     { 0, 1, 2, -1, 3, 4, 5 }, // 01245
     { 0, 1, -1, 2, 3, 4, 5 }, // 01345
     { 0, -1, 1, 2, 3, 4, 5 }, // 02345
     { -1, 0, 1, 2, 3, 4, 5 }, // 12345
     { 0, 1, 2, 3, -1, 4, 5 }, // (??)
     { 0, 1, 2, 3, -1, 4, 5 }  // (??)
   };*/

   __shared__ int map[7][7];

   //in ogni blocco, istruiamo il thread (0,0) per inizializzare la memoria condivisa
   //map è uguale per tutti i blocchi
   	if((threadIdx.y == 0) && (threadIdx.x == 0)){

			 map[0][0] = 0;
			 map[0][1] = 1;
			 map[0][2] = 2;
			 map[0][3] = 3;
			 map[0][4] = -1;
			 map[0][5] = 4;
			 map[0][6] = 5;

			 map[1][0] = 0;
			 map[1][1] = 1;
			 map[1][2] = 2;
			 map[1][3] = -1;
			 map[1][4] = 3;
			 map[1][5] = 4;
			 map[1][6] = 5;

			 map[2][0] = 0;
			 map[2][1] = 1;
			 map[2][2] = -1;
			 map[2][3] = 2;
			 map[2][4] = 3;
			 map[2][5] = 4;
			 map[2][6] = 5;

			 map[3][0] = 0;
			 map[3][1] = -1;
			 map[3][2] = 1;
			 map[3][3] = 2;
			 map[3][4] = 3;
			 map[3][5] = 4;
			 map[3][6] = 5;

			 map[4][0] = -1;
			 map[4][1] = 0;
			 map[4][2] = 1;
			 map[4][3] = 2;
			 map[4][4] = 3;
			 map[4][5] = 4;
			 map[4][6] = 5;

			 map[5][0] = 0;
			 map[5][1] = 1;
			 map[5][2] = 2;
			 map[5][3] = 3;
			 map[5][4] = -1;
			 map[5][5] = 4;
			 map[5][6] = 5;

			 map[6][0] = 0;
			 map[6][1] = 1;
			 map[6][2] = 2;
			 map[6][3] = 3;
			 map[6][4] = -1;
			 map[6][5] = 4;
			 map[6][6] = 5;
   	}
    __syncthreads();

             /*if((blockIdx.x == 0) && (blockIdx.y == 0) && (threadIdx.y == 1)){
            	 if(threadIdx.x < 7){
             	   debug_data[threadIdx.x] = map[threadIdx.y][threadIdx.x];
            	 }
             }
             if((blockIdx.x == 1) && (blockIdx.y == 1) && (threadIdx.y == 1)){
                 if(threadIdx.x < 7){
                   debug_data[threadIdx.x+7] = map[threadIdx.y][threadIdx.x];
                 }
             }*/

  ie = blockIdx.x; // event index
  ir = blockIdx.y; // road index

  ic = threadIdx.x; // combination index
  //ip = threadIdx.y; // fitter index

  fit_dev->fit_err_sum[ie] = fep_dev->fep_err_sum[ie];

  if ( ( ie < maxEvt ) && ( ir < fep_dev->fep_nroads[ie] ) /*&& ( ic < fep_dev->fep_ncmb[ie][ir] )*/ ) {

    if ( fep_dev->fep_hitmap[ie][ir][ic] != 0x1f ) { 

      gf_mkaddr_GPU(edata_dev, fep_dev->fep_hitmap[ie][ir][ic], fep_dev->fep_lcl[ie][ir][ic], fep_dev->fep_zid[ie][ir],
                  &coe_addr, &int_addr, &mka_addr, fit_dev->fit_err_sum);
    
      int_addr = (int_addr<<OFF_SUBA_LSB) + fep_dev->fep_road[ie][ir];

      iz = fep_dev->fep_zid[ie][ir]&7;
      which = coe_addr/6; 
      lwhich = which;

      which = edata_dev->whichFit[iz][which];
    
      for (ih = 0; ih < SVTNHITS; ih++) {

    	  for(ip = 0; ip < NFITTER; ip++)
    		  coeff[ip][ih] = map[lwhich][ih] < 0 ? 0 : (edata_dev->lfitparfcon[ip][map[lwhich][ih]][iz][which]);
      
          if ( ih<NSVX_PLANE ) {
        
            hit[ih] = ((fep_dev->fep_hit[ie][ir][ic][ih] << 1) + 1) & gf_mask_GPU(15);
       
          } else if (ih == HIT_PHI) {
        
            hit[ih] = fep_dev->fep_phi[ie][ir][ic];
            hit[ih] -= edata_dev->wedge[ie]*SVTSIM_XFTPHIBINS/SVTSIM_NWEDGE;
            hit[ih] = ((hit[ih] << 3) + (1 << 2)) & gf_mask_GPU(15);
  
          } else if (ih == HIT_CRV) {

            sign_crv = fep_dev->fep_crv_sign[ie][ir][ic];
            hit[ih] = ((fep_dev->fep_crv[ie][ir][ic] << 8) + (1 << 7)) & gf_mask_GPU(15);
        
          }

      } /* end for(ih = 0; ih < SVTNHITS; ih++) */

      for(ip = 0; ip < NFITTER; ip++){
    	  theintcp = edata_dev->lfitparfcon[ip][6][iz][which] << 18;
    	  gf_fit_proc_GPU(hit, sign_crv, coeff[ip], theintcp, &(fit_dev->fit_fit[ie][ip][ir][ic][0]), &(fit_dev->fit_err[ie][ir][ic][0]));
      }

    } /* end if(tf->fep_hitmap[ie][ir][ic] != 0x1f) */
  } /* enf if on indexes */
}

__global__ void /*__launch_bounds__( MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP )*/ kFit2 (struct fep_arrays* fep_dev, struct extra_data* edata_dev,
                     struct fit_arrays* fit_dev, int maxEvt) {

   int ir, ic, ip, ih, il;
   int hit[SVTNHITS];
   __shared__ long long int coeff[NFITTER][SVTNHITS];
   int coe_addr, int_addr; /* Address for coefficients and intercept */
   int mka_addr; /* Address for MKADDR memory */
   long long int theintcp = 0;
   int sign_crv = 0;
   int which, lwhich;
   int iz;
   int ie;
   int newhitmap;

   __shared__ int map[7][7];

   //in ogni blocco, istruiamo il thread (0,0) per inizializzare la memoria condivisa
   //map è uguale per tutti i blocchi
   	if((threadIdx.y == 0) && (threadIdx.x == 0)){

			 map[0][0] = 0;
			 map[0][1] = 1;
			 map[0][2] = 2;
			 map[0][3] = 3;
			 map[0][4] = -1;
			 map[0][5] = 4;
			 map[0][6] = 5;

			 map[1][0] = 0;
			 map[1][1] = 1;
			 map[1][2] = 2;
			 map[1][3] = -1;
			 map[1][4] = 3;
			 map[1][5] = 4;
			 map[1][6] = 5;

			 map[2][0] = 0;
			 map[2][1] = 1;
			 map[2][2] = -1;
			 map[2][3] = 2;
			 map[2][4] = 3;
			 map[2][5] = 4;
			 map[2][6] = 5;

			 map[3][0] = 0;
			 map[3][1] = -1;
			 map[3][2] = 1;
			 map[3][3] = 2;
			 map[3][4] = 3;
			 map[3][5] = 4;
			 map[3][6] = 5;

			 map[4][0] = -1;
			 map[4][1] = 0;
			 map[4][2] = 1;
			 map[4][3] = 2;
			 map[4][4] = 3;
			 map[4][5] = 4;
			 map[4][6] = 5;

			 map[5][0] = 0;
			 map[5][1] = 1;
			 map[5][2] = 2;
			 map[5][3] = 3;
			 map[5][4] = -1;
			 map[5][5] = 4;
			 map[5][6] = 5;

			 map[6][0] = 0;
			 map[6][1] = 1;
			 map[6][2] = 2;
			 map[6][3] = 3;
			 map[6][4] = -1;
			 map[6][5] = 4;
			 map[6][6] = 5;
   	}

   	__syncthreads();

  ie = blockIdx.x; // event index
  ir = blockIdx.y; // road index

  ic = threadIdx.x; // combination index
  //ip = threadIdx.y; // fitter index

  fit_dev->fit_err_sum[ie] = fep_dev->fep_err_sum[ie];

  if ( ( ie < maxEvt ) && ( ir < fep_dev->fep_nroads[ie] ) /*&& ( ic < fep_dev->fep_ncmb[ie][ir] )*/ ) {

    if ( fep_dev->fep_hitmap[ie][ir][ic] == 0x1f ) { /* 5/5 track transformed in 5 4/5 tracks*/

      for (ih = 0; ih < NSVX_PLANE; ih++) {
        for (il = 0; il < NSVX_PLANE; il++) { /* one call to gf_fit_proc  for each ih value */
        /* let's calculate the new hitmap */
          if (il != ih) {
            switch (ih) {
              case 0 :     /*  11110 */
                newhitmap = 0x1e;
              break;
              case 1 :     /*  11101 */
                newhitmap = 0x1d;
              break;
              case 2 :     /*  11011 */
                newhitmap = 0x1b;
              break;
              case 3 :     /*  10111 */
                newhitmap = 0x17;
              break;
              case 4 :     /*  01111 */
                newhitmap = 0x0f;
              break;
            }

            gf_mkaddr_GPU(edata_dev, newhitmap, fep_dev->fep_lcl[ie][ir][ic], fep_dev->fep_zid[ie][ir],
                            &coe_addr, &int_addr, &mka_addr, fit_dev->fit_err_sum);

            if (ih == 0){
              iz = fep_dev->fep_hitZ[ie][ir][ic][1];;
            } else {
              iz = fep_dev->fep_zid[ie][ir]&7;
            }
            which = coe_addr/6;
            lwhich = which;
            which = edata_dev->whichFit[iz][which];

            for(ip = 0; ip < NFITTER; ip++)
            	coeff[ip][il] = map[lwhich][il] < 0 ? 0 : (edata_dev->lfitparfcon[ip][map[lwhich][il]][iz][which]);

            hit[il] = ((fep_dev->fep_hit[ie][ir][ic][il] << 1) + 1) & gf_mask_GPU(15);

          } else { // il == ih
            hit[il] = 0 ;
            for(ip = 0; ip < NFITTER; ip++)
            	coeff[ip][il]= 1;
          }
        } /* end for(il = 0; il <  NSVX_PLANE; il++)  */

        hit[HIT_PHI] = fep_dev->fep_phi[ie][ir][ic];
        hit[HIT_PHI] -= edata_dev->wedge[ie]*SVTSIM_XFTPHIBINS/SVTSIM_NWEDGE;
        hit[HIT_PHI] = ((hit[HIT_PHI] << 3) + (1 << 2)) & gf_mask_GPU(15);

        for(ip = 0; ip < NFITTER; ip++)
        	coeff[ip][HIT_PHI] = map[lwhich][HIT_PHI] < 0 ? 0 : (edata_dev->lfitparfcon[ip][map[lwhich][HIT_PHI]][iz][which]);

        sign_crv = fep_dev->fep_crv_sign[ie][ir][ic];
        hit[HIT_CRV] = ((fep_dev->fep_crv[ie][ir][ic] << 8) + (1 << 7)) & gf_mask_GPU(15);

        for(ip = 0; ip < NFITTER; ip++){
        	coeff[ip][HIT_CRV] = map[lwhich][HIT_CRV] < 0 ? 0 : (edata_dev->lfitparfcon[ip][map[lwhich][HIT_CRV]][iz][which]);

        	/* INTERCEPT */
        	theintcp = edata_dev->lfitparfcon[ip][6][iz][which] << 18;

        	gf_fit_proc_GPU(hit, sign_crv, coeff[ip], theintcp, &(fit_dev->fit_fit[ie][ip][ir][ic][ih]), &(fit_dev->fit_err[ie][ir][ic][ih]));
        }
        fit_dev->fit_err_sum[ie] |= fit_dev->fit_err[ie][ir][ic][ih];

      } /* end for(ih = 0; ih < NSVX_PLANE; ih++) */
    } /* end if(tf->fep_hitmap[ie][ir][ic] != 0x1f) */
  } /* enf if on indexes */
}


__global__ void gf_comparator_GPU(struct fep_arrays* fep_dev, struct evt_arrays* evt_dev, 
                                  struct fit_arrays* fit_dev, struct fout_arrays* fout_dev, int maxEvt) {

  int ie, ir, ic;
  int ChiSqCut, gvalue, gvalue_best;

  int ich = 0;
  int ind_best = 0;
  int chi2_best = 0;

  int gvalue_cut = 0x70;
  int bestTrackFound = 0;

  long long int chi[3], chi2;

  ie = blockIdx.x;
  ir = blockIdx.y;

  ic = threadIdx.x;

  if ( ( ie < maxEvt ) &&
        ( ir < fep_dev->fep_nroads[ie] ) &&
        ( ic < fep_dev->fep_ncmb[ie][ir] )) {

    ChiSqCut = 0x40;
    gvalue_best = 0x70;

    if (fep_dev->fep_ncomb5h[ie][ir][ic] == 1) {
      for (int i=0; i<NCHI; i++)
        chi[i] = fit_dev->fit_fit[ie][i+3][ir][ic][0];
      gf_chi2_GPU(chi, &fit_dev->fit_err[ie][ir][ic][0], &chi2);

      if (chi2 <= ChiSqCut) {
        chi2 = chi2 >> 2;
        gvalue = gf_gfunc_GPU(fep_dev->fep_ncomb5h[ie][ir][ic], ich, fep_dev->fep_hitmap[ie][ir][ic], fep_dev->fep_lcl[ie][ir][ic], (chi2 & gf_mask_GPU(CHI2SUM_WIDTH)));     
        if (gvalue < gvalue_cut) 
          gf_formatter_GPU(ie, ir, ic, 0, chi2, fep_dev, fit_dev, evt_dev, fout_dev);
      }
    } else if (fep_dev->fep_ncomb5h[ie][ir][ic] == 5) {
      bestTrackFound = 0;
      gvalue_best = 999;
      ind_best = 999;
      chi2_best = 999;
      for (ich = 0; ich < fep_dev->fep_ncomb5h[ie][ir][ic]; ich++) {
        for (int i=0; i<NCHI; i++) 
          chi[i] = fit_dev->fit_fit[ie][i+3][ir][ic][ich];
        /*  calculate chisq */
        gf_chi2_GPU(chi, &fit_dev->fit_err[ie][ir][ic][ich], &chi2);
        /* check chiSq  */
        if (chi2 <= ChiSqCut) {
          chi2 = chi2 >> 2; /* FC - hack .. see matching shift in gf_chi2 */
          gvalue = gf_gfunc_GPU(fep_dev->fep_ncomb5h[ie][ir][ic], ich, fep_dev->fep_hitmap[ie][ir][ic], fep_dev->fep_lcl[ie][ir][ic], (chi2 & gf_mask_GPU(CHI2SUM_WIDTH)));
          if  ((gvalue < gvalue_cut) && (gvalue < gvalue_best)) {
            gvalue_best = gvalue;
            ind_best = ich;
            chi2_best = chi2;
            bestTrackFound = 1;
          }
        } /*  end if(chi2 <= ChiSqCut) */
      } /*  end for(ich = 0; ich < gf->fep->ncomb5h[ir][ic]; ich++) */

      if (bestTrackFound) 
        gf_formatter_GPU(ie, ir, ic, ind_best, chi2_best, fep_dev, fit_dev, evt_dev, fout_dev);

    } /* end  if(gf->fep->ncomb5h[ir][ic] == 1) */

  } /* end if on indexes */

}

__global__ void gf_compute_eeword_GPU( struct fep_arrays* fep_dev, struct fit_arrays* fit_dev, 
                                       struct fout_arrays* fout_dev, int maxEvt) {

  int   eoe_err;
  int   ie = blockIdx.x * blockDim.x + threadIdx.x;

  if ( ie < maxEvt ) {
    fout_dev->fout_err_sum[ie] = (fep_dev->fep_err_sum[ie] | fit_dev->fit_err_sum[ie]);
    gf_formatter_err_GPU(fout_dev->fout_err_sum[ie], GF_ERRMASK_CDF,
                      GF_ERRMASK_SVT, GF_ERRMASK_EOE,
                      &eoe_err, &fout_dev->fout_cdferr[ie],
                      &fout_dev->fout_svterr[ie]);

    fout_dev->fout_ee_word[ie] = (fep_dev->fep_ee_word[ie] &
                                (gf_mask_GPU(SVT_WORD_WIDTH) & ~(1<<SVT_PAR_BIT)));
    fout_dev->fout_ee_word[ie] |= (eoe_err<<SVT_ERR_LSB);
    fout_dev->fout_ee_word[ie] |= (fout_dev->fout_parity[ie]<<SVT_PAR_BIT); 
  } 
}

void gf_fit_GPU(struct fep_arrays* fep_dev, struct evt_arrays* evt_dev, struct extra_data* edata_dev,
                struct fit_arrays* fit_dev, struct fout_arrays* fout_dev, int maxEvt, cudaStream_t stream) {

  dim3 blocks(NEVTS,MAXROAD);
  extern cudaStream_t stream0, stream1;

  /*cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);*/

  //TODO: cambiare dimensioni in <NEVTS, dim3(MAXROAD, MAXCOMB)>
  /*kFit2<<<blocks, MAXCOMB, 0, stream1>>>(fep_dev, edata_dev, fit_dev, maxEvt);
  kFit<<<blocks, MAXCOMB, 0, stream0>>>(fep_dev, edata_dev, fit_dev, maxEvt);
  cudaDeviceSynchronize();*/

  kFit2<<<blocks, MAXCOMB, 0, stream>>>(fep_dev, edata_dev, fit_dev, maxEvt);
  kFit<<<blocks, MAXCOMB, 0, stream>>>(fep_dev, edata_dev, fit_dev, maxEvt);


  /*cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("elapsed time %f s\n", elapsedTime);*/

  //kFit<<<blocks, dim3(MAXCOMB,NFITTER)>>>(fep_dev, edata_dev, fit_dev, maxEvt);
  gf_fit_format_GPU<<<blocks, dim3(MAXCOMB, MAXCOMB5H), 0, stream>>>(fep_dev, fit_dev, maxEvt);
  gf_comparator_GPU<<<blocks, dim3(MAXCOMB), 0, stream>>>(fep_dev, evt_dev, fit_dev, fout_dev, maxEvt);
  gf_compute_eeword_GPU<<<(NEVTS+255)/256, 256, 0, stream>>>(fep_dev, fit_dev, fout_dev, maxEvt);

} 

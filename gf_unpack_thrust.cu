#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/scatter.h>
#include <thrust/functional.h>

#include "svt_utils.h"


typedef thrust::tuple<unsigned int, unsigned int>     DataTuple;

typedef thrust::tuple<unsigned int, unsigned int,
		      unsigned int, unsigned int>     UnpackTuple;

typedef thrust::device_vector<unsigned int>::iterator IntIterator;
typedef thrust::tuple<IntIterator, IntIterator,
		      IntIterator, IntIterator>       IteratorTuple;
typedef thrust::zip_iterator<IteratorTuple>           ZipIterator;



struct unpacker : public thrust::unary_function<DataTuple, UnpackTuple> {

  __device__ UnpackTuple operator()(DataTuple t) {
    unsigned int word = thrust::get<0>(t);
    unsigned int prev_word = thrust::get<1>(t);
    unsigned int val1 = 0, val2 = 0, val3 = 0;

    int ee, ep, lyr;

    lyr = -999; /* Any invalid numbers != 0-7 */

    /* check if this is a EP or EE word */
    ee = (word >> SVT_EE_BIT)  & gf_mask_GPU(1);
    ep = (word >> SVT_EP_BIT)  & gf_mask_GPU(1);

    int p_ee = (prev_word >> SVT_EE_BIT) & gf_mask_GPU(1);
    int p_ep = (prev_word >> SVT_EP_BIT) & gf_mask_GPU(1);

    // check if this is the second XFT word
    bool xft = !p_ee && !p_ep && ((prev_word >> SVT_LYR_LSB) & gf_mask_GPU(SVT_LYR_WIDTH)) == XFT_LYR ? 1 : 0;

    if (ee && ep) { /* End of Event word */
      val1 = word; // ee_word
      lyr = EE_LYR;
    } else if (ee) { /* only EE bit ON is error condition */
      lyr = EE_LYR; /* We have to check */
    } else if (ep) { /* End of Packet word */
      lyr = EP_LYR;
      val1 = 6; // sector
      val2 = word  & gf_mask_GPU(AMROAD_WORD_WIDTH); // amroad
    } else if (xft) { /* Second XFT word */
      val1 = (word >> SVT_CRV_LSB)  & gf_mask_GPU(SVT_CRV_WIDTH); // crv
      val2 = (word >> (SVT_CRV_LSB + SVT_CRV_WIDTH))  & gf_mask_GPU(1); // crv_sign
      val3 = word & gf_mask_GPU(SVT_PHI_WIDTH); // phi
      lyr = XFT_LYR_2;
    } else { /* SVX hits or the first XFT word */
      lyr = (word >> SVT_LYR_LSB)  & gf_mask_GPU(SVT_LYR_WIDTH);
      if (lyr == XFT_LYRID) lyr = XFT_LYR; // probably don't need - stp
      val1 = (word >> SVT_Z_LSB)  & gf_mask_GPU(SVT_Z_WIDTH); // zid
      val2 = (word >> SVT_LCLS_BIT) & gf_mask_GPU(1); // lcl
      val3 = word & gf_mask_GPU(SVT_HIT_WIDTH); // hit
    }

    return thrust::make_tuple(lyr,val1,val2,val3);
  }
};


struct isNewRoad : public thrust::unary_function<unsigned int, bool> {
  __host__ __device__ bool operator()(const unsigned int &id) {
    return id == EP_LYR;
  }
};

struct isNewHit : public thrust::unary_function<unsigned int, bool> {
  __host__ __device__ bool operator()(const unsigned int &id) {
    return id < XFT_LYR || id == XFT_LYR_2;
  }
};

struct isNewEvt : public thrust::unary_function<unsigned int, bool> {
  __host__ __device__ bool operator()(const unsigned int &id) {
    return id == EE_LYR;
  }
};

struct isEqualLayer : public thrust::binary_function<unsigned int, unsigned int, bool> {
  __host__ __device__ bool operator()(const unsigned int &a, const unsigned int &b) {
    return a == b || ((a == XFT_LYR || a == XFT_LYR_2) && (b == XFT_LYR || b == XFT_LYR_2));
  }
};

struct layerHitMultiply {
  template <typename T>
  __host__ __device__ T operator()(const T &a, const T &b) {
    return MAX(a,1) * MAX(b,1);
  }
};


struct fill_tf_gpu {
  int *totEvts; // pointer in device memory
  struct evt_arrays *tf; // pointer in device memory
  __host__ __device__ fill_tf_gpu(struct evt_arrays *_tf, int *_totEvts) : tf(_tf), totEvts(_totEvts) {
    //*totEvts = 0;
  } // constructor

  template <typename Tuple>
   __device__ void operator()(Tuple t) {
    unsigned int id      = thrust::get<0>(t);
    unsigned int id_next = thrust::get<1>(t);
    unsigned int out1    = thrust::get<2>(t);
    unsigned int out2    = thrust::get<3>(t);
    unsigned int out3    = thrust::get<4>(t);
    unsigned int evt     = thrust::get<5>(t);
    unsigned int road    = thrust::get<6>(t);
    unsigned int rhit    = thrust::get<7>(t) -1;
    unsigned int lhit    = thrust::get<8>(t) -1;

    // SVX Data
    if (id < XFT_LYR) {
      int zid = out1;
      int lcl = out2;
      int hit = out3;

      tf->evt_hit[evt][road][id][lhit] = hit;
      tf->evt_hitZ[evt][road][id][lhit] = zid;
      tf->evt_lcl[evt][road][id][lhit] = lcl;
      tf->evt_lclforcut[evt][road][id][lhit] = lcl;
      tf->evt_layerZ[evt][road][id] = zid;

      if (rhit == 0) {
        atomicOr(&tf->evt_zid[evt][road], zid & gf_mask_GPU(GF_SUBZ_WIDTH));
      } else if (id_next == XFT_LYR) {
        atomicOr(&tf->evt_zid[evt][road], (zid & gf_mask_GPU(GF_SUBZ_WIDTH)) << GF_SUBZ_WIDTH);
      }

      atomicAdd(&tf->evt_nhits[evt][road][id], 1);

      // Error Checking
      if (lhit == MAX_HIT)
        tf->evt_err[evt][road] |= (1 << OFLOW_HIT_BIT);

    } else if (id == XFT_LYR) {
      // we ignore but leave here to not trigger 'else' case - stp
    } else if (id == XFT_LYR_2) {
      id = XFT_LYR; // for XFT_LYR_2 kludge - stp
      int crv      = out1;
      int crv_sign = out2;
      int phi      = out3;

      tf->evt_crv[evt][road][lhit] = crv;
      tf->evt_crv_sign[evt][road][lhit] = crv_sign;
      tf->evt_phi[evt][road][lhit] = phi;

      atomicAdd(&tf->evt_nhits[evt][road][id], 1);

      // Error Checking
      if (lhit == MAX_HIT)
        tf->evt_err[evt][road] |= (1 << OFLOW_HIT_BIT);
    } else if (id == EP_LYR) {
      int sector = out1;
      int amroad = out2;

      tf->evt_cable_sect[evt][road] = sector;
      tf->evt_sect[evt][road] = sector;
      tf->evt_road[evt][road] = amroad;
      tf->evt_err_sum[evt] |= tf->evt_err[evt][road];

      atomicAdd(&tf->evt_nroads[evt], 1);

    } else if (id == EE_LYR) {
      int ee_word = out1;

      tf->evt_ee_word[evt] = ee_word;
      atomicAdd(totEvts, 1);
    } else {
      tf->evt_err[evt][road] |= (1 << INV_DATA_BIT);
    }
  }
};


//void gf_unpack_GPU(unsigned int *data_in, int n_words, struct evt_arrays *evt_dev, int *d_tEvts ) {
void gf_unpack_thrust_GPU(thrust::device_vector<unsigned int> d_vec, int n_words, struct evt_arrays *evt_dev, int *d_tEvts ) {
/*
  thrust::device_vector<unsigned int> d_vec(n_words+1);
  d_vec[0] = 0;
  thrust::copy(data_in, data_in + n_words, d_vec.begin()+1);
  stop_time("input copy and initialize");

  start_time();
*/  
  thrust::device_vector<unsigned int> d_idt(n_words);
  thrust::device_vector<unsigned int> d_out1t(n_words);
  thrust::device_vector<unsigned int> d_out2t(n_words);
  thrust::device_vector<unsigned int> d_out3t(n_words);

  thrust::transform(
  thrust::make_zip_iterator(thrust::make_tuple(d_vec.begin()+1, d_vec.begin())),
  thrust::make_zip_iterator(thrust::make_tuple(d_vec.end(), d_vec.end()-1)),
  thrust::make_zip_iterator(thrust::make_tuple(d_idt.begin(), d_out1t.begin(),
                 d_out2t.begin(), d_out3t.begin())),
                 unpacker());
  
  thrust::device_vector<unsigned int> d_evt(n_words);
  thrust::device_vector<unsigned int> d_road(n_words);
  thrust::device_vector<unsigned int> d_rhit(n_words);
  thrust::device_vector<unsigned int> d_lhit(n_words);

  thrust::exclusive_scan(
    thrust::make_transform_iterator(d_idt.begin(), isNewEvt()),
    thrust::make_transform_iterator(d_idt.end(),   isNewEvt()),
    d_evt.begin());
  thrust::exclusive_scan_by_key(
    d_evt.begin(), d_evt.end(), // keys
    thrust::make_transform_iterator(d_idt.begin(), isNewRoad()), // vals
    d_road.begin());
  thrust::inclusive_scan_by_key(
    d_road.begin(), d_road.end(), // keys
    thrust::make_transform_iterator(d_idt.begin(), isNewHit()), //vals
    d_rhit.begin());
  thrust::inclusive_scan_by_key(
    d_idt.begin(), d_idt.end(), // keys
    thrust::make_transform_iterator(d_idt.begin(), isNewHit()), //vals
    d_lhit.begin(),
    isEqualLayer()); // binary predicate

  // fill tf structures array
  thrust::for_each(
      thrust::make_zip_iterator(thrust::make_tuple(
        d_idt.begin(), d_idt.begin()+1, d_out1t.begin(), d_out2t.begin(), d_out3t.begin(),
        d_evt.begin(), d_road.begin(), d_rhit.begin(), d_lhit.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(
        d_idt.end(), d_idt.end()+1, d_out1t.end(), d_out2t.end(), d_out3t.end(),
        d_evt.end(), d_road.end(), d_rhit.end(), d_lhit.end())),
      fill_tf_gpu(evt_dev, d_tEvts));

}


#include <cuda.h>
#include "svtsim_functions.h"
//#include <thrust/device_vector.h>

#define MAX(x,y) ((x)>(y) ? (x):(y))

#define SET_GRID_DIM(npoints, threadsPerBlock) ceil((npoints+((threadsPerBlock)-1))/(threadsPerBlock))


#define MY_CUDA_CHECK( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s (%d).\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err), err );              \
        exit(EXIT_FAILURE);                                                  \
    } }

#define MY_CHECK_ERROR(errorMessage) {                                    \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    }

#define gf_mask_GPU(x) (gf_maskdata_GPU[(x)])
__device__ const int gf_maskdata_GPU[] = {
  0x00000000,
  0x00000001, 0x00000003, 0x00000007, 0x0000000f,
  0x0000001f, 0x0000003f, 0x0000007f, 0x000000ff,
  0x000001ff, 0x000003ff, 0x000007ff, 0x00000fff,
  0x00001fff, 0x00003fff, 0x00007fff, 0x0000ffff,
  0x0001ffff, 0x0003ffff, 0x0007ffff, 0x000fffff,
  0x001fffff, 0x003fffff, 0x007fffff, 0x00ffffff,
  0x01ffffff, 0x03ffffff, 0x07ffffff, 0x0fffffff,
  0x1fffffff, 0x3fffffff, 0x7fffffff, 0xffffffff
};

#define gf_mask3_GPU(x) (gf_maskdata3_GPU[(x)])
__device__ const long long int gf_maskdata3_GPU[] = {
  0x000000000000ULL,
  0x000000000001ULL, 0x000000000003ULL, 0x000000000007ULL, 0x00000000000fULL,
  0x00000000001fULL, 0x00000000003fULL, 0x00000000007fULL, 0x0000000000ffULL,
  0x0000000001ffULL, 0x0000000003ffULL, 0x0000000007ffULL, 0x000000000fffULL,
  0x000000001fffULL, 0x000000003fffULL, 0x000000007fffULL, 0x00000000ffffULL,
  0x00000001ffffULL, 0x00000003ffffULL, 0x00000007ffffULL, 0x0000000fffffULL,
  0x0000001fffffULL, 0x0000003fffffULL, 0x0000007fffffULL, 0x000000ffffffULL,
  0x000001ffffffULL, 0x000003ffffffULL, 0x000007ffffffULL, 0x00000fffffffULL,
  0x00001fffffffULL, 0x00003fffffffULL, 0x00007fffffffULL, 0x0000ffffffffULL,
  0x0001ffffffffULL, 0x0003ffffffffULL, 0x0007ffffffffULL, 0x000fffffffffULL,
  0x001fffffffffULL, 0x003fffffffffULL, 0x007fffffffffULL, 0x00ffffffffffULL,
  0x01ffffffffffULL, 0x03ffffffffffULL, 0x07ffffffffffULL, 0x0fffffffffffULL,
  0x1fffffffffffULL, 0x3fffffffffffULL, 0x7fffffffffffULL, 0xffffffffffffULL
};

/*
// CUDA timer macros

// global variable for verbose output
extern int TIMER;

extern cudaEvent_t c_start, c_stop;

inline void start_time() {
  if ( TIMER ) {
    cudaEventCreate(&c_start);
    cudaEventCreate(&c_stop);
    cudaEventRecord(c_start, 0);
  }
}

inline void stop_time(const char *msg) {
  if ( TIMER ) { 
    cudaEventRecord(c_stop, 0);
    cudaEventSynchronize(c_stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, c_start, c_stop);
    printf("Time to %s: %.3f ms\n", msg, elapsedTime);
  }
}

*/
struct evt_arrays {

  int evt_hit[NEVTS][MAXROAD][NSVX_PLANE][MAX_HIT];
  int evt_hitZ[NEVTS][MAXROAD][NSVX_PLANE][MAX_HIT];
  int evt_lcl[NEVTS][MAXROAD][NSVX_PLANE][MAX_HIT];
  int evt_lclforcut[NEVTS][MAXROAD][NSVX_PLANE][MAX_HIT];
  int evt_layerZ[NEVTS][MAXROAD][NSVX_PLANE];
  int evt_zid[NEVTS][MAXROAD];
  int evt_nhits[NEVTS][MAXROAD][NSVX_PLANE+1];
  int evt_err[NEVTS][MAXROAD];
  int evt_crv[NEVTS][MAXROAD][MAX_HIT];
  int evt_crv_sign[NEVTS][MAXROAD][MAX_HIT];
  int evt_phi[NEVTS][MAXROAD][MAX_HIT];
  int evt_err_sum[NEVTS];
  int evt_cable_sect[NEVTS][MAXROAD];
  int evt_sect[NEVTS][MAXROAD];
  int evt_nroads[NEVTS];
  int evt_road[NEVTS][MAXROAD];
  int evt_ee_word[NEVTS];

};


struct fep_arrays {

  int fep_ncmb[NEVTS][MAXROAD];
  int fep_hit[NEVTS][MAXROAD][MAXCOMB][NSVX_PLANE];
  int fep_phi[NEVTS][MAXROAD][MAXCOMB];
  int fep_crv[NEVTS][MAXROAD][MAXCOMB];
  int fep_lcl[NEVTS][MAXROAD][MAXCOMB];
  int fep_lclforcut[NEVTS][MAXROAD][MAXCOMB];
  int fep_hitmap[NEVTS][MAXROAD][MAXCOMB];
  int fep_zid[NEVTS][MAXROAD];
  int fep_road[NEVTS][MAXROAD];
  int fep_sect[NEVTS][MAXROAD];
  int fep_cable_sect[NEVTS][MAXROAD];
//  int fep_err[NEVTS][MAXROAD][MAXCOMB][MAXCOMB5H];
  int fep_err[NEVTS][MAXROAD];
  int fep_crv_sign[NEVTS][MAXROAD][MAXCOMB];
  int fep_ncomb5h[NEVTS][MAXROAD][MAXCOMB];
  int fep_hitZ[NEVTS][MAXROAD][MAXCOMB][NSVX_PLANE];
  int fep_nroads[NEVTS];
  int fep_ee_word[NEVTS];
  int fep_err_sum[NEVTS];

};

struct extra_data {

  int wedge[NEVTS];
  int whichFit[SVTSIM_NBAR][FITBLOCK];/* handles TF mkaddr degeneracy */
  long long int lfitparfcon[NFITPAR][DIMSPA+1][SVTSIM_NBAR][FITBLOCK]; /* full-precision coeffs, P0s as read from fcon files SA*/

};

struct fit_arrays {

  long long int fit_fit[NEVTS][6][MAXROAD][MAXCOMB][MAXCOMB5H];
  int fit_err[NEVTS][MAXROAD][MAXCOMB][MAXCOMB5H];
  int fit_err_sum[NEVTS];

};

struct fout_arrays {

  int fout_parity[NEVTS];
  int fout_ntrks[NEVTS];
//  int fout_iroad[NEVTS][MAXROAD*MAXCOMB];
//  int fout_icmb[NEVTS][MAXROAD*MAXCOMB];
  unsigned int fout_gfword[NEVTS][MAXROAD*MAXCOMB][NTFWORDS];
  int fout_cdferr[NEVTS];
  int fout_svterr[NEVTS];
  int fout_ee_word[NEVTS];
  int fout_err_sum[NEVTS];

};


extern "C" {

  int gf_init(tf_arrays_t* ptr_tf);
  int svtsim_fconread(tf_arrays_t tf);
  int gf_fep_unpack(tf_arrays_t tf, int n_words_in, void* data);
  int gf_fep_comb(tf_arrays_t tf);
  int gf_comparator(tf_arrays_t tf);
  int gf_fit(tf_arrays_t tf);
  void svtsim_cable_addwords(svtsim_cable_t *cable, unsigned int *word, int nword);
  void svtsim_cable_addword(svtsim_cable_t *cable, unsigned int word);
  void svtsim_cable_copywords(svtsim_cable_t *cable, unsigned int *word, int nword);
  svtsim_cable_t * svtsim_cable_new(void);

  unsigned int gf_unpack_cuda_GPU(int *ids, int *out1, int *out2, int *out3, int n_words, struct evt_arrays *evt_dev, int* d_tEvts, struct evt_arrays *evta, unsigned int *start_word, cudaStream_t stream, cudaEvent_t event );
  //void gf_unpack_thrust_GPU(thrust::device_vector<unsigned int> d_vec, int n_words, struct evt_arrays *evt_dev, int *d_tEvts );
  void gf_fep_GPU( evt_arrays* evt_dev, fep_arrays* fep_dev, int maxEvt , cudaStream_t stream );
  void gf_fit_GPU(struct fep_arrays* fep_dev, struct evt_arrays* evt_dev, struct extra_data* edata_dev,
                struct fit_arrays* fit_dev, struct fout_arrays* fout_dev, int maxEvt, cudaStream_t stream);

  void dump_fout(tf_arrays_t tf);
}



#include <unistd.h>
#include <sys/time.h>
#include "svt_utils.h"
#include "cycles.h"
#include <math.h>
#include <vector>
#include <sched.h>
#include "semaphore.c"
//#include <thrust/device_vector.h>

//#define DUMP_FOUT
#define DUMP_RUNINFO

#ifdef DUMP_RUNINFO
	#include <libxml/parser.h>
	#include <libxml/tree.h>
	#include <libxml/xpath.h>
	#include <libxml/xpathInternals.h>
#endif

//Utility functions definition
int countlines(char *filename);
inline void start_time();
inline float stop_time(const char *msg);
void get_mean(float *times_array, int count, float *mean, float *stdev);
void setedata_GPU(tf_arrays_t tf, struct extra_data *edata_dev, cudaStream_t stream, cudaEvent_t event);
void set_outcable(tf_arrays_t tf);
void help(char* prog);

void xmlData_create(xmlDocPtr *doc, xmlNodePtr *root_node);
int xmlData_close(xmlDocPtr doc, char *filename);
void xmlData_addEvt(xmlNodePtr root_node, xmlNodePtr evt_node);
void xmlData_addTiming(xmlDocPtr doc, char * node_name, float time_ms, unsigned int iter);

// global variables
int VERBOSE = 0;
int TIMER = 0;

#define num_devData 4

cudaDeviceProp deviceProp;
// CUDA timer macros
cudaEvent_t c_start, c_stop;

#ifdef DUMP_RUNINFO
xmlDocPtr doc = NULL;       	/* document pointer */
xmlNodePtr root_node = NULL;	/* root node pointer */
unsigned int run_counter = 0;
#endif
// SVT-GPU execution data-class

typedef struct tf_arrays_gpu *tf_gpu_t;

struct tf_arrays_gpu {

  int totEvts;

  // formatter_out
  int fout_parity[NEVTS];
  int fout_ntrks[NEVTS];
  int fout_iroad[NEVTS][MAXROAD*MAXCOMB];
  int fout_icmb[NEVTS][MAXROAD*MAXCOMB];
  unsigned int fout_gfword[NEVTS][MAXROAD*MAXCOMB][NTFWORDS];
  int fout_cdferr[NEVTS];
  int fout_svterr[NEVTS];
  int fout_ee_word[NEVTS];
  int fout_err_sum[NEVTS];
  int fout_found[NEVTS];


 //output "cable"
  svtsim_cable_t *out;

  //gf_memory
  int wedge[NEVTS];
  unsigned int *mem_mkaddr[NEVTS];
  short mem_coeff[NFITTER][SVTNHITS][MAXCOE_VSIZE];
  int mem_nintcp;
  short (*intcp)[NFITTER];
  int mem_fitsft[NFITTER][NSHIFTS];
  /* int chi2[NCHI][MAXCHI2A]; */
  int minhits; /* The minimum number of hits that we require
		  (including XFT hit)*/
  int chi2cut;
  int svt_emsk;
  int gf_emsk;
  int cdf_emsk;
  int eoe_emsk; /* MASK for the errors */

  int whichFit[SVTSIM_NBAR][FITBLOCK];/* handles TF mkaddr degeneracy */
  int ifitpar[NFITPAR][DIMSPA+1][SVTSIM_NBAR][FITBLOCK];       /* TF coefficients, P0s */
  long long int lfitpar[NFITPAR][DIMSPA+1][SVTSIM_NBAR][FITBLOCK];/* full-precision coeffs, P0s */
  long long int lfitparfcon[NFITPAR][DIMSPA+1][SVTSIM_NBAR][FITBLOCK]; /* full-precision coeffs, P0s as read from fcon files SA*/
  float gcon[NFITPAR][DIMSPA+1][SVTSIM_NBAR][FITBLOCK];     /* [fdc012][I0123PC][z][whichfit] */
  int xftphi_shiftbits[NFITPAR];/* TF bit shifts */
  int xftcrv_shiftbits[NFITPAR];
  int result_shiftbits[NFITPAR];
  int lMap[SVTSIM_NBAR][SVTSIM_NPL];/* map physical layer => cable layer */
  int oX[SVTSIM_NBAR], oY[SVTSIM_NBAR];/* origin used for fcon */
  float phiUnit, dvxUnit, crvUnit;/* units for fit parameters */
  float k0Unit, k1Unit, k2Unit;/* units for constraints */
  int dphiNumer, dphiDenom;/* dphi(wedge) = 2pi*N/D */
  int mkaddrBogusValue;



};

__global__ void init_arrays_GPU (fout_arrays* fout_dev, evt_arrays* evt_dev, int* events ) {

  int ie, ir, ip;

  *events = 0;

  //ie = blockIdx.x; // events index
  ir = threadIdx.x; // roads index
  //ip = threadIdx.y; // NSVX_PLANE+1

  for(ie = blockIdx.x * blockDim.x + threadIdx.x; ( ie < NEVTS ); ie += blockDim.x * gridDim.x){


	  // initialize evt arrays....
	  evt_dev->evt_nroads[ie] = 0;
	  evt_dev->evt_ee_word[ie] = 0;
	  evt_dev->evt_err_sum[ie] = 0;

	  evt_dev->evt_zid[ie][ir] = 0;
	  evt_dev->evt_err[ie][ir] = 0;
	  evt_dev->evt_cable_sect[ie][ir] = 0;
	  evt_dev->evt_sect[ie][ir] = 0;
	  evt_dev->evt_road[ie][ir] = 0;


	  for(ip = 0; ip < NSVX_PLANE+1; ++ip){
		  evt_dev->evt_nhits[ie][ir][ip] = 0;
	  }

	  // initialize fout arrays....
	  fout_dev->fout_ntrks[ie] = 0;
	  fout_dev->fout_parity[ie] = 0;
	  fout_dev->fout_ee_word[ie] = 0;
	  fout_dev->fout_err_sum[ie] = 0;
	  fout_dev->fout_cdferr[ie] = 0;
	  fout_dev->fout_svterr[ie] = 0;
  }

}

class gf_devData{

	public:
		int* d_tEvts; //number of events
		struct evt_arrays* evt_dev;
		struct fep_arrays *fep_dev;
		struct fit_arrays *fit_dev;
		struct fout_arrays *fout_dev;
		struct extra_data *edata_dev;
		cudaStream_t stream;
		cudaEvent_t event;
		tf_arrays_t tf;

		//allocation
		void alloc_devData();
		//destructor
		void free_devData();
		//tf_arrays_t self-init
		void tf_init();
};


void gf_devData::alloc_devData(){

	MY_CUDA_CHECK(cudaStreamCreate(&stream));
	MY_CUDA_CHECK(cudaEventCreate(&event));
	MY_CUDA_CHECK(cudaMalloc((void**)&d_tEvts, sizeof(int)));
	MY_CUDA_CHECK(cudaMalloc((void**)&evt_dev, sizeof(evt_arrays)));
	MY_CUDA_CHECK(cudaMalloc((void**)&fep_dev, sizeof(fep_arrays)));
	MY_CUDA_CHECK(cudaMalloc((void**)&fit_dev, sizeof(fit_arrays)));
	MY_CUDA_CHECK(cudaMalloc((void**)&fout_dev, sizeof(fout_arrays)));
	MY_CUDA_CHECK(cudaMalloc((void**)&edata_dev, sizeof(struct extra_data)));
	//inizializziamo tf con memoria pinned
	MY_CUDA_CHECK(cudaHostAlloc(&tf, sizeof(struct tf_arrays), cudaHostAllocPortable));
	//printf("sizeof(struct tf_arrays) = %d\n", sizeof(struct tf_arrays));
}

void gf_devData::free_devData(){
	MY_CUDA_CHECK(cudaStreamDestroy(stream));
	MY_CUDA_CHECK(cudaEventDestroy(event));
	MY_CUDA_CHECK( cudaFree(evt_dev) );
	MY_CUDA_CHECK( cudaFree(fep_dev) );
	MY_CUDA_CHECK( cudaFree(fit_dev) );
	MY_CUDA_CHECK( cudaFree(fout_dev));
	MY_CUDA_CHECK( cudaFree(d_tEvts));
	MY_CUDA_CHECK(cudaFree(edata_dev));
	MY_CUDA_CHECK(cudaFreeHost(tf));

}

void gf_devData::tf_init(){

	//memset(tf, 0, sizeof(struct tf_arrays));
	MY_CUDA_CHECK(cudaMemsetAsync(tf, 0, sizeof(struct tf_arrays), stream));
	//wait for tf-memset to finish
	MY_CUDA_CHECK(cudaStreamSynchronize(stream));

	// initialize structures
	MY_CUDA_CHECK(cudaMemsetAsync(evt_dev, 0, sizeof(evt_arrays), stream));
	MY_CUDA_CHECK(cudaMemsetAsync(fep_dev, 0, sizeof(fep_arrays), stream));
	MY_CUDA_CHECK(cudaMemsetAsync(fit_dev, 0, sizeof(fit_arrays), stream));
	MY_CUDA_CHECK(cudaMemsetAsync(fout_dev, 0, sizeof(fout_arrays), stream));
	MY_CUDA_CHECK(cudaMemsetAsync(edata_dev, 0, sizeof(extra_data), stream));
	init_arrays_GPU<<<SET_GRID_DIM(NEVTS, MAXROAD), MAXROAD, 0, stream>>>(fout_dev, evt_dev, d_tEvts);

	// --- START inizializzazione tf
	tf->out = svtsim_cable_new();
	tf->gf_emsk  = 0;
	tf->chi2cut = GF_TOTCHI2_CUTVAL;
	tf->svt_emsk = GF_ERRMASK_SVT;
	tf->cdf_emsk = GF_ERRMASK_CDF;
	tf->eoe_emsk = GF_ERRMASK_EOE; /* MASK for the errors */
	svtsim_fconread(tf); //TODO: Che fa sta funzione???
	// --- STOP inizializzazione tf
}

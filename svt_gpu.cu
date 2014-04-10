#include "svt_gpu.h"

/* --- SVT --- */

__global__ void k_word_decode(int N, unsigned int *words, int *ids, int *out1, int *out2, int *out3) {

  /*
    parallel word_decode kernel.
    each word is decoded and layer (id) and output values are set.
    we only use 3 output arrays since depending on the layer,
    we only need 3 different values. this saves allocating/copying empty arrays
    format (out1, out2, out3):
      id <  XFT_LYR: zid, lcl, hit
      id == XFT_LYR: crv, crv_sign, phi
      id == IP_LYR: sector, amroad, 0
      id == EE_LYR: ee_word
  */

  long idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= N) return;

  int word = words[idx];
  int ee, ep, lyr;

  lyr = -999; /* Any invalid numbers != 0-7 */

  out1[idx] = 0;
  out2[idx] = 0;
  out3[idx] = 0;

  if (word > gf_mask_GPU(SVT_WORD_WIDTH)) {
    ids[idx] = lyr;
    return;
  }

  /* check if this is a EP or EE word */
  ee = (word >> SVT_EE_BIT)  & gf_mask_GPU(1);
  ep = (word >> SVT_EP_BIT)  & gf_mask_GPU(1);

  int prev_word = (idx==0) ? 0 : words[idx-1];
  int p_ee = (prev_word >> SVT_EE_BIT) & gf_mask_GPU(1);
  int p_ep = (prev_word >> SVT_EP_BIT) & gf_mask_GPU(1);

  // check if this is the second XFT word
//  bool xft = ((prev_word >> SVT_LYR_LSB) & gf_mask_GPU(SVT_LYR_WIDTH)) == XFT_LYR ? 1 : 0;
  bool xft = !p_ee && !p_ep && ((prev_word >> SVT_LYR_LSB) & gf_mask_GPU(SVT_LYR_WIDTH)) == XFT_LYR ? 1 : 0;


  if (ee && ep) { /* End of Event word */
    out1[idx] = word; // ee_word
    lyr = EE_LYR;
  } else if (ee) { /* only EE bit ON is error condition */
    lyr = EE_LYR; /* We have to check */
  } else if (ep) { /* End of Packet word */
    lyr = EP_LYR;
    out1[idx] = 6; // sector
    out2[idx] = word  & gf_mask_GPU(AMROAD_WORD_WIDTH); // amroad
  } else if (xft) { /* Second XFT word */
    out1[idx] = (word >> SVT_CRV_LSB)  & gf_mask_GPU(SVT_CRV_WIDTH); // crv
    out2[idx] = (word >> (SVT_CRV_LSB + SVT_CRV_WIDTH))  & gf_mask_GPU(1); // crv_sign
    out3[idx] = word & gf_mask_GPU(SVT_PHI_WIDTH); // phi
    lyr = XFT_LYR_2;
  } else { /* SVX hits or the first XFT word */
    lyr = (word >> SVT_LYR_LSB)  & gf_mask_GPU(SVT_LYR_WIDTH);
    if (lyr == XFT_LYRID) lyr = XFT_LYR; // probably don't need - stp
    out1[idx] = (word >> SVT_Z_LSB)  & gf_mask_GPU(SVT_Z_WIDTH); // zid
    out2[idx] = (word >> SVT_LCLS_BIT) & gf_mask_GPU(1); // lcl
    out3[idx] = word & gf_mask_GPU(SVT_HIT_WIDTH); // hit
  }

  ids[idx] = lyr;
}


void svt_GPU(unsigned int *data_in, int n_words, float *timer, char *fileOut, int nothrust) {

  int tEvts=0;
  cudaGetDeviceProperties(&deviceProp, 0);

  // --- start INIT STRUCTURES ---
  if ( TIMER ) start_time();
  //struct evt_arrays* evta = (evt_arrays*)malloc(sizeof(struct evt_arrays));
  struct evt_arrays* evta;
  MY_CUDA_CHECK(cudaMallocHost(&evta, sizeof(struct evt_arrays)));

  //counting number of devData/stream it is possible to allocate on device
  /*size_t devDataMemory = sizeof(int) + sizeof(evt_arrays) + sizeof(fep_arrays) + sizeof(fit_arrays) + sizeof(fout_arrays);
  unsigned int num_devData = deviceProp.totalGlobalMem / devDataMemory;
  printf("num_devData %d = deviceProp %d / devDataMemory %d\n", num_devData, deviceProp.totalGlobalMem, devDataMemory);*/

  //create a vector to maintain all computations' device-data
  std::vector<gf_devData> devData_vector;

  for(unsigned int i = 0; i < num_devData; i++){
	  //declare new data object and allocate memory
	  gf_devData this_data;
	  this_data.alloc_devData();
	  //init tf and others structure for current computation
	  this_data.tf_init();
	  devData_vector.push_back(this_data);
  }
  if ( TIMER ) timer[0] = stop_time("struct tf initialize (once)");
  // --- stop INIT STRUCTURES ---
  

  // --- start COPY TOTAL INPUT HtoD ---
  if ( TIMER ) start_time();
  unsigned int *d_data_in;
  long sizeW = sizeof(int) * n_words;
  cudaMalloc((void **)&d_data_in, sizeW);
  cudaMemcpy(d_data_in, data_in, sizeW, cudaMemcpyHostToDevice);
  if ( TIMER ) timer[1] = stop_time("total input copy");
  // --- stop COPY TOTAL INPUT HtoD ---


  // --- start TOTAL INPUT DECODE ---
  if ( TIMER ) start_time();
  int N_THREADS_PER_BLOCK = 32;
  int *ids, *out1, *out2, *out3;
  int *d_ids, *d_out1, *d_out2, *d_out3;
  // unsigned int *d_data_in;

  ids  = (int *)malloc(sizeW);
  out1 = (int *)malloc(sizeW);
  out2 = (int *)malloc(sizeW);
  out3 = (int *)malloc(sizeW);

  cudaMalloc((void **)&d_ids, sizeW);
  cudaMalloc((void **)&d_out1, sizeW);
  cudaMalloc((void **)&d_out2, sizeW);
  cudaMalloc((void **)&d_out3, sizeW);
  // cudaMalloc((void **)&d_data_in, sizeW);

  // Copy data to the Device
  // cudaMemcpy(d_data_in, data_in, sizeW, cudaMemcpyHostToDevice);

  k_word_decode <<<(n_words+N_THREADS_PER_BLOCK-1)/N_THREADS_PER_BLOCK, N_THREADS_PER_BLOCK>>>
  (n_words, d_data_in, d_ids, d_out1, d_out2, d_out3);

  cudaMemcpy(ids, d_ids, sizeW, cudaMemcpyDeviceToHost);
  cudaMemcpy(out1, d_out1, sizeW, cudaMemcpyDeviceToHost);
  cudaMemcpy(out2, d_out2, sizeW, cudaMemcpyDeviceToHost);
  cudaMemcpy(out3, d_out3, sizeW, cudaMemcpyDeviceToHost);

  // cudaFree(d_data_in);
  cudaFree(d_ids);
  cudaFree(d_out1);
  cudaFree(d_out2);
  cudaFree(d_out3);
  if ( TIMER ) timer[2] = stop_time("total input decode");
  // --- stop TOTAL INPUT DECODE ---

  unsigned int readout_words = 0;
  unsigned int iter = 0;

  // open output file
  FILE* OUTCHECK = fopen(fileOut, "w");

  float total_time = 0;

  //loop over all n_words from input file
  while(readout_words < n_words ){

	  gf_devData this_data = devData_vector.at(iter % num_devData);

	  if ( TIMER ) start_time();
	  setedata_GPU(this_data.tf, this_data.edata_dev, this_data.stream, this_data.event);
	  tEvts = gf_unpack_cuda_GPU(ids, out1, out2, out3, n_words, this_data.evt_dev, this_data.d_tEvts, evta, &readout_words, this_data.stream, this_data.event );

	  if ( TIMER ) timer[3] = stop_time("+ Copy detector configuration data and unpacking NEVTS evts");

	  //printf("iter %d readout_words %d\n", iter, readout_words);

	  //MY_CUDA_CHECK(cudaMemcpyAsync(&tEvts, d_tEvts, sizeof(int), cudaMemcpyDeviceToHost, this_stream));
	  this_data.tf->totEvts = tEvts;

	  if ( TIMER ) start_time();
	  // Fep comb and set
	  gf_fep_GPU( this_data.evt_dev, this_data.fep_dev, tEvts, this_data.stream );
	  if ( TIMER ) timer[4] =stop_time("+ compute fep combinations");

	  if ( TIMER ) start_time();
	  // Fit and set Fout
	  gf_fit_GPU(this_data.fep_dev, this_data.evt_dev, this_data.edata_dev, this_data.fit_dev, this_data.fout_dev, tEvts, this_data.stream);
	  if ( TIMER ) timer[5] = stop_time("+ fit data");


	  // Output copy DtoH
	  if ( TIMER ) start_time();
	  MY_CUDA_CHECK(cudaMemcpyAsync(this_data.tf->fout_ntrks, this_data.fout_dev->fout_ntrks, NEVTS * sizeof(int), cudaMemcpyDeviceToHost, this_data.stream));
	  MY_CUDA_CHECK(cudaMemcpyAsync(this_data.tf->fout_ee_word, this_data.fout_dev->fout_ee_word, NEVTS * sizeof(int), cudaMemcpyDeviceToHost, this_data.stream));
	  MY_CUDA_CHECK(cudaMemcpyAsync(this_data.tf->fout_gfword, this_data.fout_dev->fout_gfword, NEVTS * MAXROAD * MAXCOMB * NTFWORDS * sizeof(unsigned int), cudaMemcpyDeviceToHost, this_data.stream));
	  if ( TIMER ) timer[6] = stop_time("+ copy output (DtoH)");


	  if(TIMER){
	  	for(unsigned int t = 3; t <= 6; t++){
	  		total_time += timer[t];
	  	}
	  	printf("iter %d (cumulative) time %f ms\n", iter, total_time);

	  	total_time = 0;
	  }

	  if(iter != 0){	//if it is not the first iteration

		//printf("%d out data -----\n", iter-1);
	  	gf_devData previous_data = devData_vector.at(((iter-1) % num_devData));
	  	cudaStreamSynchronize(previous_data.stream);
	  	// build "cable" output structure
	  	if ( TIMER ) start_time();
	  	set_outcable(previous_data.tf);
	  	if ( TIMER ) timer[7] = stop_time("- build 'cable' out");

	  	// fill output file
	  	if ( TIMER ) start_time();
	  	for (int i=0; i< previous_data.tf->out->ndata; i++) fprintf(OUTCHECK,"%.6x\n", previous_data.tf->out->data[i]);
	  	if ( TIMER ) timer[8] = stop_time("- print output on file");
	  	#ifdef DUMP_FOUT
	  	dump_fout(previous_data.tf);
	  	#endif
	  	if ( TIMER ) start_time();
	  	previous_data.tf_init();
	  	if ( TIMER ) timer[9] = stop_time("- reset data");

		if(TIMER){
			 for(unsigned int t = 7; t <= 9; t++){
		  		total_time += timer[t];
			 }
		  	 printf("print out and reset data %d (cumulative) time %f ms\n", iter-1, total_time);
		  	 total_time = 0;
		}

	  }

	  iter++;

  }

  cudaDeviceSynchronize();
  //printf("last %d out data -----\n", iter-1);
  //last iteration output
  gf_devData last_data = devData_vector.at(((iter-1) % num_devData));
  // build "cable" output structure
  if ( TIMER ) start_time();
  set_outcable(last_data.tf);
  if ( TIMER ) timer[7] = stop_time("- build 'cable' out");
  // fill output file
  if ( TIMER ) start_time();
  for (int i=0; i< last_data.tf->out->ndata; i++) fprintf(OUTCHECK,"%.6x\n", last_data.tf->out->data[i]);
  if ( TIMER ) timer[8] = stop_time("- print output on file");

  if(TIMER){
  	for(unsigned int t = 7; t <= 8; t++){
		total_time += timer[t];
	}
	printf("print out and reset data %d (cumulative) time %f ms\n", iter-1, total_time);
  }

  // close output file
  fclose(OUTCHECK);

  printf("total iter %d \n", iter);

  for(int y = 0; y < devData_vector.size(); y++){
	  devData_vector.at(y).free_devData();
  }

  cudaFree(d_data_in);

  //NOTE: cudaFreeHost() è MOLTO LENTA ( http://www.cs.virginia.edu/~mwb7w/cuda_support/memory_management_overhead.html )
  MY_CUDA_CHECK(cudaFreeHost(evta));
  //free(evta);

  free(ids);
  free(out1);
  free(out2);
  free(out3);

}

void setedata_GPU(tf_arrays_t tf, struct extra_data *edata_dev, cudaStream_t stream, cudaEvent_t event) {

  int len;
  len = SVTSIM_NBAR * FITBLOCK * sizeof(int);
  MY_CUDA_CHECK(cudaMemcpyAsync(edata_dev->whichFit, tf->whichFit, len, cudaMemcpyHostToDevice, stream));
  len = NFITPAR * (DIMSPA+1) * SVTSIM_NBAR * FITBLOCK * sizeof(long long int);
  MY_CUDA_CHECK(cudaMemcpyAsync(edata_dev->lfitparfcon, tf->lfitparfcon, len, cudaMemcpyHostToDevice, stream));
  len = NEVTS * sizeof(int);
  MY_CUDA_CHECK(cudaMemcpyAsync(edata_dev->wedge, tf->wedge, len, cudaMemcpyHostToDevice, stream));
  //record event to make unpacking-phase start at right time
  MY_CUDA_CHECK(cudaEventRecord(event,stream));

}

void set_outcable(tf_arrays_t tf) {

  svtsim_cable_copywords(tf->out, 0, 0);
  for (int ie=0; ie < tf->totEvts; ie++) {
    // insert data in the cable structure, how to "sort" them?!?
    // data should be insert only if fout_ntrks > 0
    for (int nt=0; nt < tf->fout_ntrks[ie]; nt++)
      svtsim_cable_addwords(tf->out, tf->fout_gfword[ie][nt], NTFWORDS);
      // insert end word in the cable
    svtsim_cable_addword(tf->out, tf->fout_ee_word[ie]);
  }

}

int main(int argc, char* argv[]) {

  int c;
  char* fileIn = "hbout_w6_100evts";
  char* fileOut = "gfout.txt";
  char* where = "gpu";
  int N_LOOPS = 1;
  int PRIORITY = 0;
  int NOTHRUST = 0;

  while ( (c = getopt(argc, argv, "i:s:o:l:uvtp:h")) != -1 ) {
    switch(c) {
      case 'i': 
        fileIn = optarg;
	      break;
      case 'o':
        fileOut = optarg;
        break;
	    case 's': 
        where = optarg;
	      break;
      case 'l':
        N_LOOPS = atoi(optarg);
        break;
      case 'v':
        VERBOSE = 1;
        break;
      case 'u':
        NOTHRUST = 1;
        break;
      case 't':
        TIMER = 1;
        break;
      case 'p':
        PRIORITY = atoi(optarg);
        break;
      case 'h':
        help(argv[0]);
        return 0;
    }
  }

  if (access(fileIn, 0) == -1) {
    printf("ERROR: File %s doesn't exist.\n", fileIn);
    return 1;
  }

  // Do we want to skip the first "skip" runs from mean calculation?
  int skip = 0;
  int n_iters = N_LOOPS+skip;

  float timerange = 0;
  float ptime[N_LOOPS][10];

  struct timeval tBegin, tEnd;

  // read input file
  FILE* hbout = fopen(fileIn,"r");

  if ( hbout == NULL ) {
    printf("ERROR: Cannot open input file\n");
    exit(1);
  }

  unsigned int hexaval;
  unsigned int *data_send = (unsigned int*)malloc(countlines(fileIn)*sizeof(unsigned));

  if ( data_send == (unsigned int*) NULL ) {
    perror("malloc");
    return 2;
  }
  
  // read input data file
  char word[16];
  int k=0; // number of words read
  while (fscanf(hbout, "%s", word) != EOF) {
    hexaval = strtol(word,NULL,16);
    data_send[k] = hexaval;
    k++;
  }

  fclose(hbout);

  for(unsigned int i = 0; i < n_iters; i++){

	if ( strcmp(where,"cpu") == 0 ) { // CPU

		//TODO: CPU COMPUTATION HERE....

	}else{
		//SVT-GPU start
		svt_GPU(data_send, k, ptime[i], fileOut ,NOTHRUST);
	}
  } // end iterations

  // write file with times
  /*if ( TIMER ) {
    char fileTimes[1024];
    FILE *ft;

      float mean[6];
      float stdev[6];
      for (int t=0; t < 6; ++t) 
        get_mean(times_array[t], N_LOOPS, &mean[t], &stdev[t]);

      sprintf(fileTimes, "ListTimesGPU-Evts_%d_Loops_%d.txt", NEVTS, N_LOOPS);

      ft = fopen(fileTimes, "w");
      fprintf(ft,"# #NEvts: %d, Loops: %d, mean: %.3f ms, stdev: %.3f ms\n", NEVTS, N_LOOPS, mean[0], stdev[0]);
      fprintf(ft,"# initialize GPU: %.3f ms; copy detector configuration data: %.3f ms\n", initg, fcon);
      fprintf(ft,"# input copy and initialize        --> mean: %.3f ms, stdev: %.3f ms\n", mean[1], stdev[1]);
      fprintf(ft,"# input unpack                     --> mean: %.3f ms, stdev: %.3f ms\n", mean[2], stdev[2]);
      fprintf(ft,"# compute fep combinations         --> mean: %.3f ms, stdev: %.3f ms\n", mean[3], stdev[3]);
      fprintf(ft,"# fit data and set output          --> mean: %.3f ms, stdev: %.3f ms\n", mean[4], stdev[4]);
      fprintf(ft,"# copy output (DtoH)               --> mean: %.3f ms, stdev: %.3f ms\n", mean[5], stdev[5]);
    

      for (int j=0 ; j < (N_LOOPS); j++) {
        for (int t=0; t < 6; ++t)
          fprintf(ft,"%.3f ",times_array[t][j]);
        fprintf(ft,"\n");
      }


    fclose(ft);

    printf("All done. See %s for timing.\n", fileTimes);
  }*/

  free(data_send);

  //cudaDeviceReset();

  return 0;
}

/* --- UTILITIES --- */

void help(char* prog) {

  printf("Use %s [-i fileIn] [-o fileOut] [-s cpu || gpu] [-l #loops] [-u] [-v] [-t] [-p priority] [-h] \n\n", prog);
  printf("  -i fileIn       Input file (Default: hbout_w6_100evts).\n");
  printf("  -o fileOut      Output file (Default: gfout.txt).\n");
  printf("  -s cpu || gpu   Switch between CPU or GPU version (Default: gpu).\n");
  printf("  -l loops        Number of executions (Default: 1).\n");
  printf("  -u              Use pure cuda version for unpack (Default: use thrust version).\n");
  printf("  -v              Print verbose messages.\n");
  printf("  -t              Calculate timing.\n");
  printf("  -p priority     Set scheduling priority to <priority> and cpu affinity - you nedd to be ROOT - (Default: disable).\n");
  printf("  -h              This help.\n");

}

inline void start_time() {
    cudaEventCreate(&c_start);
    cudaEventCreate(&c_stop);
    cudaEventRecord(c_start, 0);
}

inline float stop_time(const char *msg) {
  float elapsedTime = 0;
  cudaEventRecord(c_stop, 0);
  cudaEventSynchronize(c_stop);
  cudaEventElapsedTime(&elapsedTime, c_start, c_stop);
  printf("Time to %s: %.3f ms\n", msg, elapsedTime);
  cudaEventDestroy(c_start);
  cudaEventDestroy(c_stop);
  return elapsedTime;
}

// calculate mean and stdev on an array of count floats
void get_mean(float *times_array, int count, float *mean, float *stdev) {

  int j;
  float sum = 0;
  float sumsqr = 0;

  *mean = *stdev = 0;

  for (j=0; j < count; j++) {
    sum += times_array[j];
    sumsqr += pow(times_array[j],2);
  }

  *mean = sum/(float)count;

  *stdev = sqrt(abs((sumsqr/(float)count) - pow(*mean,2)));
}

int countlines(char *filename)
{
  // count the number of lines in the file called filename
  FILE *fp = fopen(filename,"r");
  int ch=0;
  int lines=0;

  if (fp == NULL)
  return 0;

  while(!feof(fp))
  {
    ch = fgetc(fp);
    if(ch == '\n')
    {
      lines++;
    }
  }

  fclose(fp);

  return lines;
}

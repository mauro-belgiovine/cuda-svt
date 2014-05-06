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

#ifdef DUMP_RUNINFO

  char buff[32];
  xmlNodePtr run_node = xmlNewNode(NULL, BAD_CAST "run");
  sprintf(buff,"%d",run_counter); //run_counter is a global variable
  run_counter++; //increment run_counter
  xmlNewProp(run_node, BAD_CAST "n", BAD_CAST buff);
  xmlAddChild(root_node, run_node);
  if ( TIMER ){
	  xmlNodePtr init_timing_node = xmlNewNode(NULL, BAD_CAST "init_timing");
	  xmlAddChild(run_node, init_timing_node);
	  sprintf(buff,"%f ms",timer[0]);
	  xmlNewChild(init_timing_node, NULL, BAD_CAST "struct_init", BAD_CAST buff);
	  sprintf(buff,"%f ms",timer[1]);
	  xmlNewChild(init_timing_node, NULL, BAD_CAST "total_input_copy_gpu", BAD_CAST buff);
	  sprintf(buff,"%f ms",timer[2]);
	  xmlNewChild(init_timing_node, NULL, BAD_CAST "total_input_decode_gpu", BAD_CAST buff);
  }

#endif

  // open output file
  FILE* OUTCHECK = fopen(fileOut, "w");

  float total_time = 0;

  //loop over all n_words from input file
  while(readout_words < n_words ){

#ifdef DUMP_RUNINFO

	   xmlNodePtr iter_node = xmlNewNode(NULL, BAD_CAST "iter");
	   sprintf(buff,"%d",iter);
	   xmlNewProp(iter_node, BAD_CAST "n", BAD_CAST buff);
	   xmlAddChild(run_node, iter_node);
#endif
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
	  	//printf("iter %d (cumulative) time %f ms\n", iter, total_time);

	  	total_time = 0;
	  }

#ifdef DUMP_RUNINFO

	  // ---- DEBUG PURPOSE -----

	  int nhit_evt = 0;

	  if ( TIMER ) start_time();
	  struct fep_arrays *fep = (struct fep_arrays *) malloc(sizeof(fep_arrays));
	  MY_CUDA_CHECK(cudaMemcpy(fep, this_data.fep_dev, sizeof(fep_arrays), cudaMemcpyDeviceToHost));
	  struct evt_arrays *evt = (struct evt_arrays *) malloc(sizeof(evt_arrays));
	  MY_CUDA_CHECK(cudaMemcpy(evt, this_data.evt_dev, sizeof(evt_arrays), cudaMemcpyDeviceToHost));
	  cudaStreamSynchronize(this_data.stream);
	  if ( TIMER ) stop_time("[DEBUG] copy data");

	  if ( TIMER ) start_time();

	  for(int z = 0; z < NEVTS; z++){

		  xmlNodePtr evt_node = xmlNewNode(NULL, BAD_CAST "event");
		  sprintf(buff,"%d",z);
		  xmlNewProp(evt_node, BAD_CAST "n", BAD_CAST buff);

		  sprintf(buff,"%d",this_data.tf->fout_ntrks[z]);
		  xmlNewChild(evt_node, NULL, BAD_CAST "fout_ntracks", BAD_CAST buff);

	      xmlNodePtr hit_list_node = xmlNewNode(NULL, BAD_CAST "hit_list");
	      xmlAddChild(evt_node, hit_list_node);

	      for(int r = 0; r < MAXROAD; r++){

	    	  xmlNodePtr road_node = xmlNewNode(NULL, BAD_CAST "road");
	    	  sprintf(buff,"%d",r);
	    	  xmlNewProp(road_node, BAD_CAST "n", BAD_CAST buff);

	    	  if(fep->fep_ncmb[z][r] != 0){
	    		  sprintf(buff,"%d",fep->fep_ncmb[z][r]);
	    		  xmlNewChild(road_node, NULL, BAD_CAST "fep_ncomb", BAD_CAST buff);
	    	  }

	    	  for(int p = 0; p < NSVX_PLANE; p++){
	       		for(int m = 0; m < MAX_HIT; m++){
	       			if(evt->evt_hit[z][r][p][m] != 0) nhit_evt++;
	       		}
	    	  }



	    	  if(nhit_evt > 0){

	    		  xmlAddChild(hit_list_node, road_node);

	    		  //printf("evt_hit [%d, %d] \n", z, r);
	      		  for(int p = 0; p < NSVX_PLANE; p++){

	      			xmlNodePtr plane_node = xmlNewNode(NULL, BAD_CAST "plane");
	      			sprintf(buff,"%d",p);
	      			xmlNewProp(plane_node, BAD_CAST "n", BAD_CAST buff);
	      			xmlAddChild(road_node, plane_node);

	      		  	//printf("\t plane %d: ", p);
	      		  	for(int m = 0; m < MAX_HIT; m++){
	      		  		if(evt->evt_hit[z][r][p][m] != 0){
							sprintf(buff,"%x",evt->evt_hit[z][r][p][m]);
							xmlNewChild(plane_node, NULL, BAD_CAST "hit", BAD_CAST buff);
							//printf("%x ", evt->evt_hit[z][r][p][m]);
	      		  		}
	      		  	}
	      		  //printf("\n");
	      		  }
	    	  }
	    	  nhit_evt = 0;

	      }

	      xmlData_addEvt(iter_node, evt_node);

	  }
	  if ( TIMER ) stop_time("[DEBUG] read combinations and nhit");
	  // ------------------------

#endif

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
		  	 //printf("print out and reset data %d (cumulative) time %f ms\n", iter-1, total_time);
		  	 total_time = 0;
		}

/*#ifdef DUMP_RUNINFO
		if(TIMER){
			xmlData_addTiming(doc, "cable_out_cpu", timer[7], iter-1);
			xmlData_addTiming(doc, "print_fileout_cpu", timer[8], iter-1);
			xmlData_addTiming(doc, "reset_data_cpu", timer[9], iter-1);
		}
#endif*/

	  }

#ifdef DUMP_RUNINFO
	  if(TIMER){
		  xmlNodePtr iter_timing_node = xmlNewNode(NULL, BAD_CAST "timing");
		  xmlAddChild(iter_node, iter_timing_node);
		  sprintf(buff,"%f ms",timer[3]);
		  xmlNewChild(iter_timing_node, NULL, BAD_CAST "copyconf_unpack", BAD_CAST buff);
		  sprintf(buff,"%f ms",timer[4]);
		  xmlNewChild(iter_timing_node, NULL, BAD_CAST "fep", BAD_CAST buff);
		  sprintf(buff,"%f ms",timer[5]);
		  xmlNewChild(iter_timing_node, NULL, BAD_CAST "fit", BAD_CAST buff);
		  sprintf(buff,"%f ms",timer[6]);
		  xmlNewChild(iter_timing_node, NULL, BAD_CAST "copy_output", BAD_CAST buff);
	  }
#endif

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
	//printf("print out and reset data %d (cumulative) time %f ms\n", iter-1, total_time);
  }

#ifdef DUMP_RUNINFO
  if(TIMER){
	  xmlData_addTiming(doc, "cable_out_cpu", timer[7], iter-1);
	  xmlData_addTiming(doc, "print_fileout_cpu", timer[8], iter-1);
  }
#endif

  // close output file
  fclose(OUTCHECK);

  //printf("total iter %d \n", iter);

  for(int y = 0; y < devData_vector.size(); y++){
	  devData_vector.at(y).free_devData();
  }

  devData_vector.clear();

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

  //float timerange = 0;
  float ptime[N_LOOPS][10];

  //struct timeval tBegin, tEnd;

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

#ifdef DUMP_RUNINFO
  xmlData_create(&doc, &root_node); //create xml doc
#endif

  for(unsigned int i = 0; i < n_iters; i++){

	if ( strcmp(where,"cpu") == 0 ) { // CPU

		//TODO: CPU COMPUTATION HERE....

	}else{
		//SVT-GPU start
		svt_GPU(data_send, k, ptime[i], fileOut ,NOTHRUST);
	}
  } // end iterations

#ifdef DUMP_RUNINFO
  printf("save result %d \n", xmlData_close(doc, "./test.xml"));
#endif

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
  //printf("Time to %s: %.3f ms\n", msg, elapsedTime);
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

// count the number of lines in the file called filename
int countlines(char *filename)
{
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

//create the xml-file for saving execution data
void xmlData_create(xmlDocPtr *doc, xmlNodePtr *root_node){

#if defined(LIBXML_TREE_ENABLED) && defined(LIBXML_OUTPUT_ENABLED)

    LIBXML_TEST_VERSION;

    /*
     * Creates a new document, a node and set it as a root node
     */
    *doc = xmlNewDoc(BAD_CAST "1.0");
    *root_node = xmlNewNode(NULL, BAD_CAST "svt_run_data");
    xmlDocSetRootElement(*doc, *root_node);

#else
	printf("ERROR: no libxml2 tree support!\n");
#endif

}

int xmlData_close(xmlDocPtr doc, char *filename){
#if defined(LIBXML_TREE_ENABLED) && defined(LIBXML_OUTPUT_ENABLED)

	int result;
	result = xmlSaveFormatFileEnc(filename, doc, "UTF-8", 1);
	/*free the document */
	xmlFreeDoc(doc);

	/*
	 *Free the global variables that may
	 *have been allocated by the parser.
	 */
	xmlCleanupParser();

	return result;
#else
	printf("ERROR: no libxml2 tree support!\n");
#endif
}

//add event-node to xml
void xmlData_addEvt(xmlNodePtr root_node, xmlNodePtr evt_node){

#if defined(LIBXML_TREE_ENABLED) && defined(LIBXML_OUTPUT_ENABLED)
	xmlAddChild(root_node, evt_node);
#else
	printf("ERROR: no libxml2 tree support!\n");
#endif

}

void xmlData_addTiming(xmlDocPtr doc, char * node_name, float time_ms, unsigned int iter){

	char buff[32];
	xmlXPathContextPtr xpathCtx;
	xmlXPathObjectPtr xpathObj;

	/* Create xpath evaluation context */
	xpathCtx = xmlXPathNewContext(doc);
	if(xpathCtx == NULL) {
		fprintf(stderr,"Error: unable to create new XPath context\n");
	}
	/* Evaluate xpath expression */
	sprintf(buff,"//iter[@n='%d']/timing", iter);
	xpathObj = xmlXPathEvalExpression(BAD_CAST buff, xpathCtx);
	if(xpathObj == NULL) {
        fprintf(stderr,"Error: unable to evaluate xpath expression \"%s\"\n", buff);
    }

	int size = (xpathObj) ? xpathObj->nodesetval->nodeNr : 0;
	/*
     * NOTE: the nodes are processed in reverse order, i.e. reverse document
     *       order because xmlNodeSetContent can actually free up descendant
     *       of the node and such nodes may have been selected too ! Handling
     *       in reverse order ensure that descendant are accessed first, before
     *       they get removed. Mixing XPath and modifications on a tree must be
     *       done carefully !
     */
    for(int i = size - 1; i >= 0; i--) {

    	sprintf(buff,"%f ms",time_ms);
    	xmlNewChild(xpathObj->nodesetval->nodeTab[i], NULL, BAD_CAST node_name, BAD_CAST buff);


    	/*
    	 * All the elements returned by an XPath query are pointers to
    	 * elements from the tree *except* namespace nodes where the XPath
    	 * semantic is different from the implementation in libxml2 tree.
    	 * As a result when a returned node set is freed when
    	 * xmlXPathFreeObject() is called, that routine must check the
    	 * element type. But node from the returned set may have been removed
    	 * by xmlNodeSetContent() resulting in access to freed data.
    	 * This can be exercised by running
    	 *       valgrind xpath2 test3.xml '//discarded' discarded
    	 * There is 2 ways around it:
		 *   - make a copy of the pointers to the nodes from the result set
	   	 *     then call xmlXPathFreeObject() and then modify the nodes
	   	 * or
	   	 *   - remove the reference to the modified nodes from the node set
	   	 *     as they are processed, if they are not namespace nodes.
	   	 */
	   	if (xpathObj->nodesetval->nodeTab[i]->type != XML_NAMESPACE_DECL) xpathObj->nodesetval->nodeTab[i] = NULL;
	}

    /* Cleanup of XPath data */
    xmlXPathFreeObject(xpathObj);
    xmlXPathFreeContext(xpathCtx);

}




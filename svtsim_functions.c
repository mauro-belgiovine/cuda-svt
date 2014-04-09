/******
 ==== The structure of the code ====

	 gf_process
             |
	     ---- gf_fep
	             |
                     ----gf_fep_unpack
                              |
			      ---- gf_iword_decode
                         (dump) dump_gf_ievt
	                  gf_fep_comb
			 (dump) dump_fep_out
	          gf_fit
	             |
                     ----gf_mkaddr
                         gf_fit_proc                         
                         gf_fit_format
                         (dump) dump_fit

			 
                  gf_comparator
                    |  		    
		    -----gf_chi2
                         gf_gfunc
		         gf_formatter
			 (dump)dump_fout
                     

****/


#include <linux/types.h>
#include <string.h>
#include <asm/msr.h>
#include "svtsim_functions.h"

static char svtsim_err_str[10][256];
static int svtsim_n_err;

 
 /*
  * Wrap memory allocation to allow debug, etc.
  */
 
 typedef struct memDebug_s {
   struct memDebug_s *last, *next;
   const char *fileName;
   int lineNum, userBytes;
   int deadBeef;
 } memDebug_t;
 static int n_alloc = 0, t_alloc = 0, m_alloc = 0;
 static memDebug_t *chain_alloc = 0;
 #if 0
 static void *badaddr[] = {  /* to hunt down problematic mallocs */
 };
 #endif
 
 
 
 /*
  * Wrap memory allocation for SVT simulators
  */
 int svtsim_memDebug(int chainDump);
 void *svtsim_malloc1(size_t size, const char *filename, int linenum);
 
 void svtsim_free1(void *tmp, const char *filename, int linenum);
 void *svtsim_realloc1(void *inptr, size_t size, 
 		      const char *filename, int linenum);
 #define svtsim_malloc(x) (svtsim_malloc1((x), __FILE__, __LINE__))
 #define svtsim_free(x) (svtsim_free1((x), __FILE__, __LINE__))
 #define svtsim_mallocForever(x) (svtsim_malloc1((x), __FILE__, -__LINE__))
 #define svtsim_realloc(x, y) (svtsim_realloc1((x), (y), __FILE__, __LINE__))
 
 void svtsim_assert_set(char *filename, int line)
 {
   /*char buf[254];
   sprintf(buf,"SVTSIMMODULE::ERROR %s: %d\n",filename,line);
   if(strlen(buf)+strlen(svtsim_err_str)<svtsim_str_size){
     strcat(svtsim_err_str,buf);
     }*/
   if(svtsim_n_err<10) sprintf(svtsim_err_str[svtsim_n_err],"SVTSIM::ASSERT %s: %d\n",filename,line);
   svtsim_n_err++;
 }
 
 #define svtsim_assert(x) \
   do { \
   if ((x)) continue; \
   svtsim_assert_set( __FILE__, __LINE__ ); \
   } while (0)
 
 
 
 int
 svtsim_memDebug(int nChainDump)
 {
   int i = 0, t = 0, tb = 0;
   memDebug_t *p = chain_alloc;
   fprintf(stderr, "svtsim_memDebug: n_alloc=%d t_alloc=%d chain+1=%p\n", 
 	  n_alloc, t_alloc, chain_alloc+1);
   for (; p; p = p->next, i++) {
     svtsim_assert(p->deadBeef==0xdeadBeef);
     svtsim_assert(p->next==0 || p->next->last==p);
     t++;
     tb += p->userBytes;
     if (p->lineNum<0) continue; /* allocated "forever" */
     if (nChainDump<0 || i<nChainDump)
       fprintf(stderr, "  p=%p sz=%d line=%s:%d\n", 
 	      p+1, p->userBytes, p->fileName, p->lineNum);
   }
   svtsim_assert(t==n_alloc);
   svtsim_assert(tb==t_alloc);
   return t_alloc;
 }
 
 
 void *
 svtsim_malloc1(size_t size, const char *filename, int linenum)
 {
   memDebug_t *p = 0;
   p = (void *) malloc(size+sizeof(*p));
   if(p == NULL) printf("FAILED MALLOC %s %d!\n",filename,linenum);
   svtsim_assert(p!=NULL);
   p->deadBeef = 0xdeadBeef;
   p->userBytes = size;
   p->fileName = filename;
   p->lineNum = linenum;
   p->last = 0;
   p->next = chain_alloc;
   chain_alloc = p;
   if (p->next) {
     svtsim_assert(p->next->last==0);
     p->next->last = p;
   }
   memset(p+1, 0, size);
   n_alloc++;
   t_alloc += p->userBytes;
   if (t_alloc>m_alloc) m_alloc = t_alloc;
 #if 0
   for (i = 0; i<SVTSIM_NEL(badaddr); i++) 
     if (p+1==badaddr[i]) {
       svtsim_memDebug(2);
       svtsim_assert(0);
     }
 #endif
   return((void *) (p+1));
 }
 
 void 
 svtsim_free1(void *p, const char *filename, int linenum)
 {
   int nbytes = 0;
   memDebug_t *q = ((memDebug_t *)p)-1;
   if (!p) return;
   svtsim_assert(p!=(void *)0xffffffff);
   if (q->deadBeef!=0xdeadBeef) {
     fprintf(stderr, "%p->deadBeef==0x%x (%s:%d)\n", 
 	    q, q->deadBeef, filename, linenum);
     free(p);
     return;
   }
   svtsim_assert(q->deadBeef==0xdeadBeef);
   svtsim_assert(q->lineNum>=0); /* don't free "forever" mallocs */
   if (q->last) {
     svtsim_assert(q->last->next==q);
     q->last->next = q->next; 
   } else {
     svtsim_assert(chain_alloc==q);
     chain_alloc = q->next;
   }
   if (q->next) {
     svtsim_assert(q->next->last==q);
     q->next->last = q->last;
   }
   n_alloc--;
   t_alloc -= q->userBytes;
   nbytes = sizeof(*q)+q->userBytes;
   memset(q, -1, nbytes);
   free(q);
 }
 
 void *
 svtsim_realloc1(void *inptr, size_t size, const char *filename, int linenum)
 {
   memDebug_t *p = ((memDebug_t *) inptr)-1;
   if (!inptr) return svtsim_malloc1(size, filename, linenum);
   if (!size) { svtsim_free1(inptr, filename, linenum); return 0; }
   if (0) { /* debug */
     svtsim_free1(inptr, filename, linenum);
     return svtsim_malloc1(size, filename, linenum);
   }
   svtsim_assert(p->deadBeef==0xdeadBeef);
   if (p->last) {
     svtsim_assert(p->last->next==p);
   } else {
     svtsim_assert(p==chain_alloc);
   }
   if (p->next) svtsim_assert(p->next->last==p);
   t_alloc -= p->userBytes;
   p = (memDebug_t *) realloc(p, size+sizeof(*p));
   if (p->last) p->last->next = p; else chain_alloc = p;
   if (p->next) p->next->last = p;
   p->userBytes = size;
   t_alloc += p->userBytes;
   if (t_alloc>m_alloc) m_alloc = t_alloc;
   svtsim_assert(p!=0);
 #if 0
   for (i = 0; i<SVTSIM_NEL(badaddr); i++) 
     if (p+1==badaddr[i]) {
       svtsim_memDebug(2);
       svtsim_assert(0);
     }
 #endif
   return p+1;
 }
 
 
 
 
 /*
  * Default BLOB loader: read file from online directory
  */
 static int
 getBlobFromFile(const char *name, char **datap, void *extra)
 {
   int dlen = 0;
   FILE *fp = 0;
   char str[256];
   char *dlPath = (char *) extra;
   if (!dlPath) dlPath = "/cdf/onln/code/cdfprod/svt_config";
   *datap = 0;
   sprintf(str, "%s/%s", dlPath, name);
   svtsim_assert(strlen(str)<sizeof(str));
   fp = fopen(str, "r");
   if (!fp) {
     fprintf(stderr, "svtsim::getBlobFromFile: can't open %s\n", str);
     return -2;
   }
   svtsim_assert(0==fseek(fp, 0, SEEK_END));
   dlen = ftell(fp);
   svtsim_assert(0==fseek(fp, 0, SEEK_SET));
   *datap = svtsim_malloc(dlen+1);
   svtsim_assert(1==fread(*datap, dlen, 1, fp));
   (*datap)[dlen] = 0;
   fclose(fp);
   return dlen;
 }
 
 /*
  * Pointer to function to fetch a BLOB from the database; also keep
  * a pointer to some data that the fetch function may need (e.g. data path)
  */
 static int (*pGetBlob)(const char *name, char **datap, void *extra) = getBlobFromFile;
 static void *pGetBlobExtra = 0;
 
 
 typedef struct blob_s {
   char *data;
   int dlen;
   int offset;
 } blob_t;
 
 
 
 
 
 /*
  * Fetch a BLOB from the database: return length>=0 or error<0
  */
 static int
 getBlob(const char *name, char **datap)
 {
   int rc = -1;
   if (pGetBlob) rc = (*pGetBlob)(name, datap, pGetBlobExtra);
   if (1) fprintf(stderr, "getBlob(%s) => %d\n", name, rc);
   return rc;
 }
 
 
 static blob_t *
 blobopen(const char *name, const char *dummy)
 {
   blob_t *b = 0;
   b = svtsim_malloc(sizeof(*b));
   b->offset = 0;
   b->dlen = getBlob(name, &b->data);
   if (b->dlen<0) {
     svtsim_free(b);
     return 0;
   }
   return b;
 }
 
 
 static char *
 blobgets(char *s, int size, blob_t *b)
 {
   printf("in blobgets: s = %s\n", s);
   int np = 0;
   const char *p = 0;
   if (!b || !b->data || b->offset>=b->dlen) return 0;
   p = b->data+b->offset;
   while (b->offset+np<b->dlen && np<size-1) if (p[np++]=='\n') break;
   b->offset += np;
   svtsim_assert(np<size);
   strncpy(s, p, np); s[np] = 0;
   return s;
 }
 
 
 /*
  * Useful macros for parsing input files
  */
 #define Get svtsim_assert(fgets(str, sizeof(str), fp))
 #define ExpectStr(S) \
   svtsim_assert(fgets(str, sizeof(str), fp) && \
 		1==sscanf(str, "%s", s) && !strcmp(s, (S)))
 #define ExpectInt(N) \
   svtsim_assert(fgets(str, sizeof(str), fp) && \
 		1==sscanf(str, "%d", &n) && n==(N))
 #define GetReal(X) do { \
   svtsim_assert(fgets(str, sizeof(str), fp) && \
 		1==sscanf(str, "%f", &x)); (X) = x; } while (0)
 #define GetInt(X) do { \
   svtsim_assert(fgets(str, sizeof(str), fp) && \
 		1==sscanf(str, "%d", &n)); (X) = n; } while (0)
 #define GetHex(X) do { \
   svtsim_assert(fgets(str, sizeof(str), fp) && \
 		1==sscanf(str, "%x", &n)); (X) = n; } while (0)


/* 
 * Calculate the parity of the word
 */
int 
cal_parity(int word) 
{
  int i, par=0;
  for (i=0; i<SVT_WORD_WIDTH; i++) 
    par ^= ((word>>i) & gf_mask(1));
  return par;
}

svtsim_cable_t * svtsim_cable_new(void) {
  svtsim_cable_t *cable = 0;
  cable = svtsim_malloc(sizeof(*cable));
  cable->data = 0; 
  cable->ndata = 0; 
  cable->mdata = 0;
  return cable; 
}
 


 
 /*
  * Map layerMask, lcMask into fit block (0..31) for fully general case
  * in which all fit constants are available.  Note that blocks 25..30
  * are set up for some exotic cases that may never really occur.  Note
  * also that before 2002-12-15 or so, only fit block 0 was actually
  * used, and that as of 2002-12-15, no fit constants yet exist (or are
  * planned) that make use of the LC bits.
 
  * Note even further that the upgraded TF has NO room for fit constants
  * that use the LC bits, but we do use them 
  * to select the constants when we have 5/5 roads
  */
 int
 svtsim_whichFit_full(tf_arrays_t tf, int zin, int layerMask, int lcMask)
 {
   /*
    * Key (*=LC, 5=XFT):
    *   0  =>  0  1  2  3  5
    *   1  =>  0  1  2  3* 5
    *   2  =>  0  1  2* 3  5
    *   3  =>  0  1* 2  3  5
    *   4  =>  0* 1  2  3  5
    *   5  =>  0  1  2  4  5
    *   6  =>  0  1  2  4* 5
    *   7  =>  0  1  2* 4  5
    *   8  =>  0  1* 2  4  5
    *   9  =>  0* 1  2  4  5
    *  10  =>  0  1  3  4  5
    *  11  =>  0  1  3  4* 5
    *  12  =>  0  1  3* 4  5
    *  13  =>  0  1* 3  4  5
    *  14  =>  0* 1  3  4  5
    *  15  =>  0  2  3  4  5
    *  16  =>  0  2  3  4* 5
    *  17  =>  0  2  3* 4  5
    *  18  =>  0  2* 3  4  5
    *  19  =>  0* 2  3  4  5
    *  20  =>  1  2  3  4  5
    *  21  =>  1  2  3  4* 5  ?? and  1  2  3  5 ??
    *  22  =>  1  2  3* 4  5  ?? and  1  2  4  5 ??
    *  23  =>  1  2* 3  4  5  ?? and  1  3  4  5 ??
    *  24  =>  1* 2  3  4  5  ?? and  2  3  4  5 ??
    *  25  =>  ?? 0  1  2  5 ??
    *  26  =>  ?? 0  1  3  5 ??
    *  27  =>  ?? 0  1  4  5 ??
    *  28  =>  ?? 0  2  3  5 ??
    *  29  =>  ?? 0  2  4  5 ??
    *  30  =>  ?? 0  3  4  5 ??
    *  31  =>  invalid/reserved/undecided/dunno
    */
 
 
   /* Key for TF upgrade - this is all we've got!
    * Just make sure to select if it's 5/5 that we 
    * make a good choice for which 4 layers to use
 
       0   =>  0  1  2  3  5
       1   =>  0  1  2  4  5
       2   =>  0  1  3  4  5
       3   =>  0  2  3  4  5
       4   =>  1  2  3  4  5 */
 
   /*printf("in svtsim_whichFit_full\n");
   printf("zin = %.6x, layerMask = %.6x, lcmask = %.6x \n",zin,layerMask,lcMask);
   */

   int bogus = 0, l = 0, m = 0;
   int lcUsedMask = 0;
   int nsvx = 0, ngoodsvx = 0, firstlc = 4;
   if (zin<0 || zin>=SVTSIM_NBAR) zin = 0;
   for (l = 0; l<5; l++) {
     if (layerMask>>l & 1) {
       nsvx++;
       if (lcMask>>l & 1) {
 	lcUsedMask |= 1<<m;
 	if (firstlc>m) firstlc = m;
       } else {
 	ngoodsvx++;
       }
       m++;
     }
   }
 
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
     return bogus;
     
   }
 }
 
 /*
  * Map layerMask, lcMask into fit block (0..31), allowing for
  * degeneracy in TF mkaddr map
  */
 int
 svtsim_whichFit(tf_arrays_t tf, int zin, int layerMask, int lcMask)
 {
#ifdef DEBUG_SVT 
   printf("in svtsim_whichFit: zin = %d, layerMask = %x, lcMask = %x\n", zin, layerMask, lcMask);
#endif   

   int which0 = 0, which = 0;
   if (zin<0 || zin>=SVTSIM_NBAR) zin = 0;
   which0 = svtsim_whichFit_full(tf, zin, layerMask, lcMask);
   which = tf->whichFit[zin][which0];

#ifdef DEBUG_SVT 
   printf("in svtsim_whichFit: which0 = %d, which = %x\n", which0, which);
#endif   

   return which;
 }
 
 
 /*
  * Integer binary logarithm (ceil(log(fabs(ix+epsilon))/log(2)))
  */
 static int
 ilog(long long ix)
 {
   long long one = 1;
   if (ix==one<<63) return 64;
   if (ix<0) ix = -ix;
   /*
    * Oops, VxWorks compiler won't shift long long by int
    */
 #define IHTFOS(i) if (ix>>(i)&1) return (i)+1
   IHTFOS(62); IHTFOS(61); IHTFOS(60); IHTFOS(59); IHTFOS(58); IHTFOS(57); 
   IHTFOS(56); IHTFOS(55); IHTFOS(54); IHTFOS(53); IHTFOS(52); IHTFOS(51); 
   IHTFOS(50); IHTFOS(49); IHTFOS(48); IHTFOS(47); IHTFOS(46); IHTFOS(45); 
   IHTFOS(44); IHTFOS(43); IHTFOS(42); IHTFOS(41); IHTFOS(40); IHTFOS(39); 
   IHTFOS(38); IHTFOS(37); IHTFOS(36); IHTFOS(35); IHTFOS(34); IHTFOS(33);
   IHTFOS(32); IHTFOS(31); IHTFOS(30); IHTFOS(29); IHTFOS(28); IHTFOS(27);
   IHTFOS(26); IHTFOS(25); IHTFOS(24); IHTFOS(23); IHTFOS(22); IHTFOS(21);
   IHTFOS(20); IHTFOS(19); IHTFOS(18); IHTFOS(17); IHTFOS(16); IHTFOS(15);
   IHTFOS(14); IHTFOS(13); IHTFOS(12); IHTFOS(11); IHTFOS(10); IHTFOS(9);
   IHTFOS(8); IHTFOS(7); IHTFOS(6); IHTFOS(5); IHTFOS(4); IHTFOS(3);
   IHTFOS(2); IHTFOS(1); IHTFOS(0);
 #undef IHTFOS
   return 0;
 }
 
 
 
 
int svtsim_fconread(tf_arrays_t tf)
 {
   
  
 
   int i = 0, j = 0, k = 0, which = 0;
   char fconFile[SVTSIM_NBAR][100];
   char fcf[SVTSIM_NBAR][100];
   char str[100];
   
   //initialization of variables
   
   tf->mkaddrBogusValue = 1000;
 
   tf->dphiDenom = 12;
   tf->dphiNumer = WEDGE;
   
   for (i = 0; i<NEVTS; i++) {
     tf->wedge[i]=WEDGE;
   }

   for (i = 0; i<SVTSIM_NBAR; i++) {
     sprintf(str, "offln_220050_20100810_w%dz%d.fcon", WEDGE, i);    
     strncpy(fcf[i], str, sizeof(fcf[i])-1);    
     //     printf("fcf[%d] = %s\n", i,  fcf[i]);
   }
  
   /*
     Get layer map
   */
   for (i = 0; i<SVTSIM_NBAR; i++) { 
     int j = 0; 
     for (j = 0; j<SVTSIM_NPL; j++) {
       //       int defMap[] = { -1, 0, 1, 2, 3, -1, 5 };
       int defMap[] = { -1, 0, 1, 2, 3, 4, 5 };
       tf->lMap[i][j] = defMap[j]; 
#ifdef DEBUG_READFCON
       printf("layer map --> tf->lMap[%d][%d]=%d\n", i, j, tf->lMap[i][j]);
#endif
     }
   }

    /*
     * Read fit constants for all barrels
     */
    for (k = 0; k<SVTSIM_NBAR; k++) {
      int n = 0, version = 0;
      int xorg = 0, yorg = 0;
      float x = 0, ftmp = 0;
      char str[80], s[80];
      int h[DIMSPA] = { 0, 0, 0, 0, 0, 0 };
      float dx[DIMSPA] = { 0, 0, 0, 0, 0, 0 };
      /*
       * Initialize all whichFit entries to bogus values
       */
      for (i = 0; i<SVTSIM_NEL(tf->whichFit[k]); i++) tf->whichFit[k][i] = -1;
 
      /*
       * Open .fcon
       */
  
      FILE *fp = 0;
      fp = fopen(fcf[k],"r");
      //fp = blobopen(fcf[k], "r");
#ifdef DEBUG_READFCON
      printf("fconread: opening %s for input, barrel %d\n", fcf[k],k);
#endif
      if (!fp) {
        fprintf(stderr, "svtsim: error opening %s for input\n", fcf[k]);
        return -1;
      }
      
      do {
        int l = 0, layerMask = 0, bogusLayers = 0;
        char layers[7] = { 0, 0, 0, 0, 0, 0, 0 };
        if (!fgets(str, sizeof(str), fp)) break;
        if (strncmp(str, "Version ", strlen("Version "))) continue;
        if ((1!=sscanf(str, "Version %d", &version)) || 
  	  (version<3) || (version>4)) {
  	fprintf(stdout, "svtsim_fconread: %s has wrong format\n", fcf[k]);
  	fprintf(stdout, "  expect Version 3 or 4, found %s\n", str);
  	fclose(fp);
  	return -2;
        }
        ExpectStr("//NPAR"); 
        ExpectInt(3);
        ExpectStr("//Layers");
        Get; 
        str[strlen(str)-1] = 0; 
        str[SVTSIM_NEL(layers)] = 0;
        for (l = 0; l<strlen(str); l++) {
	  int ll = str[l];
	  if (ll>='0' && ll<='5') {
	    ll = ll-'0';
	  } else if (ll=='F') {
	    ll = 6;
	  } else {
	    ll = -1;
	  }
	  layers[l] = 'X';
	  if (ll>=0 && ll<SVTSIM_NPL) {
#ifdef DEBUG_READFCON
	    printf(" str = %s, l = %d, strlen(%s) = %d\n", str, l, str, strlen(str));
	    printf("ll(before assignement) = %d\n", ll);
#endif
	    ll = tf->lMap[k][ll];
	    if (ll>=0) {
	      layers[l] = '0'+ll;
	      layerMask |= 1<<ll;
	    }
	  }
#ifdef DEBUG_READFCON
	  printf("layers[%d] = %d\n", l, layers[l]);
#endif
	  if (layers[l]=='X') bogusLayers = 1;
        }
        if (bogusLayers) continue;
        which = svtsim_whichFit_full(tf, k, layerMask, 0);
        tf->whichFit[k][which] = which;
        if(0) printf("w=%d z=%d: found detLayers=%s => "
 		    "amLayers=%s mask=%x which=%d\n", 
 		    WEDGE, k, str, layers, layerMask, which);
        ExpectStr("//Zin,");
        Get; /* Zin and longhit_mask data */
        ExpectStr("//Coordinate");
        Get; svtsim_assert(DIMSPA==sscanf(str, "%f %f %f %f %f %f", 
 					 dx, dx+1, dx+2, dx+3, dx+4, dx+5));
        Get; svtsim_assert(DIMSPA==sscanf(str, "%d %d %d %d %d %d", 
 					 h, h+1, h+2, h+3, h+4, h+5));
        if (version>=4) {
 	 ExpectStr("//Origin:");
 	 Get; svtsim_assert(2==sscanf(str, "%d %d", &xorg, &yorg));
        }
        tf->oX[k] = xorg; tf->oY[k] = yorg;
        ExpectStr("//Parameters:");
        for (i = 0; i<NFITPAR; i++) {
  	float pstep = 0;
  	int fc[DIMSPA+1] = { 0, 0, 0, 0, 0, 0 };
  	int ll[NFITPAR] = { 2, 1, 0, 3, 4, 5 }; /* internal=f,d,c, ext=c,d,f */
  	int l = ll[i];
  	if (i==3) ExpectStr("//Kernel");
 	Get; 
 	svtsim_assert(8==sscanf(str, "%f %d %d %d %d %d %d %d",
 				     &pstep, fc, fc+1, fc+2, fc+3, fc+4, fc+5, fc+6));
  	switch (l) {
  	case 0:
  	  tf->phiUnit = pstep; break;
  	case 1:
  	  tf->dvxUnit = pstep; break;
  	case 2:
  	  tf->crvUnit = pstep; break;
  	case 3:
  	  tf->k0Unit = pstep; break;
  	case 4:
  	  tf->k1Unit = pstep; break;
  	case 5:
  	  tf->k2Unit = pstep; break;
  	}
  	for (j = 0; j<DIMSPA; j++) {
  #ifdef DEBUG_READFCON
  	  printf("fc[%d] = %.6x, %.d\n", j, fc[j], fc[j]);
  #endif
  	  tf->gcon[l][j][k][which] = (pstep*fc[j])/(pow(2.0,h[j])*dx[j]);
  	  tf->lfitpar[l][j][k][which] = fc[j];
  	  tf->lfitpar[l][j][k][which] <<= 30; /* lfitpar coeffs are <<30 */
  	    tf->lfitpar[l][j][k][which] = 
  	      (tf->lfitpar[l][j][k][which] >> h[j]);
  	    /* fill lfitparfcon array with constants as read from fcon file SA*/
  	    tf->lfitparfcon[l][j][k][which] = fc[j];
  #ifdef DEBUG_READFCON
  	    printf("lfitparfcon[%d][%d][%d][%d] = %.6x\n", l, j, k, which, tf->lfitparfcon[l][j][k][which]);
  #endif
  	}
  	tf->gcon[l][DIMSPA][k][which] = fc[DIMSPA]*pstep;
  	tf->lfitpar[l][DIMSPA][k][which] = fc[DIMSPA];
  	/* fill lfitparfcon array with constants as read from fcon file SA*/
  	tf->lfitparfcon[l][DIMSPA][k][which] = fc[DIMSPA];
  #ifdef DEBUG_READFCON
  	printf("fc[%d] = %.6x, %.d\n", DIMSPA, fc[DIMSPA], fc[DIMSPA]);
  #endif

  	if (i==2) { /* phi */
  	  int twopi = floor(0.5+2*M_PI/tf->phiUnit);
  	    tf->lfitpar[l][DIMSPA][k][which] += 
  	      twopi*tf->dphiNumer/tf->dphiDenom;
  	    /* fill lfitparfcon array with constants as read from fcon file SA*/   
  	       tf->lfitparfcon[l][DIMSPA][k][which] += 
  		 twopi*tf->dphiNumer/tf->dphiDenom;
  	       if (tf->lfitpar[l][DIMSPA][k][which]<0) 
  		 tf->lfitpar[l][DIMSPA][k][which] += twopi;
  	       /* fill tf->lfitparfcon array with constants as read from fcon file SA*/   
  	       if (tf->lfitparfcon[l][DIMSPA][k][which]<0) 
  		 tf->lfitparfcon[l][DIMSPA][k][which] += twopi;
  	         tf->gcon[l][DIMSPA][k][which] += 
  		   2*M_PI*tf->dphiNumer/tf->dphiDenom;
  	}
  	tf->lfitpar[l][DIMSPA][k][which] <<= 16; /* tf->lfitpar intcp is <<16 */
        }
        GetReal(ftmp); svtsim_assert(ftmp==1.0); /* chisq scale factor */       
      } while (1);
      fclose(fp);    
    }
  
     /*
      * Compute shift values
      */
     for (i = 0; i<NFITPAR; i++) {
       int shftmax_svx = -99, shftmax_phi = -99, shftmax_crv = -99;
       int ishft_svx = 0, ishft_phi = 0, ishft_crv = 0;
       for (k = 0; k<SVTSIM_NBAR; k++) {
         for (which = 0; which<SVTSIM_NEL(tf->whichFit[k]); which++) {
   	int ishft = 0;
   	if (tf->whichFit[k][which]!=which) continue;
   	for (j = 0; j<4; j++) {
   	  ishft = ilog(tf->lfitpar[i][j][k][which])-30;
   	  if (ishft>shftmax_svx) shftmax_svx = ishft;
   	}
   	ishft = ilog(tf->lfitpar[i][4][k][which])-30;
   	if (ishft>shftmax_phi) shftmax_phi = ishft;
   	ishft = ilog(tf->lfitpar[i][5][k][which])-30;
   	if (ishft>shftmax_crv) shftmax_crv = ishft;
         }
       }
       ishft_svx = 8-shftmax_svx;
       if (ishft_svx>12) ishft_svx = 12;
       ishft_phi = shftmax_phi+ishft_svx-10;
       if (ishft_phi<-3) ishft_phi = -3;
       ishft_crv = shftmax_crv+ishft_svx-10;
       if (ishft_crv<-3) ishft_crv = -3;
       tf->result_shiftbits[i] = ishft_svx;
       tf->xftphi_shiftbits[i] = ishft_phi;
       tf->xftcrv_shiftbits[i] = ishft_crv;
     }
     /*
      * Calculate TF reduced-precision coefficients
      */
     for (k = 0; k<SVTSIM_NBAR; k++) {
       for (i = 0; i<NFITPAR; i++) {
         for (j = 0; j<DIMSPA+1; j++) {
   	int ishift = 0;
   	switch (j) {
   	case 4:
   	  ishift = 30-(tf->result_shiftbits[i]-tf->xftphi_shiftbits[i]);
   	  break;
   	case 5:
   	  ishift = 30-(tf->result_shiftbits[i]-tf->xftcrv_shiftbits[i]);
   	  break;
   	case 6:
   	  ishift = 16;
   	  break;
   	default:
   	  ishift = 30-(tf->result_shiftbits[i]);
   	}
   	for (which = 0; which<SVTSIM_NEL(tf->whichFit[k]); which++) {
   	  if (tf->whichFit[k][which]!=which) continue;
   	    tf->ifitpar[i][j][k][which] = 
   	      tf->lfitpar[i][j][k][which] >> ishift;
   	}
         }
       }
       /*
        * Now propagate existent fit constants into gaps left by 
        * nonexistent fit constants
        */
       for (i = 0; i<=0x3f; i++) {
         int which0 = -1;
         which = svtsim_whichFit_full(tf, k, i, 0);
         /*
          * If layer combination has no entry, use a bogus entry
          */
         if (tf->whichFit[k][which]<0) 
	   tf->whichFit[k][which] = tf->mkaddrBogusValue;
         which0 = tf->whichFit[k][which];
         for (j = 0; j<=0x1f; j++) {
   	which = svtsim_whichFit_full(tf,k, i, j);
   	/*
   	 * If long-cluster combination has no entry, use the no-LC entry
   	 */
   	if (tf->whichFit[k][which]<0) tf->whichFit[k][which] = which0;
         }
       }
       if (0) {
         printf("whichfit(%d):", k);
         for (i = 0; i<SVTSIM_NEL(tf->whichFit[k]); i++)
   	printf(" %d", tf->whichFit[k][i]);
         printf("\n");
       }
     }
  
  #ifdef DEBUG_READFCON
     /*check that constant are correctly saved in the array */
     
     int nparam = 6; int ipar; 
     int dimspa = 6; int idim;
     int nbarrel = 6; int ibar;
     int fitBlock = 5; int ifitBlock;
     for(ipar = 0; ipar < nparam; ipar++) {
       for(idim = 0; idim < dimspa; idim++) {
         for(ibar = 0; ibar < nbarrel; ibar++) {
   	for(ifitBlock = 0; ifitBlock < fitBlock; ifitBlock++) {
   	  printf("ipar=%d, idim=%d, ibar=%d, ifitBlock=%d, tf->ifitpar = %.6x, lfitparfcon = %.6llx\n", 
   		 ipar, idim, ibar, ifitBlock,      
   		 tf->ifitpar[ipar][idim][ibar][ifitBlock],
   		 tf->lfitparfcon[ipar][idim][ibar][ifitBlock]);
   	}
         }
         
       }
     }
  #endif
    return 0;
 }


 
int gf_init(tf_arrays_t *ptr_tf){
  
  //GF
  struct tf_arrays *tf = malloc(sizeof(struct tf_arrays));
  
  tf->out = svtsim_cable_new();

  tf->gf_emsk  = 0;
  
  tf->chi2cut = GF_TOTCHI2_CUTVAL;
  tf->svt_emsk = GF_ERRMASK_SVT;
  tf->cdf_emsk = GF_ERRMASK_CDF;
  tf->eoe_emsk = GF_ERRMASK_EOE; /* MASK for the errors */
  
  *ptr_tf = tf;

}

  
int word_decode(int word, int xft, int *ee_word, int *parity_in, int *sector, int *amroad,int *crv, int *crv_sign, int *phi, int *zid, int *lcl, int *hit, int* err) {


  int ee, ep, lyr;
  
  /* --------- Executable starts here ------------ */

  lyr = -999; /* Any invalid numbers != 0-7 */

  if (word > gf_mask(SVT_WORD_WIDTH)) {
    printf("gf_iword_decode: Input data is larger than the maximum SVT word");
    return SVTSIM_GF_ERR;
  }

  /* check if this is a EP or EE word */ 
  ee = (word >> SVT_EE_BIT)  & gf_mask(1);
  ep = (word >> SVT_EP_BIT)  & gf_mask(1);

#ifdef DEBUG_SVT
  printf("in word decode: word = %.6x\n", word);
#endif  
  if (ee && ep) { /* End of Event word */
#ifdef DEBUG_SVT
    printf("in word decode: it's a ee word\n");
#endif
    *ee_word = word;
    *parity_in = (word >> SVT_PAR_BIT) & gf_mask(1);
    lyr = EE_LYR;
  } else if (ee) { /* only EE bit ON is error condition */
    *err |= (1<<UNKNOWN_ERR);
    lyr = EE_LYR; /* We have to check */
  } else if (ep) { /* End of Packet word */
#ifdef DEBUG_SVT
     printf("in word decode: it's a ep word\n");
#endif
    lyr = EP_LYR;    
    *sector = 6;
    /*   *sector = (word >> SVT_SECT_LSB)  & gf_mask(SVT_SECT_WIDTH); */
    *amroad = word  & gf_mask(AMROAD_WORD_WIDTH);
  } else if (xft) { /* Second XFT word */
#ifdef DEBUG_SVT
    printf("in word decode: it's second XFT word\n");
#endif
    *crv = (word >> SVT_CRV_LSB)  & gf_mask(SVT_CRV_WIDTH);
    *crv_sign = (word >> (SVT_CRV_LSB+SVT_CRV_WIDTH))  & gf_mask(1);
    *phi = word & gf_mask(SVT_PHI_WIDTH);
    lyr = XFT_LYR;
#ifdef DEBUG_SVT
    printf("in word decode: lyr = %d\n", lyr);
#endif
  } else { /* SVX hits or the first XFT word */
#ifdef DEBUG_SVT
    printf("in word decode: it's  SVX hits or the first XFT word\n");
#endif
    lyr = (word >> SVT_LYR_LSB)  & gf_mask(SVT_LYR_WIDTH);
    if (lyr == XFT_LYRID) lyr = XFT_LYR;
    *zid = (word >> SVT_Z_LSB)  & gf_mask(SVT_Z_WIDTH);
    *lcl = (word >> SVT_LCLS_BIT) & gf_mask(1);  
    *hit = word & gf_mask(SVT_HIT_WIDTH);        
  }
  return lyr;

}

 
 int  gf_mkaddr(tf_arrays_t tf,
 	  int hitmap, int lclmap, int zmap, 
 	  int *coe_addr, int *int_addr, int *addr, int *err) 
 {
   int iaddr;
   unsigned int datum = 0;
   //uint2 
 
   /* --------- Executable starts here ------------ */
   if ((hitmap<0) || (hitmap > gf_mask(NSVX_PLANE+1)) || /* + XFT_LYR */
       (lclmap<0) || (lclmap > gf_mask(NSVX_PLANE)) ||
       (zmap<0)   || (zmap   > gf_mask(GF_ZID_WIDTH)))
     *err |= (1<<SVTSIM_GF_MKADDR_INVALID);
   
   iaddr = ((zmap&gf_mask(GF_SUBZ_WIDTH)) 
 	   + (lclmap<<MADDR_NCLS_LSB)
 	   + (hitmap<<MADDR_HITM_LSB));
 #define MAXMKA 8192
   if ((iaddr < 0) || (iaddr >= MAXMKA)) {
     printf("gf_mkaddr; invalid MKADDR address\n");
     return SVTSIM_GF_ERR;
   }
   
   int ldat = 0;
   svtsim_get_gfMkAddr(tf, &ldat, 1, iaddr);
   datum = ldat;
   
   *int_addr = datum & gf_mask(OFF_SUBA_WIDTH);
   *coe_addr = (datum >> OFF_SUBA_WIDTH) & gf_mask(PAR_ADDR_WIDTH);
   *addr = iaddr;
   
   return SVTSIM_GF_OK;
 
 }
 
 
 /*
  * Compute GF make-address map
  */
 int  svtsim_get_gfMkAddr(tf_arrays_t tf, int *d, int nd, int d0)
 {
 
   /* 
      d0 = iaddr
      
   */
   int j = 0;
   int md = 0x4000;
   int iz, lcl, hit;
   int nparam = 6; int ipar; 
   int dimspa = 6; int idim;
   int nbarrel = 6; int ibar;
   int nfitBlock = 32; int ifitBlock;
   ibar = 0;
   ifitBlock = 0;
 
 
   if (d0+nd>md) nd = md-d0;
   for (j = 0; j<nd; j++) {
     int i = j+d0;
     int word = 0xffff, intcp = 0, coeff = 0;
     int which;
 
     iz = i&7, lcl = i>>3 & 0x1f, hit = i>>8 & 0x3f;

     which = svtsim_whichFit(tf, iz, hit, lcl);
     coeff = iz + which*6;  /* poor choice for illegal iz=6,7, but compatible */
     intcp = which;

 #ifdef DEBUG_SVT
     printf("in svtsim_get_gfMkAddr: nd = %d, d0 = %d\n", nd,d0);
     printf("in svtsim_get_gfMkAddr: which = %d, coeff = %d, intcp = %d\n", which, coeff, intcp);

 #endif 

     word = coeff<<3 | intcp;
     d[j] = word; 
   }
   return nd;
 }
 
 
 /*
  * fit (no shifts)
  */
 
 int gf_fit_proc(int hit[], int sign_crv, long long int coeff[], long long int intcp, long long int *result, int *err) 
 {
   /*
     the fitter calculates 6 scalar products in parallel: 3 chisq, 3 parameters
     in uscita ho una parola del tipo:
     d=10bit+segno, c=7+segno, phi=13, chi1,2,3=14
   */
   
   /*
     Simulate the fit processor in the board.
     INPUT: hit[]; the hits from SVX and XFT
     coeff[]; Coefficienct parameters for the linear fit.
     intcp;   Intercept parameter for the linear fit.
     OUTPUT: *result; The result of the fit.
     *err; the error condition.
     */
   int samesign = 0;
   long long int temp = 0;
   int i = 0;
   int sign_coeff;
   long long int partRes = 0;
   int g = 0;
 
   /* --------- Executable starts here ------------ */
   *result = 0;
   *err = 0;
 #ifdef DEBUG_SVT
   printf("\n GF_FIT_PROC\n");
   for(g = 0; g < SVTNHITS; g++) {
     printf("hit[%d] = %.6x\n", g, hit[g]);
     printf("coeff[%d] = %.6x\n", g, coeff[g]);
   }
 #endif
 
   for(i = 0; i < SVTNHITS; i++) {
     if (i < NSVX_PLANE) {
       temp += hit[i] * coeff[i];
       partRes = hit[i]  * coeff[i];      
 #ifdef DEBUG_SVT
       printf("coeff[%d] = %.6llx, hit[%d] = %.6x, partRes=%.6llx, temp=%.6llx\n", i, coeff[i],i, hit[i], partRes, temp);
 #endif
     } /*  end if (i < NSVX_PLANE) { */
     else if (i == HIT_PHI) { /* XFT phi */
       hit[i] = (hit[i]&0x400) ? -((~hit[i]&0x3ff)+1) : (hit[i]&0x3ff);
       temp += hit[i] *coeff[i];
       partRes = hit[i] *coeff[i];
 #ifdef DEBUG_SVT     
       printf("coeff[%d] = %.6llx,  hit[%d] = %.6x, partRes=%.6llx, temp=%.6llx\n", i, coeff[i],i, hit[i], partRes, temp);
 #endif
     }
     else if (i == HIT_CRV) { /* XFT curvature (curv already with sign in 
 				fep */
       if(sign_crv == 1){ /* if negative bit is set */
 	temp -= hit[i] *coeff[i];
 	partRes = (hit[i]*coeff[i]);
       }
       else{
 	temp += hit[i] *coeff[i];
 	partRes = (hit[i]*coeff[i]);
       }
       
 #ifdef DEBUG_SVT     
       printf("coeff[%d] = %.6llx, hit[%d] = %.6x, partRes=%.6llx, temp=%.6llx\n", i, coeff[i],i, hit[i], partRes, temp);
 #endif      
     }
     
   }
 #ifdef DEBUG_SVT     
   printf("Intercept: %.6x \n", intcp);
 #endif
   
   *result += temp;
 
   
 #ifdef DEBUG_SVT     
   printf("result (w/o intcp) = %.6llx\n", *result);
 #endif
 
   
   samesign = (*result<0)==(intcp<0);
   *result += intcp;
 #ifdef DEBUG_SVT     
   printf("result (w intcp) = %.12llx\n", *result);
 
 
 #endif
   *result = *result<0 ? -((-*result)>>17) : *result>>17; 
 #ifdef DEBUG_SVT     
   printf("result (after shift >> 17) = %.12llx\n", *result);
 #endif
 
   /*
    if (samesign && abs(*result)>gf_mask3(FIT_DWIDTH)) {
      if (1) {
        printf(stderr, "gf_fit_proc: result overflow!\n");
        printf(stderr, "  intcp=%d result=%d DWIDTH=%d\n",
              intcp, *result, FIT_DWIDTH);
      }
      *err |= (1<<FIT_RESULT_OFLOW_BIT);
      printf("in gf_fit_proc: fit result overflow!\n");
    }
   */
   if (*result > 0)
     *result &= gf_mask3(FIT_DWIDTH);
   else
     *result = -(abs(*result)&gf_mask3(FIT_DWIDTH));
 
 #ifdef DEBUG_SVT
   printf("in gf_fit_proc: result = %.6x\n", *result);
 #endif 
   
   
   return SVTSIM_GF_OK;
 }

void svtsim_cable_addwords(svtsim_cable_t *cable, unsigned int  *word, int nword) {

  const int minwords = 8;
  int nnew = 0;
  svtsim_assert(cable); 
  nnew = cable->ndata + nword;
  if (nnew > cable->mdata) {
    cable->mdata = SVTSIM_MAX(minwords, 2*nnew);
    cable->data = svtsim_realloc(cable->data,cable->mdata*sizeof(cable->data[0]));
  }
  if (word) {
    memcpy(cable->data+cable->ndata, word, nword*sizeof(word[0]));
  } else {
    memset(cable->data+cable->ndata, 0, nword*sizeof(word[0]));
  }
  cable->ndata += nword;

  //  printf("in svtsim_cable_addwords: cable->ndata = %d\n", cable->ndata);
}


void svtsim_cable_addword(svtsim_cable_t *cable, unsigned int word){
  svtsim_cable_addwords(cable, &word, 1);
}

void svtsim_cable_copywords(svtsim_cable_t *cable, unsigned int *word, int nword) {
  svtsim_assert(cable);
  cable->ndata = 0;
  svtsim_cable_addwords(cable, word, nword);
}


int gf_chi2(tf_arrays_t tf, long long int chi[], int* trk_err, long long int *chi2) {
  
  int i;
  int oflow = 0;
  long long int temp = 0;
  long long int chi2memdata = 0;

  *chi2 = 0;  
  /* --------- Executable starts here ------------ */

  for (i=0; i<NCHI; i++) 
    {
#ifdef DEBUG_SVT
      printf("chi[%d]: %.6x \n", i, chi[i]);    
#endif
      temp = abs(chi[i]);
      if(chi[i] < 0) temp++;
      
      /*
      if (chi[i]>=MAXCHI2A) 
      	oflow = 1;
      
      temp = (temp & gf_mask(CHI_AWIDTH));      
      */
      
#ifdef DEBUG_SVT
      printf("temp: %.6x \n", temp);    
#endif

      chi2memdata = temp*temp;
      *chi2 += chi2memdata;
#ifdef DEBUG_SVT
      printf("chi2memdata: %.6llx, *chi2: %.6llx \n", chi2memdata,*chi2 );    
#endif
      
    }

  *chi2 = (*chi2 >> 2);
  
  if ((*chi2 >> 2) > gf_mask(CHI_DWIDTH)) 
    {
      /*      *chi2 = (*chi2 & gf_mask(CHI_DWIDTH)) + (1<<CHI_DWIDTH); */
      *chi2 = 0x7ff;
      *trk_err |= (1 << OFLOW_CHI_BIT);
    }
  
#ifdef DEBUG_SVT
  printf("*chi2: %.6x \n", *chi2);    
#endif

  return SVTSIM_GF_OK;
  
}


int gf_getq(int lyr_config){
  int q = 0;
  switch(lyr_config){
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


int gf_gfunc(int ncomb5h, int icomb5h, int hitmap, int lcmap, int chi2){
 
 int lyr_config;
  int gvalue;
  int newhitmap;
  int newlcmap;
  int q = 0;

#ifdef DEBUG_SVT
  printf("in gf_gfunc: ncomb5h = %d, icomb5h = %d\n",ncomb5h, icomb5h );
#endif

  if(ncomb5h == 1) {
    newhitmap = hitmap;
    newlcmap = lcmap;
  }
  else if(ncomb5h == 5) {
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
  else {
    printf("ERROR in gf_gfunc: wrong number of combinations\n");
  }
  /* end  if(ncomb5h == xx) */
#ifdef DEBUG_SVT
      printf("in gf_gfunc: newhitmap = %.6x, newlcmap = %.6x, chi2 = %.6x\n", newhitmap, newlcmap, chi2);    
#endif  
      lyr_config = newhitmap + (newlcmap << 5);
#ifdef DEBUG_SVT
      printf("layer configuration: %.6x\n", lyr_config);
#endif  
      q = gf_getq(lyr_config);

#ifdef DEBUG_SVT
      printf("quality value: %d\n", q);
      printf("q << 4  : %.6x\n", q << 4);
      printf("chi2 >> 7: %.6x\n", chi2 >> 7);
#endif  
      gvalue = (q << 4) + ((chi2 & 0x3ff) >> 6);
      return gvalue;
}


int gf_stword(int id, int err, int mask)
     /*
       Compose the GF status word in the 7th word from the GF 
       INPUT : err; error summary
               mask; MASK
       OUTPUT : return the gf_stword

       NOTE: Currently this code does not support the parity error and
             FIFO error.
     */

{
   int word;
   
   word = id;
   
   if ((err>>OFLOW_HIT_BIT)&gf_mask(1))
     word |= (1<<GFS_OFL_HIT);
   

   if ((err>>OFLOW_CHI_BIT)&gf_mask(1))
     word |= (1<<GFS_OFL_CHI);
   
   if (((err>>UFLOW_HIT_BIT)&gf_mask(1)) ||
       ((err>>OUTORDER_BIT)&gf_mask(1)))
     word |= (1<<GFS_INV_DATA);
  /* 
   if ((err>>FIT_RESULT_OFLOW_BIT)&gf_mask(1))
     word |= (1<<GFS_OFL_FIT);
*/
    
  return word;

}

void dump_fout(tf_arrays_t tf)
     /* 
 *   Dump FORMATTER output
 *     INPUT: fout; FORMATTER output
 *          */
{
  int it, i, ie;

  for(ie=0; ie < tf->totEvts; ie++){
    printf("============== dump_fout ==================\n");
    printf("#SVT tracks = %d \n", tf->fout_ntrks[ie]);

    for (it = 0; it<tf->fout_ntrks[ie]; it++) {
      printf(" \n----------------------------------------------\n");
      printf(" road %4X, comb %2d :\n GF OUTPUTs: ",
      tf->fout_iroad[ie][it], tf->fout_icmb[ie][it]);
      printf("\t");
    for (i=0; i<4; i++)
      printf("%8lX, ",tf->fout_gfword[ie][it][i]);
    printf("\n\t \t");
    for (i=4; i<7; i++)
      printf("%8lX, ",tf->fout_gfword[ie][it][i]);

    } /* for (it = 0; it<opp->ntrks; it++) { */

    printf("\n ----------------------------------------------\n");
    printf("EE word : %X,  SVT Error : %d,  OPP Error summary : %X\n",
    tf->fout_ee_word[ie], tf->fout_svterr[ie], tf->fout_err_sum[ie]);

    printf("\n");
  }//end loop on events
}



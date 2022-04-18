#include "main.cuh"

void help_message()
{
	printf("-cpu \t 0: naive\t 1: precompute\n2: Cooley-Tukey \n");
	printf("-speedcpu 0: precompute \t 1:Cooley-Tukey \n");
	printf("-speedgpu \t 0: precompute\t 1: Cooley-Tukey Naive\n");
}


//main function ========================================
int main(int argc, char** argv)
{
	LARGE_INTEGER t0, t1, freq;
	uint32_t fg0, ig0, fg1, ig1, fg2, ig2, Ni, nfg0, nig0, i;
	uint8_t cpu = 0, cpu_option = 0, gpu = 0, gpu_option = 0, speedcpu = 0, speedgpu = 0;
	uint32_t seed = 0;
#ifdef QTESLA
	if (NTTSIZE == 4) { fg0 = 3585646; ig0 = 4819347; Ni = 6303745;	}	
	else if (NTTSIZE == 8) { fg0 = 2082318; ig0 = 5750773; Ni = 7354369; }
	else if (NTTSIZE == 16) { fg0 = 684268; ig0 = 7134899; Ni = 7879681; }
	else if (NTTSIZE == 32) { fg0 = 213720; ig0 = 2379409; Ni = 8142337; }
	else if (NTTSIZE == 64) { fg0 = 90438; ig0 = 2436145; Ni = 8273665; }
	else if (NTTSIZE == 256) { fg0 = 56156; ig0 = 6454315; Ni = 8372161; }
	else if (NTTSIZE == 1024) {
		fg0 = 2893; ig0 = 7562460; Ni = 8396785; nfg0 = 6321631; nig0 = 2497826; //2083362 - 5907167 or 6321631  - 2497826, both works
	}


	if (N1 == 2) { fg1 = 8404992;  ig1 = 8404992; }
	else if (N1 == 4) { fg1 = 4819347;  ig1 = 3585646; }
	else if (N1 == 8) { fg1 = 6322675;  ig1 = 2654220; }
	else if (N1 == 16) { fg1 = 3506438;  ig1 = 714467; }
	else if (N1 == 32) { fg1 = 213720;  ig1 = 2379409;}
	else if (N1 == 64) { fg1 = 4153273;  ig1 = 4941040;	}
	else if (N1 == 128) { fg1 = 1461658 ;  ig1 = 2739168; }
	else if (N1 == 256) { fg1 = 2626986;  ig1 = 6140544; }
	else if (N1 == 512) { fg1 = 8369449;  ig1 = 1362288; }

	if (N2 == 2) { fg2 = 8404992;  ig2 = 8404992; }
	else if (N2 == 4) { fg2 = 4819347;  ig2 = 3585646; }
	else if (N2 == 8) { fg2 = 6322675;  ig2 = 2654220; }
	else if (N2 == 16) { fg2 = 3506438;  ig2 = 714467; }
	else if (N2 == 32) { fg2 = 213720;  ig2 = 2379409; }
	else if (N2 == 64) { fg2 = 4153273;  ig2 = 4941040; }
	else if (N2 == 128) { fg2 = 1461658;  ig2 = 2739168; }
	else if (N2 == 256) { fg2 = 2626986;  ig2 = 6140544; }
	else if (N2 == 512) { fg2 = 8369449;  ig2 = 1362288; }
#endif
#ifdef SMALLPRIME
	if (NTTSIZE ==  4){ fg0 = 256; ig0 = 65281; Ni = 49153; }
	if (NTTSIZE ==  8){ fg0 = 16;  ig0 = 61441; Ni = 57345; }
	if (NTTSIZE == 16){ fg0 = 4;   ig0 = 49153; Ni = 61441; }
	if (NTTSIZE == 32){ fg0 = 2;   ig0 = 32769; Ni = 63489; }

	if (N1 ==  2){ fg1 = ig1 = 65536;      } 
	if (N1 == 4) { fg1 = 256; ig1 = 65281; }
	if (N1 ==  8){ fg1 = 16;  ig1 = 61441; }
	if (N1 == 16){ fg1 = 4;   ig1 = 49153; }

	if (N2 == 2){  fg2 = ig2 = 65536; }
	if (N2 == 4){  fg2 = 256; ig2 = 65281; }
	if (N2 == 8){  fg2 = 16;  ig2 = 61441; }
	if (N2 == 16){ fg2 = 4;   ig2 = 49153; }
#endif
	printf("NTT Parameters==> NTTSIZE: %d N1: %d N2: %d fg0: %u fg1: %u fg2: %u\n", NTTSIZE, N1, N2, fg0, fg1, fg2);
	
	if (argc < 3) {
		help_message();
		return -1;
	}

	for (int i = 1; i < argc;) {
		if (strcmp(argv[i], "-cpu") == 0) {
			cpu = 1; gpu = 0;
			cpu_option = atoi(argv[i + 1]);
			i += 2;
		}
		else if (strcmp(argv[i], "-speedcpu") == 0) {
			speedcpu = 1; speedgpu = 0;
			cpu_option = atoi(argv[i + 1]);
			i += 2;
		}
		else if (strcmp(argv[i], "-speedgpu") == 0) {
			speedcpu = 0; speedgpu = 1;
			gpu_option = atoi(argv[i + 1]);
			i += 2;
		}
		else if (strcmp(argv[i], "-r") == 0) {			
			seed = atoi(argv[i + 1]);
			i += 2;
		}
		else {
			help_message();
			return -1;
		}
	}
	printf("cpu: %d %d gpu: %d %d speedcpu: %d speedgpu: %d ", cpu, cpu_option, gpu, gpu_option, speedcpu, speedgpu);
	//init and malloc ===================================================================
	uint32_t *x, *y, *z;   //time domain
	uint32_t *X, *Y, *Z;   //frequency domain
	uint32_t *tf0, *ti0;     //twiddle factors & twiddle inverse
	uint32_t *tf1, *ti1;   //sub: twiddle factors & twiddle inverse
	uint32_t *tf2, *ti2;   //sub: twiddle factors & twiddle inverse

	x = (uint32_t*)malloc(sizeof(uint32_t) * BATCH *NTTSIZE); X = (uint32_t*)malloc(sizeof(uint32_t) * BATCH * NTTSIZE);
	y = (uint32_t*)malloc(sizeof(uint32_t) * BATCH *NTTSIZE); Y = (uint32_t*)malloc(sizeof(uint32_t) * BATCH *NTTSIZE);
	z = (uint32_t*)malloc(sizeof(uint32_t) * BATCH *NTTSIZE); Z = (uint32_t*)malloc(sizeof(uint32_t) * BATCH *NTTSIZE);

	tf0 = (uint32_t*)malloc(sizeof(uint32_t) * NTTSIZE);
	ti0 = (uint32_t*)malloc(sizeof(uint32_t) * NTTSIZE);
	tf1 = (uint32_t*)malloc(sizeof(uint32_t) * N1);
	ti1 = (uint32_t*)malloc(sizeof(uint32_t) * N1);
	tf2 = (uint32_t*)malloc(sizeof(uint32_t) * N2);
	ti2 = (uint32_t*)malloc(sizeof(uint32_t) * N2);
	

	//precompute twiddle factors ==========================================================
	for (int i = 0; i < NTTSIZE; i++){
		tf0[i] = ti0[i] = 1;
		for (int k = 0; k < i; k++){
			tf0[i] = (uint64_t)tf0[i] * (uint64_t)fg0 % (uint64_t)P;
		}		
	}

	for (int i = 1; i < NTTSIZE; i++)
	{
		ti0[i] = tf0[NTTSIZE - i];		
	}

	for (int i = 0; i < N1; i++){
		tf1[i] = ti1[i] = 1;
		for (int k = 0; k < i; k++){
			tf1[i] = (uint64_t)tf1[i] * (uint64_t)fg1 % (uint64_t)P;
			ti1[i] = (uint64_t)ti1[i] * (uint64_t)ig1 % (uint64_t)P;

		}
	}
	for (int i = 0; i < N2; i++){
		tf2[i] = ti2[i] = 1;
		for (int k = 0; k < i; k++){
			tf2[i] = (uint64_t)tf2[i] * (uint64_t)fg2 % (uint64_t)P;
			ti2[i] = (uint64_t)ti2[i] * (uint64_t)ig2 % (uint64_t)P;

		}
	}

	//printf twiddle factors ==============================================================
	//printf("\nPrecomputed Twiddle Factors:\n");
	//printf("tf0: "); for (int i = 0; i < NTTSIZE; i++) printf("%lu, ", tf0[i]); printf("\n");
	//printf("ti0: "); for (int i = 0; i < NTTSIZE; i++) printf("%lu, ", ti0[i]); printf("\n");
	//printf("tf1: "); for (int i = 0; i < N1; i++) printf("% 4llu % 4llu\t", tf1[i], tf132[i]); printf("\n");
	//printf("ti1: "); for (int i = 0; i < N1; i++) printf("% 4llu % 4llu\t", ti1[i], ti132[i]); printf("\n");
	//printf("tf2: "); for (int i = 0; i < N2; i++) printf("% 4llu % 4llu \t", tf2[i], tf232[i]); printf("\n");
	//printf("ti2: "); for (int i = 0; i < N2; i++) printf("% 4llu ", ti2[i]); printf("\n");
	
	//Correctness Test ===================================================================
	if (cpu)
	{
		if (cpu_option == 0)
			test_NTT_naive(x, y, z, X, Y, Z, fg0, ig0, Ni);
		else if (cpu_option == 1)
			test_NTT_precom(x, y, z, X, Y, Z, tf0, ti0, Ni);
		else if (cpu_option == 2)
			test_NTT_CT2(x, y, z, X, Y, Z, tf0, tf1, tf2, ti0, ti1, ti2, Ni); 
	}
	//speed performance test ===================================================================
	if (speedcpu)
	{
		printf("\n\n========================\n");
		printf("Speed Test\n");
		printf("========================\n");
		QueryPerformanceFrequency(&freq);
		printf("freq:%llu\n", freq.QuadPart);
		if (cpu_option == 0)
			test_NTT_precompute_BATCH(x, y, z, X, Y, Z, tf0, ti0, Ni);
		else if (cpu_option == 1)
			test_NTT_CT2_BATCH(x, y, z, X, Y, Z, tf0, tf1, tf2, ti0, ti1, ti2, Ni);
		else if (cpu_option == 2)	// Cooley-Tukey + Gentleman-Saude, batch 
			test_NTT_GS_CT_BATCH(x, y, z, X, Y, Z, tf0, ti0, nfg0, nig0, Ni);
		else if (cpu_option == 3)	// nega-cyclic gentleman-sande, batch 
			test_NTT_nega_GS(x, y, z, X, Y, Z, tf0, ti0, nfg0, nig0, Ni);
		else if (cpu_option == 4)	// nega-cyclic cooley-tukey, batch
			test_NTT_nega_CT(x, y, z, X, Y, Z, tf0, ti0, nfg0, nig0, Ni);
		else if (cpu_option == 5)	// nega-cyclic stockham, batch
			test_NTT_Stockham_nega(x, y, z, X, Y, Z, tf0, ti0, nfg0, nig0, Ni);
		else if (cpu_option == 6)
			test_nussbaumer(x, y, z, X, Y, Z);
	}
	if (speedgpu)
	{
		printf("\n\n========================\n");
		printf("Speed Test\n");
		printf("========================\n");
		QueryPerformanceFrequency(&freq);
		printf("freq:%llu\n", freq.QuadPart);
		if (gpu_option == 0)
			test_NTT_precom_gpu(x, y, z, X, Y, Z, tf0, ti0, Ni);
		else if (gpu_option == 1)
			test_NTT_CT2_gpu(x, y, z, X, Y, Z, tf0, tf1, tf2, ti0, ti1, ti2, Ni);
		else if (gpu_option == 2)
			test_NTT_Stockham_nega_gpu(x, y, z, X, Y, Z, tf0, ti0, nfg0, nig0, Ni);
		else if (gpu_option == 3)
			test_NTT_GS_CT_nega_gpu(x, y, z, X, Y, Z, tf0, ti0, nfg0, nig0, Ni);
		else if (gpu_option == 4)
			test_NTT_CT_CT_nega_gpu(x, y, z, X, Y, Z, tf0, ti0, nfg0, nig0, Ni);
		else if (gpu_option == 5)
			test_NTT_GS_GS_nega_gpu(x, y, z, X, Y, Z, tf0, ti0, nfg0, nig0, Ni);
		else if (gpu_option == 6)
			test_NTT_CT_GS_nega_gpu(x, y, z, X, Y, Z, tf0, ti0, nfg0, nig0, Ni);
		else if (gpu_option == 7)
			test_reduction();
		else if (gpu_option == 8)
		{
			//for(i=0; i<5; i++)
			//	test_NTT_Stockham_nega_gpu(x, y, z, X, Y, Z, tf0, ti0, nfg0, nig0, Ni);
			for (i = 0; i < 5; i++)
				test_NTT_GS_CT_nega_gpu(x, y, z, X, Y, Z, tf0, ti0, nfg0, nig0, Ni);
			//for (i = 0; i < 5; i++)
			//	test_NTT_CT_CT_nega_gpu(x, y, z, X, Y, Z, tf0, ti0, nfg0, nig0, Ni);
			////for (i = 0; i < 5; i++)
			////	test_NTT_GS_GS_nega_gpu(x, y, z, X, Y, Z, tf0, ti0, nfg0, nig0, Ni);
			for (i = 0; i < 5; i++)
				test_NTT_CT_GS_nega_gpu(x, y, z, X, Y, Z, tf0, ti0, nfg0, nig0, Ni);
		}
	}	

	printf("\n");	
	return 0;
}


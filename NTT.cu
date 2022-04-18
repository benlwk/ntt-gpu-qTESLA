#include "main.cuh"
#include "constants.h"
//helper functions =====================================
void init_operand(uint32_t *x)
{
	LARGE_INTEGER t0;
	QueryPerformanceCounter(&t0);
	srand(t0.LowPart);

	if (!RANDOM)
		for (int i = 0; i < NSIZE_hlf; i++) { x[i] = NSIZE_hlf - i; }
	else
		for (int i = 0; i < NSIZE_hlf; i++) { x[i] = rand() % 10; }

	for (int i = NSIZE_hlf; i < NTTSIZE; i++) { x[i] = 0; }
}
void init_operands(uint32_t *x, uint32_t *y)
{
	init_operand(x);
	init_operand(y);
}
void clear_operand(uint32_t *x)
{
	for (int i = 0; i < NTTSIZE; i++){ x[i] = 0; }
}
void reset(uint32_t *x, uint32_t *y, uint32_t *X, uint32_t *Y)
{
	clear_operand(x);
	clear_operand(y);
	clear_operand(X);
	clear_operand(Y);
}
__host__ uint32_t _addModP_cpu(uint32_t ip1, uint32_t ip2)
{
	uint64_t ans = ip1 + ip2;
	return (ans >= P) ? ans - P : ans;
	return ans;
}

__host__  uint32_t _subModP_cpu(uint32_t ip1, uint32_t ip2)
{
	if (ip1 < ip2) ip1 += P;

	uint64_t ans = ip1 - ip2;
	return (ans >= P) ? ans - P : ans;
	//return ans;
}

// Reversing bits in a word, basic interchange scheme.
//unsigned rev_bit(unsigned x) {
//	x = (x & 0x55555555) << 1 | (x & 0xAAAAAAAA) >> 1;
//	x = (x & 0x33333333) << 2 | (x & 0xCCCCCCCC) >> 2;
//#ifdef QTESLA
//	x = (x & 0x0F0F0F0F) << 4 | (x & 0xF0F0F0F0) >> 4;
//	x = (x & 0x00FF00FF) << 8 | (x & 0xFF00FF00) >> 8;
//	//x = (x & 0x0000FFFF) << 16 | (x & 0xFFFF0000) >> 16;
//#endif
//	return x;
//}

unsigned int bitrev(unsigned int n, unsigned int bits)
{
	unsigned int nrev, N;
	unsigned int count;
	N = 1 << bits;
	count = bits - 1;   // initialize the count variable
	nrev = n;
	for (n >>= 1; n; n >>= 1)
	{
		nrev <<= 1;
		nrev |= n & 1;
		count--;
	}

	nrev <<= count;
	nrev &= N - 1;

	return nrev;
}

void bit_reverse_copy(uint32_t* ip, uint32_t* op)
{
	int k = 0, j=0;
	for (k = 0; k < BATCH; k++)
		for (j = 0; j < NTTSIZE; j++)
		{
			op[NTTSIZE * k + j] = ip[NTTSIZE * k + bitrev(j, 10)];
			//printf("%u, ", bitrev(j, 10));
		}
	//printf("%08lx\n", rev_bit(k));
}
void bit_reverse_copy_tbl(uint32_t* ip, uint32_t* op)
{
	int k = 0, j = 0;
	for (k = 0; k < BATCH; k++)
		for (j = 0; j < NTTSIZE; j++)
		{
			op[NTTSIZE * k + j] = ip[NTTSIZE * k + bitrev_tbl[j]];
		}
}

#define modadd(c,a,b) \
do { \
  uint32_t _t = a+b; \
  c = _t + (_t < a); \
} while (0)

#define modsub(c,a,b) c = (a-b) - (b > a)

#define modmul(c,a,b) \
do { \
  uint64_t _T = (uint64_t) a * (uint64_t) b; \
  modadd (c, ((uint32_t) _T), ((uint32_t) ((uint64_t) _T >> (uint64_t) 32))); \
} while (0)


#define modmuladd(c,a,b) \
do { \
  uint64_t _T = (uint64_t) a * (uint64_t) b + c; \
  modadd (c, ((uint32_t) _T), ((uint32_t) ((uint64_t) _T >> (uint64_t) 32))); \
} while (0)

#define div2(c,a) c= (uint32_t) (((uint64_t) (a) + (uint64_t) ((uint32_t)(0-((a)&1))&0xFFFFFFFF))>>1)
#define normalize(c,a) c = (a) + ((a) == 0xFFFFFFFF)

/* Define the basic building blocks for the FFT. */
#define SET_ZERO(x) (x)=0
#define add(c,a,b) modadd(c,a,b)
#define sub(c,a,b) modsub(c,a,b)
#define mul(c,a,b) modmul(c,a,b)
#define moddiv2(c,a)  normalize(c,a); div2(c,c)
#define neg(c,a)   (c)=0xFFFFFFFF-(a); normalize(c,c)
#define squ(c,a)   mul(c,a,a)
#define set(c,a)   (c)=(a)

/* Reverse the bits, approach from "Bit Twiddling Hacks"
 * See: https://graphics.stanford.edu/~seander/bithacks.html
 */
static uint32_t reverse(uint32_t x) {
	x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
	x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
	x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
	x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
	return ((x >> 16) | (x << 16));
}

static void naive(uint32_t* z, const uint32_t* x, const uint32_t* y, unsigned int n) {
	unsigned int i, j, k;
	uint32_t A, B;

	for (i = 0; i < n; i++) {
		SET_ZERO(B);
		// A = x[0] * y [i]
		mul(A, x[0], y[i]);

		for (j = 1; j <= i; j++) {
			modmuladd(A, x[j], y[i - j]);
		}

		for (k = 1; j < n; j++, k++) {
			modmuladd(B, x[j], y[n - k]);
		}
		sub(z[i], A, B);
	}
}

static void nussbaumer_fft(uint32_t* z, const uint32_t* x, const uint32_t* y) {
	uint32_t** X1;
	uint32_t** Y1;
	uint32_t** Z1;
	uint32_t* T1;
	unsigned int i;
	int j;

	X1 = (uint32_t * *)malloc(64 * sizeof(uint32_t*));
	Y1 = (uint32_t * *)malloc(64 * sizeof(uint32_t*));
	Z1 = (uint32_t * *)malloc(64 * sizeof(uint32_t*));
	T1 = (uint32_t*)malloc(64 * sizeof(uint32_t));
	for (int i = 0; i < 64; i++) {
		X1[i] = (uint32_t*)malloc(64 * sizeof(uint32_t));
		Y1[i] = (uint32_t*)malloc(64 * sizeof(uint32_t));
		Z1[i] = (uint32_t*)malloc(64 * sizeof(uint32_t));
	}

	for (i = 0; i < 32; i++) {
		for (j = 0; j < 32; j++) {
			X1[i][j] = x[32 * j + i];
			X1[i + 32][j] = x[32 * j + i];

			Y1[i][j] = y[32 * j + i];
			Y1[i + 32][j] = y[32 * j + i];
		}
	}

	for (j = 4; j >= 0; j--) {
		for (i = 0; i < (1U << (5 - j)); i++) {//i = 1->31 (2->32)
			unsigned int t, ssr = reverse(i);
			for (t = 0; t < (1U << j); t++) {	// t = 15->0 (16->1)
				unsigned int s, sr, I, L, a;
				s = i;
				sr = (ssr >> (32 - 5 + j));
				sr <<= j;
				s <<= (j + 1);

				// X_i(w) = X_i(w) + w^kX_l(w) can be computed as
				// X_ij = X_ij - X_l(j-k+r)  for  0 <= j < k
				// X_ij = X_ij + X_l(j-k)    for  k <= j < r
				I = s + t, L = s + t + (1 << j);

				for (a = sr; a < 32; a++) {
					T1[a] = X1[L][a - sr];
				}
				for (a = 0; a < sr; a++) {
					neg(T1[a], X1[L][32 + a - sr]);
				}

				for (a = 0; a < 32; a++) {
					sub(X1[L][a], X1[I][a], T1[a]);
					add(X1[I][a], X1[I][a], T1[a]);
				}

				for (a = sr; a < 32; a++) {
					T1[a] = Y1[L][a - sr];
				}
				for (a = 0; a < sr; a++) {
					neg(T1[a], Y1[L][32 + a - sr]);
				}

				for (a = 0; a < 32; a++) {
					sub(Y1[L][a], Y1[I][a], T1[a]);
					add(Y1[I][a], Y1[I][a], T1[a]);
				}
			}
		}
	}

	for (i = 0; i < 2 * 32; i++) {
		naive(Z1[i], X1[i], Y1[i], 32);
	}

	for (j = 0; j <= (int)5; j++) {
		for (i = 0; i < (1U << (5 - j)); i++) {//i = 31->0 (32->1)
			unsigned int t, ssr = reverse(i);
			for (t = 0; t < (1U << j); t++) {// j = 0->5				
				unsigned int s, sr, A, B, a;
				s = i;
				sr = (ssr >> (32 - 5 + j));
				sr <<= j;
				s <<= (j + 1);

				A = s + t;
				B = s + t + (1 << j);
				for (a = 0; a < 32; a++) {
					sub(T1[a], Z1[A][a], Z1[B][a]);
					moddiv2(T1[a], T1[a]);
					add(Z1[A][a], Z1[A][a], Z1[B][a]);
					moddiv2(Z1[A][a], Z1[A][a]);
				}

				// w^{-(r/m)s'} (Z_{s+t}(w)-Z_{s+t+2^j}(w))
				for (a = 0; a < 32 - sr; a++) {
					Z1[B][a] = T1[a + sr];
				}
				for (a = 32 - sr; a < 32; a++) {
					neg(Z1[B][a], T1[a - (32 - sr)]);
				}
			}
		}
	}

	for (i = 0; i < 32; i++) {
		sub(z[i], Z1[i][0], Z1[32 + i][32 - 1]);
		for (j = 1; j < 32; j++) {
			add(z[32 * j + i], Z1[i][j], Z1[32 + i][j - 1]);
		}
	}
}



//#include<time.h>
__global__ void red_assembly(uint32_t *time)
{
	uint32_t r1 = 0, r2 = 0, res = 0, tmp1=0, tmp2=0, tmp3=0, tmp4=0, i = 0;
	clock_t start, stop, boy = 0;
	uint64_t r = (uint64_t)8404993* (uint64_t)8404992 + 5;
	
	start = clock();
	//tmp1 = (r >> 24) + (r >> 25) + (r >> 26) + (r >> 27) + (r >> 28) + (r >> 29) + (r >> 30) + (r >> 31) + (r >> 32) + (r >> 42) + (r >> 43) + (r >> 44) + (r >> 45);
	//asm("mov.b64 { %0, %1 }, %2;" : "=r"(r1), "=r"(r2) : "l"(r));//r1=lo word, r2=hi word	
	//asm("shr.b32 %0, %1, %2;" : "=r"(tmp2) : "r"(r2), "r"(10));	//r >> 42
	//asm("shr.b32 %0, %1, %2;" : "=r"(tmp3) : "r"(r2), "r"(11));	//r >> 43	
	//asm("add.u32 %0, %1, %2;" : "=r"(tmp4) : "r"(tmp3), "r"(tmp2));
	//asm("shr.b32 %0, %1, %2;" : "=r"(tmp3) : "r"(r2), "r"(12));	//r >> 44
	//asm("add.u32 %0, %1, %2;" : "=r"(tmp2) : "r"(tmp3), "r"(tmp4));
	//asm("shr.b32 %0, %1, %2;" : "=r"(tmp3) : "r"(r2), "r"(13));	//r >> 45
	//asm("add.u32 %0, %1, %2;" : "=r"(tmp4) : "r"(tmp3), "r"(tmp2));
	//asm("shr.b32 %0, %1, %2;" : "=r"(tmp3) : "r"(r1), "r"(24)); //r >> 24
	//asm("shl.b32 %0, %1, %2;" : "=r"(tmp2) : "r"(r2), "r"(8)); 
	//asm("xor.b32 %0, %1, %2;" : "=r"(tmp1) : "r"(tmp2), "r"(tmp3)); //tmp1 = r2<<8 | r1 >>24
	//asm("add.u32 %0, %1, %2;" : "=r"(tmp1) : "r"(tmp4), "r"(tmp1));
	//asm("shr.b32 %0, %1, %2;" : "=r"(tmp3) : "r"(r1), "r"(25)); //r >> 25
	//asm("shl.b32 %0, %1, %2;" : "=r"(tmp2) : "r"(r2), "r"(7));
	//asm("xor.b32 %0, %1, %2;" : "=r"(tmp4) : "r"(tmp2), "r"(tmp3)); //tmp4 = r2<<7 | r1 >>25
	//asm("add.u32 %0, %1, %2;" : "=r"(tmp1) : "r"(tmp4), "r"(tmp1));
	//asm("shr.b32 %0, %1, %2;" : "=r"(tmp3) : "r"(r1), "r"(26)); //r >> 26
	//asm("shl.b32 %0, %1, %2;" : "=r"(tmp2) : "r"(r2), "r"(6));
	//asm("xor.b32 %0, %1, %2;" : "=r"(tmp4) : "r"(tmp2), "r"(tmp3));
	//asm("add.u32 %0, %1, %2;" : "=r"(tmp1) : "r"(tmp4), "r"(tmp1));
	//asm("shr.b32 %0, %1, %2;" : "=r"(tmp3) : "r"(r1), "r"(27)); //r >> 27
	//asm("shl.b32 %0, %1, %2;" : "=r"(tmp2) : "r"(r2), "r"(5));
	//asm("xor.b32 %0, %1, %2;" : "=r"(tmp4) : "r"(tmp2), "r"(tmp3)); 
	//asm("add.u32 %0, %1, %2;" : "=r"(tmp1) : "r"(tmp4), "r"(tmp1));
	//asm("shr.b32 %0, %1, %2;" : "=r"(tmp3) : "r"(r1), "r"(28)); //r >> 28
	//asm("shl.b32 %0, %1, %2;" : "=r"(tmp2) : "r"(r2), "r"(4));
	//asm("xor.b32 %0, %1, %2;" : "=r"(tmp4) : "r"(tmp2), "r"(tmp3));
	//asm("add.u32 %0, %1, %2;" : "=r"(tmp1) : "r"(tmp4), "r"(tmp1));
	//asm("shr.b32 %0, %1, %2;" : "=r"(tmp3) : "r"(r1), "r"(29)); //r >> 29
	//asm("shl.b32 %0, %1, %2;" : "=r"(tmp2) : "r"(r2), "r"(3));
	//asm("xor.b32 %0, %1, %2;" : "=r"(tmp4) : "r"(tmp2), "r"(tmp3));
	//asm("add.u32 %0, %1, %2;" : "=r"(tmp1) : "r"(tmp4), "r"(tmp1));
	//asm("shr.b32 %0, %1, %2;" : "=r"(tmp3) : "r"(r1), "r"(30)); //r >> 30
	//asm("shl.b32 %0, %1, %2;" : "=r"(tmp2) : "r"(r2), "r"(2));
	//asm("xor.b32 %0, %1, %2;" : "=r"(tmp4) : "r"(tmp2), "r"(tmp3));
	//asm("add.u32 %0, %1, %2;" : "=r"(tmp1) : "r"(tmp4), "r"(tmp1));
	//asm("shr.b32 %0, %1, %2;" : "=r"(tmp3) : "r"(r1), "r"(31)); //r >> 31
	//asm("shl.b32 %0, %1, %2;" : "=r"(tmp2) : "r"(r2), "r"(1));
	//asm("xor.b32 %0, %1, %2;" : "=r"(tmp4) : "r"(tmp2), "r"(tmp3));
	//asm("add.u32 %0, %1, %2;" : "=r"(tmp1) : "r"(tmp4), "r"(tmp1));
	//asm("add.u32 %0, %1, %2;" : "=r"(tmp1) : "r"(r2), "r"(tmp1)); // r >> 32

	stop = clock();
	stop = clock();
	//res = r - (uint64_t)tmp1 * (uint64_t)8404993;
	//while (res > P) res = res - P;
	stop = clock();
	//__syncthreads();
	time[blockIdx.x * blockDim.x + threadIdx.x] = stop - start;
	//printf("%llu %lu %lu \n", elapsed, start, stop);
}
uint32_t barrett_red_cpu(uint64_t ip)
{
	uint32_t res = 0, q1 = 0, q3 = 0, count = 0;
	uint64_t q2 = 0;

	//q1 = ip >> 23;
	//q2 = (uint64_t)q1 * (uint64_t)MIU;
	//q3 = q2 >> 25;
	//res = ip - (uint64_t)q3 * (uint64_t)P;
	//while (res > P) res = res - P;	//Final reduction, make it constant time?
	//q2 = (uint64_t)(ip >> 24) + (ip >> 25) + (ip >> 26) + (ip >> 27) + (ip >> 28) + (ip >> 29) + (ip >> 30) + (ip >> 31) + (ip >> 32) + (ip >> 42) + (ip >> 43) + (ip >> 44) + (ip >> 45);
	q2 = (ip >> 24) + (ip >> 25) + (ip >> 26) + (ip >> 27) + (ip >> 28) + (ip >> 29) + (ip >> 30) + (ip >> 31) + (ip >> 32) + (ip >> 42) + (ip >> 43) + (ip >> 44) + (ip >> 45) + (ip >> 47) + (ip >> 48) + (ip >> 49) + (ip >> 50) + (ip >> 54) + (ip >> 60);
	res = ip - (uint64_t)q2 * (uint64_t)P;
	while (res > P)
	{
		res = res - P;
		count++;
	}
	if (count > 0) printf("count: %d %u\n", count, q2);
	return res;
}

#define THREAD 256
#define BLOCK 16
void test_reduction()
{
	//uint32_t* time, *c_time;
	//cudaMalloc((void**)&time, THREAD * BLOCK * sizeof(uint32_t));
	//cudaMallocHost((void**)&c_time, THREAD * BLOCK * sizeof(uint32_t));

	//red_assembly << <BLOCK, THREAD >> > (time);
	//cudaMemcpy(c_time, time, THREAD * BLOCK * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	//for(int i=0; i<32; i++) printf("%llu \n", c_time[i+THREAD]);
	uint64_t a = (uint64_t) 8404993 * 8404992 + 8404991;
	uint32_t b = barrett_red_cpu(17422367029524);
	printf("a: %llu b: %lu\n", a, b);
}
// Reduce 64-bit ip to 32-bit output (ip mod P)
__device__ uint32_t barrett_red(uint64_t ip)
{
	uint32_t count = 0, res = 0, q1 = 0, q3 = 0, r1 = 0, r2 = 0, tmp1 = 0, tmp2 = 0, tmp3 = 0, tmp4 = 0;
	uint64_t q2 = 0;

	//wklee, Barrett reduction from Ch14 Handbook of Appl. Crypt.
	q1 = ip >> 23;
	q2 = (uint64_t)q1 * (uint64_t)MIU;
	q3 = q2 >> 25;
	res = ip - (uint64_t)q3 * (uint64_t)P;
	//while (res > P) res = res - P;	//Final reduction, make it constant time?
 // Montgomery reduction
	//int64_t u;
	////int32_t result;
	//q2 = (ip * PARAM_QINV) & 0xFFFFFFFF;
	//q2 *= P;
	//ip += q2;
	//res = (int32_t)(ip >> 32);			
		
	//asm("mov.b64 { %0, %1 }, %2;" : "=r"(r1), "=r"(r2) : "l"(ip));//r1=lo word, r2=hi word
	//asm("shr.b32 %0, %1, %2;" : "=r"(tmp2) : "r"(r2), "r"(10));	//r >> 42
	//asm("shr.b32 %0, %1, %2;" : "=r"(tmp3) : "r"(r2), "r"(11));	//r >> 43	
	//asm("add.u32 %0, %1, %2;" : "=r"(tmp4) : "r"(tmp3), "r"(tmp2));
	//asm("shr.b32 %0, %1, %2;" : "=r"(tmp3) : "r"(r2), "r"(12));	//r >> 44
	//asm("add.u32 %0, %1, %2;" : "=r"(tmp2) : "r"(tmp3), "r"(tmp4));
	//asm("shr.b32 %0, %1, %2;" : "=r"(tmp3) : "r"(r2), "r"(13));	//r >> 45
	//asm("add.u32 %0, %1, %2;" : "=r"(tmp4) : "r"(tmp3), "r"(tmp2));

	//asm("shr.b32 %0, %1, %2;" : "=r"(tmp3) : "r"(r1), "r"(24)); //r >> 24
	//asm("shl.b32 %0, %1, %2;" : "=r"(tmp2) : "r"(r2), "r"(8));
	//asm("xor.b32 %0, %1, %2;" : "=r"(tmp1) : "r"(tmp2), "r"(tmp3)); //tmp1 = r2<<8 | r1 >>24
	//asm("add.u32 %0, %1, %2;" : "=r"(tmp1) : "r"(tmp4), "r"(tmp1));
	//asm("shr.b32 %0, %1, %2;" : "=r"(tmp3) : "r"(r1), "r"(25)); //r >> 25
	//asm("shl.b32 %0, %1, %2;" : "=r"(tmp2) : "r"(r2), "r"(7));
	//asm("xor.b32 %0, %1, %2;" : "=r"(tmp4) : "r"(tmp2), "r"(tmp3)); //tmp4 = r2<<7 | r1 >>25
	//asm("add.u32 %0, %1, %2;" : "=r"(tmp1) : "r"(tmp4), "r"(tmp1));
	//asm("shr.b32 %0, %1, %2;" : "=r"(tmp3) : "r"(r1), "r"(26)); //r >> 26
	//asm("shl.b32 %0, %1, %2;" : "=r"(tmp2) : "r"(r2), "r"(6));
	//asm("xor.b32 %0, %1, %2;" : "=r"(tmp4) : "r"(tmp2), "r"(tmp3));
	//asm("add.u32 %0, %1, %2;" : "=r"(tmp1) : "r"(tmp4), "r"(tmp1));
	//asm("shr.b32 %0, %1, %2;" : "=r"(tmp3) : "r"(r1), "r"(27)); //r >> 27
	//asm("shl.b32 %0, %1, %2;" : "=r"(tmp2) : "r"(r2), "r"(5));
	//asm("xor.b32 %0, %1, %2;" : "=r"(tmp4) : "r"(tmp2), "r"(tmp3));
	//asm("add.u32 %0, %1, %2;" : "=r"(tmp1) : "r"(tmp4), "r"(tmp1));
	//asm("shr.b32 %0, %1, %2;" : "=r"(tmp3) : "r"(r1), "r"(28)); //r >> 28
	//asm("shl.b32 %0, %1, %2;" : "=r"(tmp2) : "r"(r2), "r"(4));
	//asm("xor.b32 %0, %1, %2;" : "=r"(tmp4) : "r"(tmp2), "r"(tmp3));
	//asm("add.u32 %0, %1, %2;" : "=r"(tmp1) : "r"(tmp4), "r"(tmp1));
	//asm("shr.b32 %0, %1, %2;" : "=r"(tmp3) : "r"(r1), "r"(29)); //r >> 29
	//asm("shl.b32 %0, %1, %2;" : "=r"(tmp2) : "r"(r2), "r"(3));
	//asm("xor.b32 %0, %1, %2;" : "=r"(tmp4) : "r"(tmp2), "r"(tmp3));
	//asm("add.u32 %0, %1, %2;" : "=r"(tmp1) : "r"(tmp4), "r"(tmp1));
	//asm("shr.b32 %0, %1, %2;" : "=r"(tmp3) : "r"(r1), "r"(30)); //r >> 30
	//asm("shl.b32 %0, %1, %2;" : "=r"(tmp2) : "r"(r2), "r"(2));
	//asm("xor.b32 %0, %1, %2;" : "=r"(tmp4) : "r"(tmp2), "r"(tmp3));
	//asm("add.u32 %0, %1, %2;" : "=r"(tmp1) : "r"(tmp4), "r"(tmp1));
	//asm("shr.b32 %0, %1, %2;" : "=r"(tmp3) : "r"(r1), "r"(31)); //r >> 31
	//asm("shl.b32 %0, %1, %2;" : "=r"(tmp2) : "r"(r2), "r"(1));
	//asm("xor.b32 %0, %1, %2;" : "=r"(tmp4) : "r"(tmp2), "r"(tmp3));
	//asm("add.u32 %0, %1, %2;" : "=r"(tmp1) : "r"(tmp4), "r"(tmp1));
	//asm("add.u32 %0, %1, %2;" : "=r"(tmp1) : "r"(r2), "r"(tmp1)); // r >> 32
	//res = ip - (uint64_t)tmp1 * (uint64_t)P;
	//res = res - P;// wklee, divergence here, how to solve?
	//if(res > P)res = res - P;

	//q2 = (ip >> 27) + (ip >> 28) + (ip >> 29) + (ip >> 30) + (ip >> 31) + (ip >> 32) + (ip >> 33) + (ip >> 34) + (ip >> 35) + (ip >> 42) + (ip >> 43) + (ip >> 44) + (ip >> 45) + (ip >> 47);
	//res = ip - (uint64_t)q2 * (uint64_t)P;
	while (res > P) {
		//count++;
		res = res - P;
	}
	//if(count>1) printf("count: %d %llu\n", count, ip);
	return res;                         
}

__device__ uint32_t _addModP(uint32_t ip1, uint32_t ip2)
{
	uint64_t ans = ip1 + ip2;
	//ans = barrett_red(ans);
	//return ans;
	return (ans >= P) ? ans - P : ans;
}

__device__  uint32_t _subModP(uint32_t ip1, uint32_t ip2)
{
	if (ip1 < ip2) ip1 += P;

	uint64_t ans = ip1 - ip2;
	return (ans >= P) ? ans - P : ans;
	//ans = barrett_red(ans);
	//return ans;
}

//__host__ uint32_t _addModP_cpu(uint32_t ip1, uint32_t ip2)
//{
//	uint64_t ans = ip1 + ip2;
//	return (ans >= P) ? ans - P : ans;	
//}
//
//__host__  uint32_t _subModP_cpu(uint32_t ip1, uint32_t ip2)
//{
//	if (ip1 < ip2) ip1 += P;
//
//	uint64_t ans = ip1 - ip2;
//	return (ans >= P) ? ans - P : ans;
//	//return ans;
//}

__global__ void bit_reverse_copy_tbl_gpu(uint32_t* ip, uint32_t* op)
{
	int b = blockIdx.x, tid = threadIdx.x;

	op[NTTSIZE * b + tid] = ip[NTTSIZE * b + bitrev_tbl_gpu[tid]];
}

__global__ void bit_reverse_copy_tbl_invPhi_gpu(uint32_t* ip, uint32_t* op)
{
	int b = blockIdx.x, tid = threadIdx.x;

	op[NTTSIZE * b + tid] = ip[NTTSIZE * b + bitrev_tbl_gpu[tid]];
	op[NTTSIZE * b + tid] = barrett_red((uint64_t)op[NTTSIZE * b + tid] * (uint64_t)invPhi_gpu[tid]);
}

__global__ void bit_reverse_copy_tbl_Phi_gpu(uint32_t* ip, uint32_t* op)
{
	int b = blockIdx.x, tid = threadIdx.x;

	ip[NTTSIZE * b + tid] = barrett_red((uint64_t)ip[NTTSIZE * b + tid] * (uint64_t)Phi_gpu[tid]);
	__syncthreads();
	op[NTTSIZE * b + tid] = ip[NTTSIZE * b + bitrev_tbl_gpu[tid]];
}
//====================================================
// ip : polynomial coefficients
// op : NTT transformed polynomial coefficients
// fg : forward primitive root of unity
//====================================================
void NTT_naive(uint32_t *ip, uint32_t *op, uint32_t fg)
{
	uint32_t tg;
	for (int i = 0; i < NTTSIZE; i++) {
		op[i] = 0;
		for (int j = 0; j < NTTSIZE; j++) {
			tg = 1;
			for (int k = 0; k < i * j; k++) {
				tg *= fg;
				tg %= P;
			}
			op[i] += ip[j] * tg;
			op[i] %= P;
		}
	}
}
//====================================================
// ip : NTT transformed polynomial coefficients
// op : polynomial coefficients
// ig : inverse primitive root of unity
// Ni : multiplicative inverse of N that fulfill N * Ni mod P = 1
//====================================================
void INTT_naive(uint32_t *ip, uint32_t *op, uint32_t ig, uint32_t Ni)
{
	uint32_t tg;
	for (int i = 0; i < NTTSIZE; i++) {
		op[i] = 0;
		for (int j = 0; j < NTTSIZE; j++) {
			tg = 1;
			for (int k = 0; k < i * j; k++) {
				tg *= ig;
				tg %= P;
			}
			op[i] += ip[j] * tg;
			op[i] %= P;
		}
		op[i] *= Ni;
		op[i] %= P;
	}
}
//====================================================
// ip : polynomial coefficients
// op : NTT transformed polynomial coefficients
// tf : forward twiddle factors
//====================================================
void NTT_precom(uint32_t *ip, uint32_t *op, uint32_t *tf)
{
	uint64_t temp = 0;
	for (int i = 0; i < NTTSIZE; i++) {
		op[i] = 0;
		for (int j = 0; j < NTTSIZE; j++) {
			temp = ((uint64_t) op[i] + (uint64_t)ip[j] * (uint64_t)tf[(i * j) % NTTSIZE]);
			op[i] = (uint64_t)temp % (uint64_t)P;
		}
	}
}

void NTT_precom_batch(uint32_t *ip, uint32_t *op, uint32_t *tf)
{
	uint64_t temp = 0, b = 0;
	for (b = 0; b < BATCH; b++)
	{
		for (int i = 0; i < NTTSIZE; i++) {
			op[b * NTTSIZE + i] = 0;
			for (int j = 0; j < NTTSIZE; j++) {
				temp = ((uint64_t)op[b * NTTSIZE + i] + (uint64_t)ip[b * NTTSIZE + j] * (uint64_t)tf[(i * j) % NTTSIZE]);
				op[b * NTTSIZE + i] = temp % P;
			}
		}
	}
}
//====================================================
// ip : polynomial coefficients
// op : NTT transformed polynomial coefficients
// tf : forward twiddle factors
//====================================================
__global__ void NTT_precom_gpu(uint32_t *ip, uint32_t *op, uint32_t *tf)
{
	unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int tid = threadIdx.x;
	uint64_t temp = 0;
	
	op[gid] = 0;
	for (int j = 0; j < NTTSIZE; j++) {
		temp = (uint64_t) op[gid] + (uint64_t)ip[j] * (uint64_t)tf[(tid * j) % NTTSIZE];
		op[gid] = temp % P;
	}
}

//====================================================
// ip : NTT transformed polynomial coefficients 
// op : polynomial coefficients
// ti : inverse twiddle factors
// Ni : multiplicative inverse of N that fulfill N * Ni mod P = 1
//====================================================
void INTT_precom(uint32_t *ip, uint32_t *op, uint32_t *ti, uint32_t Ni)
{
	uint64_t temp = 0;
	for (int i = 0; i < NTTSIZE; i++) {
		op[i] = 0;
		for (int j = 0; j < NTTSIZE; j++) {
			temp = ((uint64_t)op[i] + (uint64_t) ip[j] * (uint64_t) ti[(i * j) % NTTSIZE]);
			op[i] = temp % P;
		}
		temp = (uint64_t) op[i] * (uint64_t) Ni;
		op[i] = (uint64_t) temp % (uint64_t)P;
	}
}

void INTT_precom_batch(uint32_t *ip, uint32_t *op, uint32_t *ti, uint32_t Ni)
{
	uint64_t temp = 0, b =  0;
	for (b = 0; b < BATCH; b++)
	{
		for (int i = 0; i < NTTSIZE; i++) {
			op[b * NTTSIZE + i] = 0;
			for (int j = 0; j < NTTSIZE; j++) {
				temp = ((uint64_t)op[b * NTTSIZE + i] + (uint64_t)ip[b * NTTSIZE + j] * (uint64_t)ti[(i * j) % NTTSIZE]);
				op[b * NTTSIZE + i] = temp % P;
			}
			temp = (uint64_t)op[b * NTTSIZE + i] * (uint64_t)Ni;
			op[b * NTTSIZE + i] = temp % P;
		}
	}
}
//====================================================
// ip : NTT transformed polynomial coefficients 
// op : polynomial coefficients
// ti : inverse twiddle factors
// Ni : multiplicative inverse of N that fulfill N * Ni mod P = 1
//====================================================

__global__ void INTT_precom_gpu(uint32_t *ip, uint32_t *op, uint32_t *ti, uint32_t Ni)
{
	unsigned int tid = threadIdx.x;
	unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t temp = 0;
	op[gid] = 0;
	for (int j = 0; j < NTTSIZE; j++) {
		temp = (uint64_t)op[gid] + (uint64_t)ip[j] * (uint64_t)ti[(tid * j) % NTTSIZE];
		op[gid] = temp % P;
	}
	temp = (uint64_t)op[gid] * (uint64_t)Ni;
	op[gid] = temp % P;
}
//====================================================
// ip : polynomial coefficients
// op : NTT transformed polynomial coefficients
// tf0 : forward twiddle factors (N)
// tf1 : forward twiddle factors (N1)
// tf2 : forward twiddle factors (N2)
//====================================================
void NTT_CT2(uint32_t *ip, uint32_t *op, uint32_t *tf0, uint32_t *tf1, uint32_t *tf2)
{
	int j1, j2, i1, i2;
	int Ncol = N1, Nrow = N2;
	uint64_t tmp1, tmp2, tmpX[NTTSIZE], temp;
	//col FFT
	for (j1 = 0; j1 < Ncol; j1++){
		for (i2 = 0; i2 < Nrow; i2++){
			tmp1 = 0;
			for (i1 = 0; i1 < Ncol; i1++){			
				tmp2 = (uint64_t) ip[Nrow * i1 + i2] * (uint64_t)tf1[(i1 * j1) % Ncol];
				tmp2 = (uint64_t)tmp2 % (uint64_t)P;

				tmp1 += tmp2;
				tmp1 = (uint64_t) tmp1 % (uint64_t)P;
			}
			tmpX[Nrow * j1 + i2] = tmp1;
		}
	}

	//mul fac
	for (j1 = 0; j1 < Ncol; j1++){
		for (i2 = 0; i2 < Nrow; i2++){
			temp = (uint64_t)tmpX[Nrow * j1 + i2] * (uint64_t)tf0[(i2 * j1) % NTTSIZE];
			tmpX[Nrow * j1 + i2]= (uint64_t)temp % (uint64_t)P;
		}
	}

	//row FFT
	for (j2 = 0; j2 < Nrow; j2++){
		for (i1 = 0; i1 < Ncol; i1++){
			for (tmp1 = i2 = 0; i2 < Nrow; i2++){
				tmp2 = (uint64_t)tmpX[Nrow * i1 + i2] * (uint64_t)tf2[(i2 * j2) % Nrow];
				tmp2 = (uint64_t)tmp2 % (uint64_t)P;
				
				tmp1 += tmp2;
				tmp1 = (uint64_t)tmp1 % (uint64_t)P;
			}
			op[Ncol * j2 + i1] = tmp1;
		}
	}
}

void NTT_CT2_BATCH(uint32_t *ip, uint32_t *op, uint32_t *tf0, uint32_t *tf1, uint32_t *tf2)
{
	int j1, j2, i1, i2, b;
	int Ncol = N1, Nrow = N2;
	uint32_t *tmpX;
	uint64_t tmp1, tmp2, temp;

	tmpX = (uint32_t*)malloc(sizeof(uint32_t) * BATCH * NTTSIZE);

	for (b = 0; b < BATCH; b++)
	{
		//col FFT
		for (j1 = 0; j1 < Ncol; j1++) {
			for (i2 = 0; i2 < Nrow; i2++) {
				tmp1 = 0;
				for (i1 = 0; i1 < Ncol; i1++) {
					tmp2 = (uint64_t) ip[b * NTTSIZE + Nrow * i1 + i2] * (uint64_t) tf1[(i1 * j1) % Ncol];
					tmp2 %= P;

					tmp1 += tmp2;
					tmp1 %= P;

				}
				tmpX[b * NTTSIZE + Nrow * j1 + i2] = tmp1;
			}
		}

		//mul fac
		for (j1 = 0; j1 < Ncol; j1++) {
			for (i2 = 0; i2 < Nrow; i2++) {
				temp = (uint64_t)tmpX[b * NTTSIZE + Nrow * j1 + i2] * (uint64_t)tf0[(i2 * j1) % NTTSIZE];
				tmpX[b * NTTSIZE + Nrow * j1 + i2] = temp % P;
			}
		}

		//row FFT
		for (j2 = 0; j2 < Nrow; j2++) {
			for (i1 = 0; i1 < Ncol; i1++) {
				for (tmp1 = i2 = 0; i2 < Nrow; i2++) {
					tmp2 = (uint64_t)tmpX[b * NTTSIZE + Nrow * i1 + i2] * (uint64_t)tf2[(i2 * j2) % Nrow];
					tmp2 %= P;

					tmp1 += tmp2;
					tmp1 %= P;
				}
				op[b * NTTSIZE + Ncol * j2 + i1] = tmp1;
			}
		}
	}
}
//====================================================
// ip : polynomial coefficients
// op : NTT transformed polynomial coefficients
// ti0 : inverse twiddle factors (N)
// ti1 : inverse twiddle factors (N1)
// ti2 : inverse twiddle factors (N2)
//====================================================
__global__ void NTT_CT2_gpu(uint32_t *ip, uint32_t *op, uint32_t *tf0, uint32_t *tf1, uint32_t *tf2, uint32_t *tmpX)
{
	int j1, j2, i1, i2;
	int Ncol = N1, Nrow = N2;
	uint64_t tmp1, tmp2;
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;

	for (j1 = 0; j1 < Ncol; j1++) {
			for (tmp1 = i1 = 0; i1 < Ncol; i1++) {
				tmp2 = (uint64_t)ip[Nrow * i1 + bid * NTTSIZE + tid] * (uint64_t)tf1[(i1 * j1) % Ncol];
				tmp2 %= P;
				tmp1 += tmp2;
				tmp1 %= P;
			}
			tmpX[Nrow * j1 + bid * NTTSIZE + tid] = tmp1;
			//__syncthreads();
		}
	for (j1 = 0; j1 < Ncol; j1++) {
		tmp1 = (uint64_t)tmpX[Nrow * j1 + bid * NTTSIZE + tid] * (uint64_t)tf0[(tid * j1) % NTTSIZE];
		tmpX[Nrow * j1 + bid * NTTSIZE + tid] = tmp1% P;
		//__syncthreads();
	}
	for (j2 = 0; j2 < Nrow; j2++) {
		for (i1 = 0; i1 < Ncol; i1++) {
			for (tmp1 = i2 = 0; i2 < Nrow; i2++) {
				tmp2 = (uint64_t)tmpX[bid * NTTSIZE + Nrow * i1 + i2] * (uint64_t)tf2[(i2 * tid) % Nrow];
				tmp2 %= P;

				tmp1 += tmp2;
				tmp1 %= P;
			}
			op[bid * NTTSIZE + Ncol * tid + i1] = tmp1;
		}
	}
}

void INTT_CT2(uint32_t *ip, uint32_t *op, uint32_t *ti0, uint32_t *ti1, uint32_t *ti2, uint32_t Ni)
{
	int j1, j2, i1, i2;
	int Ncol = N1, Nrow = N2;
	uint64_t tmp1, tmp2, temp;
	uint32_t *tmpX;
	tmpX = (uint32_t*)malloc(sizeof(uint32_t) * BATCH * NTTSIZE);
	//col FFT
	for (j1 = 0; j1 < Ncol; j1++) {
		for (i2 = 0; i2 < Nrow; i2++) {
			for (tmp1 = i1 = 0; i1 < Ncol; i1++) {
				tmp2 = (uint64_t)ip[Nrow * i1 + i2] * (uint64_t)ti1[(i1 * j1) % Ncol];
				tmp2 %= P;
				tmp1 += tmp2;
				tmp1 %= P;
			}
			tmpX[Nrow * j1 + i2] = tmp1;
		}
	}

	//mul fac
	for (j1 = 0; j1 < Ncol; j1++) {
		for (i2 = 0; i2 < Nrow; i2++) {
			temp = (uint64_t)tmpX[Nrow * j1 + i2] * (uint64_t)ti0[(i2 * j1) % NTTSIZE];
			tmpX[Nrow * j1 + i2] = temp % P;
		}
	}

	//row FFT
	for (j2 = 0; j2 < Nrow; j2++) {
		for (i1 = 0; i1 < Ncol; i1++) {
			for (tmp1 = i2 = 0; i2 < Nrow; i2++) {
				tmp2 = (uint64_t)tmpX[Nrow * i1 + i2] * (uint64_t)ti2[(i2 * j2) % Nrow];
				tmp2 %= P;
				tmp1 += tmp2;
				tmp1 %= P;
			}
			op[Ncol * j2 + i1] = tmp1;
		}
	}

	//multiply by inverse of NTTSIZE, N^-1
	for (int i = 0; i < NTTSIZE; i++) {
		temp = (uint64_t)op[i] * (uint64_t)Ni;
		op[i] = temp % P;
	}
}

void INTT_CT2_BATCH(uint32_t *ip, uint32_t *op, uint32_t *ti0, uint32_t *ti1, uint32_t *ti2, uint32_t Ni)
{
	int j1, j2, i1, i2, b;
	int Ncol = N1, Nrow = N2;
	uint32_t *tmpX;
	uint64_t tmp1, tmp2, temp;

	tmpX = (uint32_t*)malloc(sizeof(uint32_t) * BATCH * NTTSIZE);

	for (b = 0; b < BATCH; b++)
	{
		//col FFT
		for (j1 = 0; j1 < Ncol; j1++) {
			for (i2 = 0; i2 < Nrow; i2++) {
				for (tmp1 = i1 = 0; i1 < Ncol; i1++) {
					tmp2 = (uint64_t)ip[b * NTTSIZE + Nrow * i1 + i2] * (uint64_t)ti1[(i1 * j1) % Ncol];
					tmp2 %= P;
					tmp1 += tmp2;
					tmp1 %= P;
				}
				tmpX[b * NTTSIZE + Nrow * j1 + i2] = tmp1;
			}
		}

		//mul fac
		for (j1 = 0; j1 < Ncol; j1++) {
			for (i2 = 0; i2 < Nrow; i2++) {
				temp = (uint64_t)tmpX[b * NTTSIZE + Nrow * j1 + i2] * (uint64_t)ti0[(i2 * j1) % NTTSIZE];
				tmpX[b * NTTSIZE + Nrow * j1 + i2] = temp % P;
			}
		}

		//row FFT
		for (j2 = 0; j2 < Nrow; j2++) {
			for (i1 = 0; i1 < Ncol; i1++) {
				for (tmp1 = i2 = 0; i2 < Nrow; i2++) {
					tmp2 = (uint64_t)tmpX[b * NTTSIZE + Nrow * i1 + i2] * (uint64_t)ti2[(i2 * j2) % Nrow];
					tmp2 %= P;

					tmp1 += tmp2;
					tmp1 %= P;
				}
				op[b * NTTSIZE + Ncol * j2 + i1] = tmp1;
			}
		}

		//multiply by inverse of NTTSIZE, N^-1
		for (int i = 0; i < NTTSIZE; i++) {
			temp = (uint64_t)op[b * NTTSIZE + i] * (uint64_t)Ni;
			op[b * NTTSIZE + i] = temp % P;
		}
	}
}

//====================================================
// ip : polynomial coefficients
// op : NTT transformed polynomial coefficients
// ti0 : inverse twiddle factors (N)
// ti1 : inverse twiddle factors (N1)
// ti2 : inverse twiddle factors (N2)
//====================================================
__global__ void INTT_CT2_gpu(uint32_t *ip, uint32_t *op, uint32_t *ti0, uint32_t *ti1, uint32_t *ti2, uint32_t *tmpX, uint32_t Ni)
{
	int j1, i1, i2;
	int Ncol = N1, Nrow = N2;
	uint64_t tmp1, tmp2, temp;
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	//col FFT
	for (j1 = 0; j1 < Ncol; j1++) {
		for (tmp1 = i1 = 0; i1 < Ncol; i1++) {
			tmp2 = (uint64_t)ip[bid * NTTSIZE + Nrow * i1 + tid] * (uint64_t)ti1[(i1 * j1) % Ncol];
			tmp2 %= P;
			tmp1 += tmp2;
			tmp1 %= P;
		}
		tmpX[bid * NTTSIZE + Nrow * j1 + tid] = tmp1;		
	}
	for (j1 = 0; j1 < Ncol; j1++) {
		tmp1 = (uint64_t)tmpX[bid * NTTSIZE + Nrow * j1 + tid] * (uint64_t)ti0[(tid * j1) % NTTSIZE];
		tmpX[bid * NTTSIZE + Nrow * j1 + tid] = tmp1% P;
		//__syncthreads();
	}
	for (i1 = 0; i1 < Ncol; i1++) {
		for (tmp1 = i2 = 0; i2 < Nrow; i2++) {
			tmp2 = (uint64_t)tmpX[bid * NTTSIZE + Nrow * i1 + i2] * (uint64_t)ti2[(i2 * tid) % Nrow];
			tmp2 %= P;

			tmp1 += tmp2;
			tmp1 %= P;
		}
		op[bid * NTTSIZE + Ncol * tid + i1] = tmp1;
	}
	__syncthreads();
	//multiply by inverse of NTTSIZE, N^-1
	for (int i = 0; i < N1; i++) {
		temp = (uint64_t)op[bid * NTTSIZE + i * N2 + tid] * (uint64_t)Ni;
		op[bid * NTTSIZE + i * N2 + tid] = temp % P;
	}
}

__global__ void GS_radix2NTT_gpu0(uint32_t* ip, uint32_t* twiddleFactor, uint32_t lvl)
{
	uint32_t op1, op2, temp, b, tid;
	uint32_t stride = 0, m = 0;
	b = blockIdx.x;
	tid = threadIdx.x;
	uint32_t repeat = NTTSIZE / blockDim.x;

	//__shared__ uint32_t s_tf[NTTSIZE];
	//__shared__ uint32_t s_Phi_gpu[NTTSIZE];

	//for (int i = 0; i < repeat; i++)
	//{
	//	s_tf[i * blockDim.x + tid] = twiddleFactor[i * blockDim.x + tid];
	//	s_Phi_gpu[i * blockDim.x + tid] = Phi_gpu[i * blockDim.x + tid];
	//}
	//for (int i = 0; i < repeat; i++)
	//	ip[i * blockDim.x + b * NTTSIZE + tid] = barrett_red(ip[i * blockDim.x + b * NTTSIZE + tid] * (uint64_t)Phi_gpu[i * blockDim.x + tid]);
	//__syncthreads();
	m = NTTSIZE >> lvl;
	stride = 1 << lvl;
	for (int k = 0; k < NTTSIZE; k = k + m) {
		op1 = _addModP(ip[b * NTTSIZE + k + tid ], ip[b * NTTSIZE + k + tid  + m / 2]);
		op2 = _subModP(ip[b * NTTSIZE + k + tid ], ip[b * NTTSIZE + k + tid  + m / 2]);
		op2 = barrett_red(op2 * (uint64_t)tf0_gpu[(tid * stride)]);
		ip[b * NTTSIZE + k + tid ] = op1;
		ip[b * NTTSIZE + k + tid  + m / 2] = op2;
	}
}

__global__ void GS_radix2NTT_gpu1(uint32_t* ip, uint32_t* twiddleFactor, uint32_t lvl)
{
	uint32_t op1, op2, temp, b, tid;
	uint32_t stride = 0, m = 0;
	b = blockIdx.x;
	tid = threadIdx.x;
	//uint32_t repeat = NTTSIZE / blockDim.x;

	//__shared__ uint32_t s_tf[NTTSIZE];

	//for (int i = 0; i < repeat; i++)
	//{
	//	s_tf[i * blockDim.x + tid] = twiddleFactor[i * blockDim.x + tid];
	//}
	m = NTTSIZE >> lvl;
	stride = 1 << lvl;
	for (int k = 0; k < NTTSIZE; k = k + m) {
		op1 = _addModP(ip[b * NTTSIZE + k + tid], ip[b * NTTSIZE + k + tid + m / 2]);
		op2 = _subModP(ip[b * NTTSIZE + k + tid], ip[b * NTTSIZE + k + tid + m / 2]);
		op2 = barrett_red(op2 * (uint64_t)tf0_gpu[(tid * stride)]);
		ip[b * NTTSIZE + k + tid] = op1;
		ip[b * NTTSIZE + k + tid + m / 2] = op2;
	}
}

__global__ void GS_radix2NTT_gpu2(uint32_t* ip, uint32_t* twiddleFactor, uint32_t lvl)
{
	uint32_t op1, op2, temp, tf, b, tid;
	uint32_t stride = 0, m = 0;
	b = blockIdx.x;
	m = NTTSIZE >> lvl;
	stride = 1 << lvl;
	tid = threadIdx.x * m;
	uint32_t repeat = NTTSIZE / blockDim.x;
	__shared__ uint32_t s_tf[NTTSIZE];

	//for (int i = 0; i < repeat; i++)
	//{
	//	s_tf[i * blockDim.x + tid] = twiddleFactor[i * blockDim.x + tid];
	//}
	//__syncthreads();
	for (int j = 0; j < m / 2; j++) {
		op1 = _addModP(ip[b * NTTSIZE + tid  + j], ip[b * NTTSIZE + tid  + j + m / 2]);
		op2 = _subModP(ip[b * NTTSIZE + tid  + j], ip[b * NTTSIZE + tid  + j + m / 2]);
		op2 = barrett_red((uint64_t)op2 * (uint64_t)tf0_gpu[(j * stride)]);
		ip[b * NTTSIZE + tid  + j] = op1;
		ip[b * NTTSIZE + tid  + j + m / 2] = op2;
	}
}

__global__ void GS_radix2INTT_gpu2(uint32_t* ip, uint32_t* twiddleFactor, uint32_t lvl)
{
	uint32_t op1, op2, temp, tf, b, tid;
	uint32_t stride = 0, m = 0;
	b = blockIdx.x;
	m = NTTSIZE >> lvl;
	stride = 1 << lvl;
	tid = threadIdx.x * m;
	uint32_t repeat = NTTSIZE / blockDim.x;
	__shared__ uint32_t s_tf[NTTSIZE];

	//for (int i = 0; i < repeat; i++)
	//{
	//	s_tf[i * blockDim.x + tid] = twiddleFactor[i * blockDim.x + tid];
	//}
	//__syncthreads();
	for (int j = 0; j < m / 2; j++) {
		op1 = _addModP(ip[b * NTTSIZE + tid + j], ip[b * NTTSIZE + tid + j + m / 2]);
		op2 = _subModP(ip[b * NTTSIZE + tid + j], ip[b * NTTSIZE + tid + j + m / 2]);
		op2 = barrett_red((uint64_t)op2 * (uint64_t)ti0_gpu[(j * stride)]);
		ip[b * NTTSIZE + tid + j] = op1;
		ip[b * NTTSIZE + tid + j + m / 2] = op2;
	}
}

void radix2NTTGS(uint32_t* ip, uint32_t* twiddleFactor)
{
	uint32_t op1, op2, temp, tf, b;
	int stride = 0, m = 0;

	for (b = 0; b < BATCH; b++) {
		for (int level = 0; level < 10; level++) {
			m = NTTSIZE >> level;
			stride = 1 << level;
			//printf("\nLevel %d: Radix-2 fft: --------------------------------------------\n", level );
			for (int k = 0; k < NTTSIZE; k = k + m) {
				for (int j = 0; j < m / 2; j++) {
					op1 = _addModP_cpu(ip[b * NTTSIZE + k + j], ip[b * NTTSIZE + k + j + m / 2]);
					op2 = _subModP_cpu(ip[b * NTTSIZE + k + j], ip[b * NTTSIZE + k + j + m / 2]);
					//op2 = _mulModP(op2, twiddleFactor[j * stride]);

					op2 = (uint64_t)op2 * (uint64_t)twiddleFactor[(j * stride) % NTTSIZE] % P;
					//printf("%u %u\t", k + j, k + j + m / 2);
					//printf("%u %u - %u\t", ip[k+j], ip[k+j + m / 2], j * stride);
					ip[b * NTTSIZE + k + j] = op1;
					ip[b * NTTSIZE + k + j + m / 2] = op2;
					//printf("[%u, %u]\n", op1, op2);
				}
			}
		}
	}
}
__global__ void NTTStock_gpu0(uint32_t* ip, uint32_t* twiddleFactor, uint32_t* op, uint32_t stride)
{
	uint32_t op1, op2, b, tid;
	b = blockIdx.x;
	tid = threadIdx.x;
	uint32_t repeat = NTTSIZE / blockDim.x;	
	//__shared__ uint32_t tf[NTTSIZE];
	//__shared__ uint32_t s_Phi_gpu[NTTSIZE];

	//for (int i = 0; i < repeat; i++)
	//{
	//	tf[i * blockDim.x + tid] = twiddleFactor[i * blockDim.x + tid];
	//	s_Phi_gpu[i * blockDim.x + tid] = Phi_gpu[i * blockDim.x + tid];
	//}
	for (int i = 0; i < repeat; i++)
		ip[i * blockDim.x + b * NTTSIZE + tid] = barrett_red(ip[i * blockDim.x + b * NTTSIZE + tid] * (uint64_t)Phi_gpu[i * blockDim.x + tid]);
	__syncthreads();

	for (int j = 0; j < stride; j++)
	{
		op1 = _addModP(ip[b * NTTSIZE + tid * stride + j], ip[b * NTTSIZE + tid * stride + j + NTTSIZE / 2]);
		op2 = _subModP(ip[b * NTTSIZE + tid * stride + j], ip[b * NTTSIZE + tid * stride + j + NTTSIZE / 2]);
		op2 = barrett_red(op2 * (uint64_t)tf0_gpu[(tid * stride)]);
		op[b * NTTSIZE + 2 * tid * stride + j] = op1;
		op[b * NTTSIZE + (2 * tid + 1) * stride + j] = op2;
	}
}


__global__ void NTTStock_gpu1(uint32_t* ip, uint32_t* twiddleFactor, uint32_t* op, uint32_t stride)
{
	uint32_t op1, op2, b, tid;
	b = blockIdx.x;
	tid = threadIdx.x;
	uint32_t repeat = NTTSIZE / blockDim.x;
	//__shared__ uint32_t tf[NTTSIZE];

	//for (int i = 0; i < repeat; i++)
	//	tf[i * blockDim.x + tid] = twiddleFactor[i * blockDim.x + tid];

	for (int j = 0; j < stride; j++)
	{
		op1 = _addModP(ip[b * NTTSIZE + tid * stride + j], ip[b * NTTSIZE + tid * stride + j + NTTSIZE / 2]);
		op2 = _subModP(ip[b * NTTSIZE + tid * stride + j], ip[b * NTTSIZE + tid * stride + j + NTTSIZE / 2]);
		op2 = barrett_red(op2 * (uint64_t)tf0_gpu[(tid * stride)]);
		op[b * NTTSIZE + 2 * tid * stride + j] = op1;
		op[b * NTTSIZE + (2 * tid + 1) * stride + j] = op2;
	}	
}

__global__ void NTTStock_gpu2(uint32_t* ip, uint32_t* twiddleFactor, uint32_t* op, uint32_t stride)
{
	uint32_t op1, op2, b, tid;
	b = blockIdx.x;
	tid = threadIdx.x;
	//uint32_t repeat = NTTSIZE / blockDim.x;
	//__shared__ uint32_t tf[NTTSIZE];

	//for (int i = 0; i < repeat; i++)
	//	tf[i * blockDim.x + tid] = twiddleFactor[i * blockDim.x + tid];
	for (int s = 0; s < NTTSIZE / (2 * stride); s++)
	{
		op1 = _addModP(ip[b * NTTSIZE + s * stride + tid], ip[b * NTTSIZE + s * stride + tid  + NTTSIZE / 2]);
		op2 = _subModP(ip[b * NTTSIZE + s * stride + tid ], ip[b * NTTSIZE + s * stride + tid  + NTTSIZE / 2]);
		op2 = barrett_red(op2 * (uint64_t)tf0_gpu[(s * stride)]);
		op[b * NTTSIZE + 2 * s * stride + tid ] = op1;
		op[b * NTTSIZE + (2 * s + 1) * stride + tid ] = op2;
	}	
}

__global__ void pointwise_mult(uint32_t* ip1, uint32_t* ip2, uint32_t* op)
{
	uint32_t bid = blockIdx.x;
	uint32_t tid = threadIdx.x;
	op[bid*NTTSIZE + tid] = barrett_red((uint64_t)ip1[bid * NTTSIZE + tid] * (uint64_t)ip2[bid * NTTSIZE + tid]);
}

void radix2NTTStock(uint32_t *ip, uint32_t *twiddleFactor, uint32_t* op)
{
	uint32_t op1, op2, temp, J, stride, stride2, k, b;
	for (b = 0; b < BATCH; b++) {
		for (int i = 0; i < NTTSIZE; i++)
			ip[b * NTTSIZE + i] = barrett_red_cpu(ip[b * NTTSIZE + i] * (uint64_t)Phi[i]);
			//ip[b * NTTSIZE + i] = (uint64_t)ip[b * NTTSIZE + i] * (uint64_t)Phi[i] % P;
		
		for (int i = 0; i < 10; i++) {
			stride = 1 << i;
			//printf("Level %d: Radix-2 fft: --------------------------------------------\n", NTTSIZE / (2 * stride));
			for (int j = 0; j < stride; j++)
			{				
				for (int s = 0; s < NTTSIZE / (2 * stride); s++)
				{					
					op1 = _addModP_cpu(ip[b*NTTSIZE + s * stride + j], ip[b * NTTSIZE + s * stride + j + NTTSIZE / 2]);
					op2 = _subModP_cpu(ip[b * NTTSIZE + s * stride + j], ip[b * NTTSIZE + s * stride + j + NTTSIZE / 2]);
					op2 = barrett_red_cpu(op2 * (uint64_t)twiddleFactor[(s * stride) % NTTSIZE]);
	
					op[b * NTTSIZE + 2 * s * stride + j] = op1;
					op[b * NTTSIZE + (2 * s + 1) * stride + j] = op2;
				}
			}
			
			for (k = 0; k < NTTSIZE; k++)
			{
				temp = ip[b * NTTSIZE + k];
				ip[b * NTTSIZE + k] = op[b * NTTSIZE + k];
				op[b * NTTSIZE + k] = temp;
			}
			//printf("\nintermediate result 1\n");
			//for (k = 0; k < NTTSIZE; k++)	printf("%u ", ip[k]);
			//printf("\nintermediate result 2\n");
			//for (k = 0; k < NTTSIZE; k++)	printf("%u ", op[k]);
			//printf("\n");
		}
	}
}

void radix2NTT(uint32_t* ip, uint32_t* twiddleFactor)
{
	uint32_t op1, op2, temp, tf, b;
	uint32_t k = 0, l = 0, s = 0, j = 0, m = 0;

	for (b = 0; b < BATCH; b++) {
		k = NTTSIZE / 2;
		for (l = 1; l < NTTSIZE; l = 2 * l) {
			//printf("\nLevel %d: Radix-2 fft: %d --------------------------------------------\n", l, k);
			for (s = 0; s < NTTSIZE; s = s + 2 * l) {
				for (j = 0; j < l; j++) {
					temp = (uint64_t)ip[b * NTTSIZE + j + l + s] * (uint64_t)twiddleFactor[j * k] % P;
					op1 = _addModP_cpu(ip[b * NTTSIZE + j + s], temp);
					op2 = _subModP_cpu(ip[b * NTTSIZE + j + s], temp);
					ip[b * NTTSIZE + j + l + s] = op2;
					ip[b * NTTSIZE + j + s] = op1;
				}
			}
			k = k >> 1;
		}
	}
}

__global__ void GS_radix2INTT_gpu0(uint32_t* ip, uint32_t* twiddleFactor, uint32_t level)
{
	uint32_t op1, op2, temp, tf, b, tid;
	int stride = 0, m = 0;
	b = blockIdx.x;
	tid = threadIdx.x;
	m = NTTSIZE >> level;
	stride = 1 << level;

	for (int k = 0; k < NTTSIZE; k = k + m) {
		op1 = _addModP(ip[b * NTTSIZE + k + tid], ip[b * NTTSIZE + k + tid + m / 2]);
		op2 = _subModP(ip[b * NTTSIZE + k + tid], ip[b * NTTSIZE + k + tid + m / 2]);
		op2 = (uint64_t)op2 * (uint64_t)ti0_gpu[(tid * stride) % NTTSIZE] % P;
		ip[b * NTTSIZE + k + tid] = op1;
		ip[b * NTTSIZE + k + tid + m / 2] = op2;
	}
}
void radix2INTTGS(uint32_t* ip, uint32_t* twiddleFactor, uint32_t Ni)
{
	uint32_t op1, op2, temp, tf, b;
	
	int stride = 0, m = 0;
	for (b = 0; b < BATCH; b++) {
		stride = 0, m = 0;
		for (int level = 0; level < 10; level++) {
			m = NTTSIZE >> level;
			stride = 1 << level;
			//printf("Level %d: Radix-2 fft: --------------------------------------------\n", level);
			for (int k = 0; k < NTTSIZE; k = k + m) {
				for (int j = 0; j < m / 2; j++) {
					op1 = _addModP_cpu(ip[b * NTTSIZE + k + j], ip[b * NTTSIZE + k + j + m / 2]);
					op2 = _subModP_cpu(ip[b * NTTSIZE + k + j], ip[b * NTTSIZE + k + j + m / 2]);
					//op2 = _mulModP(op2, twiddleFactor[j * stride]);
					op2 = (uint64_t)op2 * (uint64_t)twiddleFactor[(j * stride) % NTTSIZE] % P;
					//printf("%u\t", j * stride);
					//printf("[%u, %u](TF: %u) --> [%u, %u]\n", ip[k + j], ip[k + j + m / 2], twiddleFactor[j * stride], op1, op2);
					ip[b * NTTSIZE + k + j] = op1;
					ip[b * NTTSIZE + k + j + m / 2] = op2;
				}
			}
		}
	}
}

__global__ void INTTStock_gpu0(uint32_t* ip, uint32_t* twiddleFactor, uint32_t* op, uint32_t stride)
{
	uint32_t op1, op2, temp, J, k, b, tid;
	b = blockIdx.x;
	tid = threadIdx.x;
	//uint32_t repeat = NTTSIZE / blockDim.x;
	//__shared__ uint32_t tf[NTTSIZE];

	//for (int i = 0; i < repeat; i++)
	//	tf[i * blockDim.x + tid] = twiddleFactor[i * blockDim.x + tid];
	for (int j = 0; j < stride; j++)
	{
		op1 = _addModP(ip[b * NTTSIZE + tid * stride + j], ip[b * NTTSIZE + tid * stride + j + NTTSIZE / 2]);
		op2 = _subModP(ip[b * NTTSIZE + tid * stride + j], ip[b * NTTSIZE + tid * stride + j + NTTSIZE / 2]);
		op2 = barrett_red(op2 * (uint64_t)ti0_gpu[(tid * stride)]);

		op[b * NTTSIZE + 2 * tid * stride + j] = op1;
		op[b * NTTSIZE + (2 * tid + 1) * stride + j] = op2;
	}
}

__global__ void INTTStock_gpu1(uint32_t* ip, uint32_t* twiddleFactor, uint32_t* op, uint32_t stride)
{
	uint32_t op1, op2, temp, J, k, b, tid;
	b = blockIdx.x;
	tid = threadIdx.x;
	uint32_t repeat = NTTSIZE / blockDim.x;
	//__shared__ uint32_t tf[NTTSIZE];

	//for (int i = 0; i < repeat; i++)
	//	tf[i * blockDim.x + tid] = twiddleFactor[i * blockDim.x + tid];
	
	for (int s = 0; s < NTTSIZE / (2 * stride); s++)
	{
		op1 = _addModP(ip[b * NTTSIZE + s * stride + tid], ip[b * NTTSIZE + s * stride + tid + NTTSIZE / 2]);
		op2 = _subModP(ip[b * NTTSIZE + s * stride + tid], ip[b * NTTSIZE + s * stride + tid + NTTSIZE / 2]);
		op2 = barrett_red(op2 * (uint64_t)ti0_gpu[(s * stride)]);

		op[b * NTTSIZE + 2 * s * stride + tid ] = op1;
		op[b * NTTSIZE + (2 * s + 1) * stride + tid ] = op2;	
	}
}

__global__ void INTTStock_gpu2(uint32_t* ip, uint32_t* twiddleFactor, uint32_t* op, uint32_t stride)
{
	uint32_t op1, op2, b, tid;
	b = blockIdx.x;
	tid = threadIdx.x;
	uint32_t repeat = NTTSIZE / blockDim.x;
	//__shared__ uint32_t tf[NTTSIZE];
	//__shared__ uint32_t s_invPhi_gpu[NTTSIZE];

	//for (int i = 0; i < repeat; i++)
	//{
	//	tf[i * blockDim.x + tid] = twiddleFactor[i * blockDim.x + tid];
	//	s_invPhi_gpu[i * blockDim.x + tid] = invPhi_gpu[i * blockDim.x + tid];
	//}
	for (int s = 0; s < NTTSIZE / (2 * stride); s++)
	{
		op1 = _addModP(ip[b * NTTSIZE + s * stride + tid], ip[b * NTTSIZE + s * stride + tid + NTTSIZE / 2]);
		op2 = _subModP(ip[b * NTTSIZE + s * stride + tid], ip[b * NTTSIZE + s * stride + tid + NTTSIZE / 2]);
		op2 = barrett_red(op2 * (uint64_t)ti0_gpu[(s * stride) ]);

		op[b * NTTSIZE + 2 * s * stride + tid] = op1;
		op[b * NTTSIZE + (2 * s + 1) * stride + tid] = op2;
	}

	for (int i = 0; i < repeat; i++)
		ip[i * blockDim.x + b * NTTSIZE + tid] = barrett_red(op[i * blockDim.x + b * NTTSIZE + tid] * (uint64_t)invPhi_gpu[i * blockDim.x + tid]);
}

void radix2INTTStock(uint32_t* ip, uint32_t* twiddleFactor, uint32_t Ni, uint32_t* op, uint32_t* invPhi)
{
	uint32_t op1, op2, temp, J, stride, k, b;
	for (b = 0; b < BATCH; b++) {
		for (int i = 0; i < 10; i++) {
			stride = 1 << i;
			//printf("Level %d: Radix-2 fft: --------------------------------------------\n", i);
			for (int j = 0; j < stride; j++)
			{
				for (int s = 0; s < NTTSIZE / (2 * stride); s++)
				{
					op1 = _addModP_cpu(ip[b * NTTSIZE + s * stride + j], ip[b * NTTSIZE + s * stride + j + NTTSIZE / 2]);
					op2 = _subModP_cpu(ip[b * NTTSIZE + s * stride + j], ip[b * NTTSIZE + s * stride + j + NTTSIZE / 2]);
					op2 = barrett_red_cpu(op2 * (uint64_t)twiddleFactor[(s * stride) % NTTSIZE]);

					op[b * NTTSIZE + 2 * s * stride + j] = op1;
					op[b * NTTSIZE + (2 * s + 1) * stride + j] = op2;
				}
			}

			for (k = 0; k < NTTSIZE; k++)
			{
				temp = ip[b * NTTSIZE + k];
				ip[b * NTTSIZE + k] = op[b * NTTSIZE + k];
				op[b * NTTSIZE + k] = temp;
			}
		}

		for (int i = 0; i < NTTSIZE; i++)
		{
			ip[b * NTTSIZE + i] = barrett_red_cpu(ip[b * NTTSIZE + i] * (uint64_t)invPhi[i % NTTSIZE]);
		}
	}
}

__global__ void radix2INTT_gpu0(uint32_t* ip, uint32_t* twiddleFactor, uint32_t stride, uint32_t lvl)
{
	uint32_t op1, op2, temp, tf, b;
	uint32_t k = 0, s = 0, j = 0, m = 0, tid;
	b = blockIdx.x;	
	k = (NTTSIZE)>>lvl;
	tid = threadIdx.x* stride;	

	for (j = tid ; j < tid  + stride / 2; j++) {
		temp = barrett_red((uint64_t)ip[b * NTTSIZE + j + stride / 2] * (uint64_t)ti0_gpu[m * k]);
		op1 = _addModP(ip[b * NTTSIZE + j], temp);
		op2 = _subModP(ip[b * NTTSIZE + j], temp);
		ip[b * NTTSIZE + j + stride/2] = op2;
		ip[b * NTTSIZE + j] = op1;
		m++;
	}
}

__global__ void radix2INTT_gpu1(uint32_t* ip, uint32_t* twiddleFactor, uint32_t stride, uint32_t lvl)
{
	uint32_t op1, op2, temp, b;
	uint32_t k = 0, s = 0, j = 0, tid;

	b = blockIdx.x;
	tid = threadIdx.x;	
	k = NTTSIZE >> lvl;
	for (s = 0; s < NTTSIZE; s = s + 2 * stride) {
		temp = barrett_red((uint64_t)ip[b * NTTSIZE + tid + stride + s] * (uint64_t)ti0_gpu[tid * k]);
		op1 = _addModP(ip[b * NTTSIZE + tid  + s], temp);
		op2 = _subModP(ip[b * NTTSIZE + tid  + s], temp);

		ip[b * NTTSIZE + tid  + stride + s] = op2;
		ip[b * NTTSIZE + tid  + s] = op1;		
	}
}


__global__ void radix2INTT_gpu2(uint32_t* ip, uint32_t* twiddleFactor, uint32_t stride, uint32_t lvl)
{
	uint32_t op1, op2, temp, b;
	uint32_t k = 0, s = 0, j = 0, tid;
	uint32_t repeat = NTTSIZE / blockDim.x;

	b = blockIdx.x;
	tid = threadIdx.x;
	k = NTTSIZE >> lvl;
	for (s = 0; s < NTTSIZE; s = s + 2 * stride) {
		temp = barrett_red((uint64_t)ip[b * NTTSIZE + tid + stride + s] * (uint64_t)ti0_gpu[tid * k]);
		op1 = _addModP(ip[b * NTTSIZE + tid + s], temp);
		op2 = _subModP(ip[b * NTTSIZE + tid + s], temp);

		ip[b * NTTSIZE + tid + stride + s] = op2;
		ip[b * NTTSIZE + tid + s] = op1;
	}

	for (int i = 0; i < repeat; i++)
	{
		ip[b * NTTSIZE + i*blockDim.x + tid] = barrett_red((uint64_t)ip[b * NTTSIZE + i * blockDim.x + tid] * (uint64_t)invPhi_gpu[i * blockDim.x + tid]) ;
	}
}


__global__ void radix2NTT_gpu0(uint32_t* ip, uint32_t* twiddleFactor, uint32_t stride, uint32_t lvl)
{
	uint32_t op1, op2, temp, tf, b;
	uint32_t k = 0, s = 0, j = 0, m = 0, tid;
	b = blockIdx.x;
	k = (NTTSIZE) >> lvl;
	tid = threadIdx.x * stride;

	for (j = tid; j < tid + stride / 2; j++) {
		temp = barrett_red((uint64_t)ip[b * NTTSIZE + j + stride / 2] * (uint64_t)tf0_gpu[m * k]);
		op1 = _addModP(ip[b * NTTSIZE + j], temp);
		op2 = _subModP(ip[b * NTTSIZE + j], temp);
		ip[b * NTTSIZE + j + stride / 2] = op2;
		ip[b * NTTSIZE + j] = op1;
		m++;
	}
}

__global__ void radix2NTT_gpu1(uint32_t* ip, uint32_t* twiddleFactor, uint32_t stride, uint32_t lvl)
{
	uint32_t op1, op2, temp, b;
	uint32_t k = 0, s = 0, j = 0, tid;

	b = blockIdx.x;
	tid = threadIdx.x;
	k = NTTSIZE >> lvl;
	for (s = 0; s < NTTSIZE; s = s + 2 * stride) {
		temp = barrett_red((uint64_t)ip[b * NTTSIZE + tid + stride + s] * (uint64_t)tf0_gpu[tid * k]);
		op1 = _addModP(ip[b * NTTSIZE + tid + s], temp);
		op2 = _subModP(ip[b * NTTSIZE + tid + s], temp);

		ip[b * NTTSIZE + tid + stride + s] = op2;
		ip[b * NTTSIZE + tid + s] = op1;
	}
}


void radix2INTT(uint32_t* ip, uint32_t* twiddleFactor, uint32_t Ni)
{
	uint32_t op1, op2, temp, tf, b;
	uint32_t k = 0, l = 0, s = 0, j = 0, m = 0;
		
	for (b = 0; b < BATCH; b++) {
		k = NTTSIZE / 2;
		for (l = 1; l < NTTSIZE; l = 2 * l) {
			//printf("\nLevel %d: Radix-2 fft: %d --------------------------------------------\n", l, k);
			for (s = 0; s < NTTSIZE; s = s + 2* l) {
				for (j = 0; j < l; j++) {
					temp = (uint64_t)ip[b*NTTSIZE + j + l +s] * (uint64_t)twiddleFactor[j*k] % P;
					op1 = _addModP_cpu(ip[b * NTTSIZE + j + s], temp);
					op2 = _subModP_cpu(ip[b * NTTSIZE + j + s], temp);
					ip[b * NTTSIZE + j + l + s] = op2;
					ip[b * NTTSIZE + j + s] = op1;
				}
			}
			k = k >> 1;
		}
	}
}
void test_NTT_naive(uint32_t *x, uint32_t *y, uint32_t *z, uint32_t *X, uint32_t *Y, uint32_t *Z, uint32_t fg, uint32_t ig, uint32_t Ni)
{
	init_operands(x, y);
	for (int i = 0; i < NTTSIZE; i++) { z[i] = x[i]; Z[i] = y[i]; }
	printf("\n========================\n");
	printf("test_NTT_naive");
	printf("\n========================\n");
	printf("Operands:\n");
	printf("x: "); for (int i = 0; i < NTTSIZE; i++) printf("% 4llu ", x[i]); printf("\n");
	printf("y: "); for (int i = 0; i < NTTSIZE; i++) printf("% 4llu ", y[i]); printf("\n");

	NTT_naive(x, X, fg);
	NTT_naive(y, Y, fg);

	printf("\nNTT:\n");
	printf("X: "); for (int i = 0; i < NTTSIZE; i++) printf("% 4llu ", X[i]); printf("\n");
	printf("Y: "); for (int i = 0; i < NTTSIZE; i++) printf("% 4llu ", Y[i]); printf("\n");
	// clear the buffer, reuse it for checking the correctness of results.
	memset(x, 0, sizeof(uint32_t) * BATCH *NTTSIZE);
	memset(y, 0, sizeof(uint32_t) * BATCH *NTTSIZE);
	INTT_naive(X, x, ig, Ni);
	INTT_naive(Y, y, ig, Ni);

	printf("\nINTT:\n");
	printf("x: "); for (int i = 0; i < NTTSIZE; i++) printf("% 4llu ", x[i]); printf("\n");
	printf("y: "); for (int i = 0; i < NTTSIZE; i++) printf("% 4llu ", y[i]); printf("\n");
	// Check for correctness. The recovered x and y should be the same as the original one (backup in z and Z).
	for (int i = 0; i < NTTSIZE; i++) {
		if (z[i] != x[i] || Z[i] != y[i]) {
			printf("Incorrect result.\n");
			break;
		}
		if (i == NTTSIZE - 1) {
			printf("Identical.\n");
		}
	}
}

void test_NTT_precom(uint32_t *x, uint32_t *y, uint32_t *z, uint32_t *X, uint32_t *Y, uint32_t *Z, uint32_t *tf, uint32_t *ti, uint32_t Ni)
{
	init_operands(x, y);
	for (int i = 0; i < NTTSIZE; i++) { z[i] = x[i]; Z[i] = y[i]; }
	printf("\n========================\n");
	printf("test_NTT_precom");
	printf("\n========================\n");
	printf("Operands:\n");
	//printf("x: "); for (int i = 0; i < NTTSIZE; i++) printf("% 4llu ", x[i]); printf("\n");
	//printf("y: "); for (int i = 0; i < NTTSIZE; i++) printf("% 4llu ", y[i]); printf("\n");

	NTT_precom(x, X, tf);
	NTT_precom(y, Y, tf);
	printf("\nNTT:\n");
	// clear the buffer, reuse it for checking the correctness of results.
	memset(x, 0, sizeof(uint32_t) * BATCH *NTTSIZE);
	memset(y, 0, sizeof(uint32_t) * BATCH *NTTSIZE);

	INTT_precom(X, x, ti, Ni);
	INTT_precom(Y, y, ti, Ni);
	printf("\nINTT:\n");
	//printf("x: "); for (int i = 0; i < NTTSIZE; i++) printf("% 4llu ", x[i]); printf("\n");
	//printf("y: "); for (int i = 0; i < NTTSIZE; i++) printf("% 4llu ", y[i]); printf("\n");
	// Check for correctness. The recovered x and y should be the same as the original one (backup in z and Z).
	for (int i = 0; i < NTTSIZE; i++) {
		if (z[i] != x[i] || Z[i] != y[i]) {
			printf("Incorrect result at %d x: % 4llu % 4llu  y: % 4llu % 4llu.\n", i, z[i], x[i], Z[i], y[i]);
			break;
		}
		if (i == NTTSIZE - 1) {
			printf("Identical.\n");
		}
	}
}

void test_NTT_precompute_BATCH(uint32_t *x, uint32_t *y, uint32_t *z, uint32_t *X, uint32_t *Y, uint32_t *Z, uint32_t *tf, uint32_t *ti, uint32_t Ni)
{
	LARGE_INTEGER t0, t1, freq;
	QueryPerformanceFrequency(&freq);

	init_operands(x, y);
	for (int i = 0; i < BATCH *NTTSIZE; i++)
	{
		x[i] = x[i % NTTSIZE]; // use same input for different batches
		z[i] = x[i];			// copy this to z for future verification
		y[i] = y[i % NTTSIZE]; // use same input for different batches
		Z[i] = y[i];			// copy this to Z for future verification
		X[i] = 0; Y[i] = 0;
	}
	printf("\n========================\n");
	printf("test_NTT_precom_BATCH: Batch Size is %d", BATCH);
	printf("\n========================\n");

	QueryPerformanceCounter(&t0);
	NTT_precom_batch(x, X, tf);
	NTT_precom_batch(y, Y, tf);
	printf("\nNTT:\n");

	INTT_precom_batch(X, x, ti, Ni);
	INTT_precom_batch(Y, y, ti, Ni);
	printf("\nINTT****************:\n");

	QueryPerformanceCounter(&t1);
	printf("Performance CPU test_NTT_precompute_BATCH   : % 4f ms.\n", ((double)((t1.QuadPart - t0.QuadPart) * 1000) / freq.QuadPart));

	for (int i = 0; i < BATCH*NTTSIZE; i++) {
		if (z[i] != x[i] || Z[i] != y[i]) {
			printf("Incorrect result at %d x: % 4llu % 4llu  y: % 4llu % 4llu.\n", i, z[i], x[i], Z[i], y[i]);
			break;
		}
		if (i == BATCH*NTTSIZE - 1) {
			printf("Identical.\n");
		}
	}
}

void test_NTT_precom_gpu(uint32_t *x, uint32_t *y, uint32_t *z, uint32_t *X, uint32_t *Y, uint32_t *Z, uint32_t *tf, uint32_t *ti, uint32_t Ni)
{
	LARGE_INTEGER t0, t1, freq;
	QueryPerformanceFrequency(&freq);
	uint32_t *d_x, *d_X, *d_y, *d_Y, *d_tf, *d_ti;
	printf("\n========================\n");
	printf("test_NTT_precom_gpu: Batch Size is %d", BATCH);
	printf("\n========================\n");

	init_operands(x, y);

	for (int i = 0; i < BATCH *NTTSIZE; i++)
	{
		x[i] = x[i % NTTSIZE]; // use same input for different batches
		z[i] = x[i];			// copy this to z for future verification
		y[i] = y[i % NTTSIZE]; // use same input for different batches
		Z[i] = y[i];			// copy this to Z for future verification
		X[i] = 0; Y[i] = 0;
	}
	cudaMalloc((void **)&d_x, BATCH*NTTSIZE * sizeof(uint32_t));
	cudaMalloc((void **)&d_X, BATCH*NTTSIZE * sizeof(uint32_t));
	cudaMalloc((void **)&d_tf, NTTSIZE * sizeof(uint32_t));
	cudaMalloc((void **)&d_ti, NTTSIZE * sizeof(uint32_t));
	cudaMalloc((void **)&d_y, BATCH*NTTSIZE * sizeof(uint32_t));
	cudaMalloc((void **)&d_y, BATCH*NTTSIZE * sizeof(uint32_t));

	QueryPerformanceCounter(&t0);
	// Copy the operands (x and y) and twiddle factors to the DRAM of GPU
	cudaMemcpy(d_x, x, BATCH*NTTSIZE * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, BATCH*NTTSIZE * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_tf, tf, NTTSIZE * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ti, ti, NTTSIZE * sizeof(uint32_t), cudaMemcpyHostToDevice);

	NTT_precom_gpu << <BATCH, NTTSIZE >> >(d_x, d_X, d_tf);
	NTT_precom_gpu << <BATCH, NTTSIZE >> >(d_y, d_Y, d_tf);

	// //Reset the buffers to 0, for debugging purpose.
	//cudaMemset(d_x, 0, BATCH*NTTSIZE * sizeof(uint32_t));
	//cudaMemset(d_y, 0, BATCH*NTTSIZE * sizeof(uint32_t));
	//memset(x, 0, BATCH*NTTSIZE * sizeof(uint32_t));
	//memset(y, 0, BATCH*NTTSIZE * sizeof(uint32_t));
	printf("\nINTT:\n");
	INTT_precom_gpu << <BATCH, NTTSIZE >> >(d_X, d_x, d_ti, Ni);
	INTT_precom_gpu << <BATCH, NTTSIZE >> >(d_Y, d_y, d_ti, Ni);
	// Copy the results from the DRAM of GPU
	cudaMemcpy(x, d_x, BATCH*NTTSIZE * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(y, d_y, BATCH*NTTSIZE * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	QueryPerformanceCounter(&t1);
	printf("Performance GPU test_NTT_precom_gpu   : % 4f ms.\n", ((double)((t1.QuadPart - t0.QuadPart) * 1000) / freq.QuadPart));

	// Check for correctness. The recovered x and y should be the same as the original one (backup in z and Z).
	for (int i = 0; i < BATCH*NTTSIZE; i++) {
		if (z[i] != x[i] || Z[i] != y[i]) {
			printf("Incorrect result at %d x: % 4llu % 4llu  y: % 4llu % 4llu.\n", i, z[i], x[i], Z[i], y[i]);
			break;
		}
		if (i == BATCH*NTTSIZE - 1) {
			printf("Identical.\n");
		}
	}
}

void test_NTT_CT2(uint32_t *x, uint32_t *y, uint32_t *z, uint32_t *X, uint32_t *Y, uint32_t *Z, uint32_t *tf0, uint32_t *tf1, uint32_t *tf2, uint32_t *ti0, uint32_t *ti1, uint32_t *ti2, uint32_t Ni)
{
	init_operands(x, y);
	for (int i = 0; i < NTTSIZE; i++){ z[i] = x[i]; Z[i] = y[i]; }
	printf("\n========================\n");
	printf("test_NTT_CT2");
	printf("\n========================\n");
	printf("Operands:\n");
	//printf("x: "); for (int i = 0; i < NTTSIZE; i++) printf("% 4llu ", x[i]); printf("\n");
	//printf("y: "); for (int i = 0; i < NTTSIZE; i++) printf("% 4llu ", y[i]); printf("\n");

	NTT_CT2(x, X, tf0, tf1, tf2);
	NTT_CT2(y, Y, tf0, tf1, tf2);
	printf("\nNTT:\n");
	// clear the buffer, reuse it for checking the correctness of results.
	memset(x, 0, sizeof(uint32_t) * NTTSIZE);
	memset(y, 0, sizeof(uint32_t) * NTTSIZE);

	INTT_CT2(X, x, ti0, ti1, ti2, Ni);
	INTT_CT2(Y, y, ti0, ti1, ti2, Ni);
	printf("\nINTT:\n");

	for (int i = 0; i < NTTSIZE; i++){
		if (z[i] != x[i] || Z[i] != y[i]){
			printf("Incorrect result at %d % 4llu % 4llu % 4llu % 4llu.\n", i, x[i], z[i], y[i], Z[i]);
			break;
		}
		if (i == NTTSIZE - 1){
			printf("Identical.\n");
		}
	}
}


void test_NTT_CT2_BATCH(uint32_t *x, uint32_t *y, uint32_t *z, uint32_t *X, uint32_t *Y, uint32_t *Z, uint32_t *tf0, uint32_t *tf1, uint32_t *tf2, uint32_t *ti0, uint32_t *ti1, uint32_t *ti2, uint32_t Ni)
{
	LARGE_INTEGER t0, t1, freq;
	QueryPerformanceFrequency(&freq);

	init_operands(x, y);
	for (int i = 0; i < BATCH *NTTSIZE; i++) 
	{ 
		x[i] = x[i % NTTSIZE] ; // use same input for different batches
		z[i] = x[i];			// copy this to z for future verification
		y[i] = y[i % NTTSIZE]; // use same input for different batches
		Z[i] = y[i];			// copy this to Z for future verification
		X[i] = 0; Y[i] = 0;
	}
	printf("\n========================\n");
	printf("test_NTT_CT2_BATCH: Batch Size is %d", BATCH);
	printf("\n========================\n");
	
	QueryPerformanceCounter(&t0);
	NTT_CT2_BATCH(x, X, tf0, tf1, tf2);
	NTT_CT2_BATCH(y, Y, tf0, tf1, tf2);
	INTT_CT2_BATCH(X, x, ti0, ti1, ti2, Ni);
	INTT_CT2_BATCH(Y, y, ti0, ti1, ti2, Ni);
	QueryPerformanceCounter(&t1);
	printf("Performance CPU test_NTT_CT2_BATCH   : % 4f ms.\n", ((double)((t1.QuadPart - t0.QuadPart) * 1000) / freq.QuadPart));

	for (int i = 0; i < BATCH * NTTSIZE; i++) {
		if (z[i] != x[i] || Z[i] != y[i]) {
			printf("Incorrect result at %d x: % 4llu % 4llu  y: % 4llu % 4llu.\n", i, z[i], x[i], Z[i], y[i]);
			break;
		}
		if (i == BATCH * NTTSIZE - 1) {
			printf("Identical.\n");
		}
	}
}

void test_NTT_CT2_gpu(uint32_t *x, uint32_t *y, uint32_t *z, uint32_t *X, uint32_t *Y, uint32_t *Z, uint32_t *tf0, uint32_t *tf1, uint32_t *tf2, uint32_t *ti0, uint32_t *ti1, uint32_t *ti2, uint32_t Ni)
{
	LARGE_INTEGER t0, t1, freq;
	uint32_t *d_x, *d_X, *d_y, *d_Y, *d_tf0, *d_tf1, *d_tf2, *tmpX, *d_tmpX, *d_tmpY, *d_ti0, *d_ti1, *d_ti2;
	cudaMalloc((void **)&d_x, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMalloc((void **)&d_X, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMalloc((void **)&d_tf0, NTTSIZE * sizeof(uint32_t));
	cudaMalloc((void **)&d_tf1, N1 * sizeof(uint32_t));
	cudaMalloc((void **)&d_tf2, N2 * sizeof(uint32_t));
	cudaMalloc((void **)&d_ti0, NTTSIZE * sizeof(uint32_t));
	cudaMalloc((void **)&d_ti1, N1 * sizeof(uint32_t));
	cudaMalloc((void **)&d_ti2, N2 * sizeof(uint32_t));
	cudaMalloc((void **)&d_y, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMalloc((void **)&d_Y, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMallocHost((void **)&tmpX, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMalloc((void **)&d_tmpX, BATCH *NTTSIZE * sizeof(uint32_t));
	cudaMalloc((void **)&d_tmpY, BATCH *NTTSIZE * sizeof(uint32_t));
	cudaMemset(d_tmpX, 0, BATCH *NTTSIZE * sizeof(uint32_t));
	cudaMemset(d_tmpY, 0, BATCH *NTTSIZE * sizeof(uint32_t));
	QueryPerformanceFrequency(&freq);

	init_operands(x, y);
	for (int i = 0; i < BATCH *NTTSIZE; i++)
	{
		x[i] = x[i % NTTSIZE]; // use same input for different batches
		z[i] = x[i];			// copy this to z for future verification
		y[i] = y[i % NTTSIZE]; // use same input for different batches
		Z[i] = y[i];			// copy this to Z for future verification
		X[i] = 0; Y[i] = 0;
	}
	printf("\n========================\n");
	printf("test_NTT_CT2_BATCH32_gpu: Batch Size is %d", BATCH);
	printf("\n========================\n");

	QueryPerformanceCounter(&t0);
	cudaMemcpy(d_x, x, BATCH *NTTSIZE * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, BATCH *NTTSIZE * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_tf0, tf0, NTTSIZE * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_tf1, tf1, N1 * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_tf2, tf2, N2 * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ti0, ti0, NTTSIZE * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ti1, ti1, N1 * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ti2, ti2, N2 * sizeof(uint32_t), cudaMemcpyHostToDevice);

	// NTT for X operand
	NTT_CT2_gpu << <BATCH, N2 >> >(d_x, d_X, d_tf0, d_tf1, d_tf2, d_tmpX);
	// NTT for Y operand
	NTT_CT2_gpu << <BATCH, N2 >> >(d_y, d_Y, d_tf0, d_tf1, d_tf2, d_tmpY);

	// Reset values to 0, reuse the same arrays.
	cudaMemset(d_tmpX, 0, BATCH *NTTSIZE * sizeof(uint32_t));
	cudaMemset(d_tmpY, 0, BATCH *NTTSIZE * sizeof(uint32_t));
	cudaMemset(d_x, 0, BATCH *NTTSIZE * sizeof(uint32_t));
	cudaMemset(d_y, 0, BATCH *NTTSIZE * sizeof(uint32_t));
	memset(x, 0, BATCH *NTTSIZE * sizeof(uint32_t));	
	memset(y, 0, BATCH *NTTSIZE * sizeof(uint32_t));
	
	INTT_CT2_gpu << <BATCH, N2 >> >(d_X, d_x, d_ti0, d_ti1, d_ti2, d_tmpX, Ni);
	INTT_CT2_gpu << <BATCH, N2 >> >(d_Y, d_y, d_ti0, d_ti1, d_ti2, d_tmpX, Ni);
	cudaMemcpy(x, d_x, BATCH * NTTSIZE * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(y, d_y, BATCH * NTTSIZE * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	QueryPerformanceCounter(&t1);
	printf("Performance GPU test_NTT_CT2_gpu   : % 4f ms.\n", ((double)((t1.QuadPart - t0.QuadPart) * 1000) / freq.QuadPart));
	printf("\nINTT:\n");
	// Check for correctness. The recovered x and y should be the same as the original one (backup in z and Z).
	for (int i = 0; i < BATCH * NTTSIZE; i++) {
		if (z[i] != x[i] || Z[i] != y[i]) {
		//if(z[i] != x[i]){
			printf("Incorrect result at %d x: % 4llu % 4llu  y: % 4llu % 4llu.\n", i, z[i], x[i], Z[i], y[i]);
			break;
		}
		if (i == BATCH * NTTSIZE - 1) {
			printf("Identical.\n");
		}
	}
}


void test_NTT_GS_CT_BATCH(uint32_t* x, uint32_t* y, uint32_t* z, uint32_t* X, uint32_t* Y, uint32_t* Z, uint32_t* tf0, uint32_t* ti0, uint32_t nfg0, uint32_t nig0, uint32_t Ni)
{
	for (int i = 0; i < BATCH * NTTSIZE; i++) { x[i] = 1; y[i] = 1; }

	printf("\n========================\n");
	printf("test_NTT_negacyclic GS-CT");
	printf("\n========================\n");

	for (int i = 0; i < BATCH * NTTSIZE; i++)
	{
		x[i] = (uint64_t)x[i] * ((uint64_t)Phi[i % NTTSIZE] % P);
		y[i] = (uint64_t)y[i] * ((uint64_t)Phi[i % NTTSIZE] % P);
	}

	radix2NTTGS(x, tf0);
	radix2NTTGS(y, tf0);

	//printf("\nAfter NTT\n");
	//printf("X: "); for (int i = 0; i < BATCH * NTTSIZE; i++) {
	//	if (i % NTTSIZE == 0) printf("\n\n");
	//	printf("%u ", x[i]);		
	//}
	//printf("Y: "); for (int i = 0; i < NTTSIZE; i++) printf("%u ", y[i]); printf("\n");
	for (int i = 0; i < BATCH * NTTSIZE; i++)
		z[i] = ((uint64_t)x[i] * (uint64_t)y[i]) % P;

	radix2INTT(z, ti0, Ni);
	for (int i = 0; i < BATCH * NTTSIZE; i++)
	{
		z[i] = ((uint64_t)z[i] * (uint64_t)invPhi[i % NTTSIZE]) % P;
	}

	printf("z: "); for (int i = 0; i < BATCH * NTTSIZE; i++)
	{
		if (i % NTTSIZE == 0) printf("\n\n");
		printf("%u ", z[i]); 
	}
}


void test_NTT_nega_GS(uint32_t* x, uint32_t* y, uint32_t* z, uint32_t* X, uint32_t* Y, uint32_t* Z, uint32_t* tf0, uint32_t* ti0, uint32_t nfg0, uint32_t nig0, uint32_t Ni)
{
	for (int i = 0; i < BATCH * NTTSIZE; i++) { x[i] = 1; y[i] = 1; }
	printf("\n========================\n");
	printf("test_NTT_negacyclic CT");
	printf("\n========================\n");
	for (int i = 0; i < BATCH * NTTSIZE; i++)
	{
		x[i] = (uint64_t)x[i] * (uint64_t)Phi[i%NTTSIZE] % P;
		y[i] = (uint64_t)y[i] * (uint64_t)Phi[i % NTTSIZE] % P;
	}

	//printf("\nOperands:\n");
	//printf("x: "); for (int i = 0; i < NTTSIZE; i++) printf("%u ", x[i]); printf("\n");
	//printf("y: "); for (int i = 0; i < NTTSIZE; i++) printf("%u ", y[i]); printf("\n");

	radix2NTTGS(x, tf0);
	radix2NTTGS(y, tf0);
	bit_reverse_copy(x, X);
	bit_reverse_copy(y, Y);
	
	//printf("\nAfter bit-reversing\n");
	//printf("X: "); for (int i = 0; i < NTTSIZE; i++) printf("%u ", X[i]); printf("\n");
	//printf("Y: "); for (int i = 0; i < NTTSIZE; i++) printf("%u ", Y[i]); printf("\n");
	for (int i = 0; i < BATCH * NTTSIZE; i++)
		Z[i] = ((uint64_t)X[i] * (uint64_t)Y[i]) % P;

	radix2INTTGS(Z, ti0, Ni);
	bit_reverse_copy(Z, z);

	//printf("\nAfter INTT:\n"); 	
	//for (int i = 0; i < BATCH * NTTSIZE; i++)
	//{
	//	if (i % NTTSIZE == 0) printf("\n\n");
	//	printf("%u ", z[i]);
	//}
	for (int i = 0; i < BATCH*NTTSIZE; i++)
	{
		z[i] = (uint64_t)z[i] * (uint64_t)invPhi[i%NTTSIZE] % P;
	}

	printf("z: "); for (int i = 0; i < BATCH * NTTSIZE; i++)
	{
		if (i % NTTSIZE == 0) printf("\n\n");
		printf("%u ", z[i]);
	}
}

void test_NTT_nega_CT(uint32_t* x, uint32_t* y, uint32_t* z, uint32_t* X, uint32_t* Y, uint32_t* Z, uint32_t* tf0, uint32_t* ti0, uint32_t nfg0, uint32_t nig0, uint32_t Ni)
{
	for (int i = 0; i < BATCH * NTTSIZE; i++) { x[i] = 1; y[i] = 1; }
	printf("\n========================\n");
	printf("test_NTT_negacyclic GS");
	printf("\n========================\n");
	for (int i = 0; i < BATCH * NTTSIZE; i++)
	{
		x[i] = (uint64_t)x[i] * (uint64_t)Phi[i % NTTSIZE] % P;
		y[i] = (uint64_t)y[i] * (uint64_t)Phi[i % NTTSIZE] % P;
	}

	//printf("\nOperands:\n");
	//printf("x: "); for (int i = 0; i < NTTSIZE; i++) printf("%u ", x[i]); printf("\n");
	//printf("y: "); for (int i = 0; i < NTTSIZE; i++) printf("%u ", y[i]); printf("\n");
	bit_reverse_copy(x, X);
	bit_reverse_copy(y, Y);
	radix2NTT(X, tf0);
	radix2NTT(Y, tf0);

	//printf("\nAfter bit-reversing\n");
	//printf("X: "); for (int i = 0; i < NTTSIZE; i++) printf("%u ", X[i]); printf("\n");
	//printf("Y: "); for (int i = 0; i < NTTSIZE; i++) printf("%u ", Y[i]); printf("\n");
	for (int i = 0; i < BATCH * NTTSIZE; i++)
		Z[i] = ((uint64_t)X[i] * (uint64_t)Y[i]) % P;
	
	bit_reverse_copy(Z, z);
	radix2INTT(z, ti0, Ni);

	//printf("\nAfter INTT:\n"); 	
	//for (int i = 0; i < BATCH * NTTSIZE; i++)
	//{
	//	if (i % NTTSIZE == 0) printf("\n\n");
	//	printf("%u ", z[i]);
	//}
	for (int i = 0; i < BATCH * NTTSIZE; i++)
	{
		z[i] = (uint64_t)z[i] * (uint64_t)invPhi[i % NTTSIZE] % P;
	}

	printf("z: "); for (int i = 0; i < BATCH * NTTSIZE; i++)
	{
		if (i % NTTSIZE == 0) printf("\n\n");
		printf("%u ", z[i]);
	}
}

void test_NTT_Stockham_nega(uint32_t* x, uint32_t* y, uint32_t* z, uint32_t* X, uint32_t* Y, uint32_t* Z, uint32_t* tf0, uint32_t* ti0, uint32_t nfg0, uint32_t nig0, uint32_t Ni)
{	
	for (int i = 0; i < BATCH * NTTSIZE; i++) { x[i] = 1; y[i] = 1; }
	printf("\n========================\n");
	printf("test_NTT_negacyclic Stockham");
	printf("\n========================\n");

	//printf("\nOperands:\n");
	//printf("x: "); for (int i = 0; i < NTTSIZE; i++) printf("%u ", x[i]); printf("\n");
	//printf("y: "); for (int i = 0; i < NTTSIZE; i++) printf("%u ", y[i]); printf("\n");

	radix2NTTStock(x, tf0, X);
	radix2NTTStock(y, tf0, Y);
	//printf("X: "); for (int i = 0; i < BATCH * NTTSIZE; i++)
	//{
	//	if (i % NTTSIZE == 0) printf("\n\n");
	//	printf("%u ", x[i]);
	//}

	for (int i = 0; i < BATCH * NTTSIZE; i++)
		Z[i] = ((uint64_t)x[i] * (uint64_t)y[i]) % P;

	radix2INTTStock(Z, ti0, Ni, z, invPhi);
	
	printf("z: "); for (int i = 0; i < BATCH * NTTSIZE; i++)
	{
		if (i % NTTSIZE == 0) printf("\n\n");
		printf("%u ", Z[i]);
	}
}


void test_nussbaumer(uint32_t* x, uint32_t* y, uint32_t* z, uint32_t* X, uint32_t* Y, uint32_t* Z)
{
	for (int i = 0; i < BATCH * NTTSIZE; i++) { x[i] = 1; y[i] = 1; }
	printf("\n========================\n");
	printf("test_nussbaumer");
	printf("\n========================\n");

	nussbaumer_fft(z, x, y);

	//printf("\nOperands:\n");
	//printf("x: "); for (int i = 0; i < NTTSIZE; i++) printf("%u ", x[i]); printf("\n");
	//printf("y: "); for (int i = 0; i < NTTSIZE; i++) printf("%u ", y[i]); printf("\n");

	printf("z: "); for (int i = 0; i < BATCH * NTTSIZE; i++)
	{
		if (i % NTTSIZE == 0) printf("\n\n");
		printf("%u ", z[i]);
	}
}


void test_NTT_Stockham_nega_gpu(uint32_t* x, uint32_t* y, uint32_t* z, uint32_t* X, uint32_t* Y, uint32_t* Z, uint32_t* tf0, uint32_t* ti0, uint32_t nfg0, uint32_t nig0, uint32_t Ni)
{
	for (int i = 0; i < BATCH * NTTSIZE; i++) { x[i] = 1; y[i] = 1; }
	printf("\n========================\n");
	printf("test_NTT_negacyclic Stockham GPU. Batch Size is %d", BATCH);
	printf("\n========================\n");

	LARGE_INTEGER t0, t1, freq;
	uint32_t* d_x, * d_X, * d_y, * d_Y, * d_tf0, *d_Z, *d_ti0;
	double elapsed = 0.0;

	cudaMalloc((void**)& d_x, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMalloc((void**)& d_X, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMalloc((void**)& d_tf0, NTTSIZE * sizeof(uint32_t));
	cudaMalloc((void**)& d_ti0, NTTSIZE * sizeof(uint32_t));
	cudaMalloc((void**)& d_y, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMalloc((void**)& d_Y, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMalloc((void**)& d_Z, BATCH * NTTSIZE * sizeof(uint32_t));
	
	cudaMemset(d_x, 0, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMemset(d_y, 0, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMemset(d_X, 0, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMemset(d_Y, 0, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMemset(d_Z, 0, BATCH * NTTSIZE * sizeof(uint32_t));
	QueryPerformanceFrequency(&freq);

	for (int i = 0; i < NUM_AVE; i++)
	{
		QueryPerformanceCounter(&t0);
		cudaMemcpy(d_x, x, BATCH * NTTSIZE * sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_y, y, BATCH * NTTSIZE * sizeof(uint32_t), cudaMemcpyHostToDevice);

		NTTStock_gpu0 << <BATCH, 512 >> > (d_x, d_tf0, d_X, 1);
		NTTStock_gpu1 << <BATCH, 256 >> > (d_X, d_tf0, d_x, 2);
		NTTStock_gpu1 << <BATCH, 128 >> > (d_x, d_tf0, d_X, 4);
		NTTStock_gpu1 << <BATCH, 64 >> > (d_X, d_tf0, d_x, 8);
		NTTStock_gpu1 << <BATCH, 32 >> > (d_x, d_tf0, d_X, 16);
		NTTStock_gpu2 << <BATCH, 32 >> > (d_X, d_tf0, d_x, 32);
		NTTStock_gpu2 << <BATCH, 64 >> > (d_x, d_tf0, d_X, 64);
		NTTStock_gpu2 << <BATCH, 128 >> > (d_X, d_tf0, d_x, 128);
		NTTStock_gpu2 << <BATCH, 256 >> > (d_x, d_tf0, d_X, 256);
		NTTStock_gpu2 << <BATCH, 512 >> > (d_X, d_tf0, d_x, 512);

		NTTStock_gpu0 << <BATCH, 512 >> > (d_y, d_tf0, d_Y, 1);
		NTTStock_gpu1 << <BATCH, 256 >> > (d_Y, d_tf0, d_y, 2);
		NTTStock_gpu1 << <BATCH, 128 >> > (d_y, d_tf0, d_Y, 4);
		NTTStock_gpu1 << <BATCH, 64 >> > (d_Y, d_tf0, d_y, 8);
		NTTStock_gpu1 << <BATCH, 32 >> > (d_y, d_tf0, d_Y, 16);
		NTTStock_gpu2 << <BATCH, 32 >> > (d_Y, d_tf0, d_y, 32);
		NTTStock_gpu2 << <BATCH, 64 >> > (d_y, d_tf0, d_Y, 64);
		NTTStock_gpu2 << <BATCH, 128 >> > (d_Y, d_tf0, d_y, 128);
		NTTStock_gpu2 << <BATCH, 256 >> > (d_y, d_tf0, d_Y, 256);
		NTTStock_gpu2 << <BATCH, 512 >> > (d_Y, d_tf0, d_y, 512);

		cudaMemcpy(x, d_x, BATCH * NTTSIZE * sizeof(uint32_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(y, d_y, BATCH * NTTSIZE * sizeof(uint32_t), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();

		pointwise_mult << <BATCH, NTTSIZE >> > (d_x, d_y, d_Z);

		INTTStock_gpu0 << <BATCH, 512 >> > (d_Z, d_ti0, d_x, 1);
		INTTStock_gpu0 << <BATCH, 256 >> > (d_x, d_ti0, d_Z, 2);
		INTTStock_gpu0 << <BATCH, 128 >> > (d_Z, d_ti0, d_x, 4);
		INTTStock_gpu0 << <BATCH, 64 >> > (d_x, d_ti0, d_Z, 8);
		INTTStock_gpu0 << <BATCH, 32 >> > (d_Z, d_ti0, d_x, 16);
		INTTStock_gpu1 << <BATCH, 32 >> > (d_x, d_ti0, d_Z, 32);
		INTTStock_gpu1 << <BATCH, 64 >> > (d_Z, d_ti0, d_x, 64);
		INTTStock_gpu1 << <BATCH, 128 >> > (d_x, d_ti0, d_Z, 128);
		INTTStock_gpu1 << <BATCH, 256 >> > (d_Z, d_ti0, d_x, 256);
		INTTStock_gpu2 << <BATCH, 512 >> > (d_x, d_ti0, d_Z, 512);
		cudaMemcpy(Z, d_x, BATCH * NTTSIZE * sizeof(uint32_t), cudaMemcpyDeviceToHost);
		QueryPerformanceCounter(&t1);
		elapsed += ((double)((t1.QuadPart - t0.QuadPart) * 1000) / freq.QuadPart);
	}
	
	printf("Performance GPU Stockham GPU \n Time\t\t: % .4f ms. \nThroughput\t: %.2f Multiplications per second\n", elapsed/ NUM_AVE, (BATCH* NUM_AVE /elapsed) * 1000);
#ifdef DEBUG
	printf("z: "); for (int i = 0; i < BATCH * NTTSIZE; i++)
	{
		if (i % NTTSIZE == 0) printf("\n\n");
		printf("%u ", Z[i]);
	}
#endif
	cudaFree(d_x);	cudaFree(d_X);	
	cudaFree(d_y);	cudaFree(d_Y);
	cudaFree(d_Z);
}


void test_NTT_GS_CT_nega_gpu(uint32_t* x, uint32_t* y, uint32_t* z, uint32_t* X, uint32_t* Y, uint32_t* Z, uint32_t* tf0, uint32_t* ti0, uint32_t nfg0, uint32_t nig0, uint32_t Ni)
{
	for (int i = 0; i < BATCH * NTTSIZE; i++) { x[i] = 1; y[i] = 1; }
	printf("\n========================\n");
	printf("test_NTT_negacyclic GS-CT GPU. Batch Size is %d", BATCH);
	printf("\n========================\n");

	LARGE_INTEGER t0, t1, freq;
	uint32_t* d_x, * d_X, * d_y, * d_Y, * d_tf0, * d_Z, * d_ti0;
	double elapsed = 0.0;

	cudaMalloc((void**)& d_x, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMalloc((void**)& d_X, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMalloc((void**)& d_y, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMalloc((void**)& d_Y, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMalloc((void**)& d_Z, BATCH * NTTSIZE * sizeof(uint32_t));

	cudaMemset(d_x, 0, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMemset(d_y, 0, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMemset(d_X, 0, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMemset(d_Y, 0, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMemset(d_Z, 0, BATCH * NTTSIZE * sizeof(uint32_t));
	QueryPerformanceFrequency(&freq);

	for (int i = 0; i < NUM_AVE; i++)
	{
		QueryPerformanceCounter(&t0);
		cudaMemcpy(d_x, x, BATCH * NTTSIZE * sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_y, y, BATCH * NTTSIZE * sizeof(uint32_t), cudaMemcpyHostToDevice);
			
		GS_radix2NTT_gpu0 << <BATCH, 512 >> > (d_x, d_tf0, 0);
		GS_radix2NTT_gpu1 << <BATCH, 256 >> > (d_x, d_tf0, 1);
		GS_radix2NTT_gpu1 << <BATCH, 128 >> > (d_x, d_tf0, 2);
		GS_radix2NTT_gpu1 << <BATCH, 64 >> > (d_x, d_tf0, 3);
		GS_radix2NTT_gpu1 << <BATCH, 32 >> > (d_x, d_tf0, 4);
		GS_radix2NTT_gpu2 << <BATCH, 32 >> > (d_x, d_tf0, 5);
		GS_radix2NTT_gpu2 << <BATCH, 64 >> > (d_x, d_tf0, 6);
		GS_radix2NTT_gpu2 << <BATCH, 128 >> > (d_x, d_tf0, 7);
		GS_radix2NTT_gpu2 << <BATCH, 256 >> > (d_x, d_tf0, 8);
		GS_radix2NTT_gpu2 << <BATCH, 512 >> > (d_x, d_tf0, 9);

		GS_radix2NTT_gpu0 << <BATCH, 512 >> > (d_y, d_tf0, 0);
		GS_radix2NTT_gpu1 << <BATCH, 256 >> > (d_y, d_tf0, 1);
		GS_radix2NTT_gpu1 << <BATCH, 128 >> > (d_y, d_tf0, 2);
		GS_radix2NTT_gpu1 << <BATCH, 64 >> > (d_y, d_tf0, 3);
		GS_radix2NTT_gpu1 << <BATCH, 32 >> > (d_y, d_tf0, 4);
		GS_radix2NTT_gpu2 << <BATCH, 32 >> > (d_y, d_tf0, 5);
		GS_radix2NTT_gpu2 << <BATCH, 64 >> > (d_y, d_tf0, 6);
		GS_radix2NTT_gpu2 << <BATCH, 128 >> > (d_y, d_tf0, 7);
		GS_radix2NTT_gpu2 << <BATCH, 256 >> > (d_y, d_tf0, 8);
		GS_radix2NTT_gpu2 << <BATCH, 512 >> > (d_y, d_tf0, 9);

		pointwise_mult << <BATCH, NTTSIZE >> > (d_x, d_y, d_Z);
		
		radix2INTT_gpu0 << <BATCH, 512 >> > (d_Z, d_ti0, 2, 1);
		radix2INTT_gpu0 << <BATCH, 256 >> > (d_Z, d_ti0, 4, 2);
		radix2INTT_gpu0 << <BATCH, 128 >> > (d_Z, d_ti0, 8, 3);
		radix2INTT_gpu0 << <BATCH, 64 >> > (d_Z, d_ti0, 16, 4);
		radix2INTT_gpu0 << <BATCH, 32 >> > (d_Z, d_ti0, 32, 5);
		radix2INTT_gpu1 << <BATCH, 32 >> > (d_Z, d_ti0, 32, 6);
		radix2INTT_gpu1 << <BATCH, 64 >> > (d_Z, d_ti0, 64, 7);
		radix2INTT_gpu1 << <BATCH, 128 >> > (d_Z, d_ti0, 128, 8);
		radix2INTT_gpu1 << <BATCH, 256 >> > (d_Z, d_ti0, 256, 9);
		radix2INTT_gpu2 << <BATCH, 512 >> > (d_Z, d_ti0, 512, 10);
		cudaMemcpy(z, d_Z, BATCH * NTTSIZE * sizeof(uint32_t), cudaMemcpyDeviceToHost);		

		QueryPerformanceCounter(&t1);
		elapsed += ((double)((t1.QuadPart - t0.QuadPart) * 1000) / freq.QuadPart);
	}

	printf("Performance GPU GS-CT GPU \n Time\t\t: % .4f ms. \nThroughput\t: %.2f Multiplications per second\n", elapsed / NUM_AVE, (BATCH * NUM_AVE / elapsed) * 1000);
#ifdef DEBUG
	printf("z: "); for (int i = 0; i < BATCH * NTTSIZE; i++)
	{
		if (i % NTTSIZE == 0) printf("\n\n");
		printf("%u ", z[i]);
	}
#endif
	cudaFree(d_x);	cudaFree(d_X);
	cudaFree(d_y);	cudaFree(d_Y);
	cudaFree(d_Z);
}


void test_NTT_CT_CT_nega_gpu(uint32_t* x, uint32_t* y, uint32_t* z, uint32_t* X, uint32_t* Y, uint32_t* Z, uint32_t* tf0, uint32_t* ti0, uint32_t nfg0, uint32_t nig0, uint32_t Ni)
{
	for (int i = 0; i < BATCH * NTTSIZE; i++) { x[i] = 1; y[i] = 1; }
	printf("\n========================\n");
	printf("test_NTT_negacyclic CT-CT GPU. Batch Size is %d", BATCH);
	printf("\n========================\n");

	LARGE_INTEGER t0, t1, freq;
	uint32_t* d_x, * d_X, * d_y, * d_Y, * d_tf0, *d_z, * d_Z, * d_ti0;
	double elapsed = 0.0;

	cudaMalloc((void**)& d_x, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMalloc((void**)& d_X, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMalloc((void**)& d_y, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMalloc((void**)& d_Y, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMalloc((void**)& d_Z, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMalloc((void**)& d_z, BATCH * NTTSIZE * sizeof(uint32_t));

	cudaMemset(d_x, 0, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMemset(d_y, 0, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMemset(d_X, 0, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMemset(d_Y, 0, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMemset(d_z, 0, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMemset(d_Z, 0, BATCH * NTTSIZE * sizeof(uint32_t));
	QueryPerformanceFrequency(&freq);

	for (int i = 0; i < NUM_AVE; i++)
	{
		QueryPerformanceCounter(&t0);
		cudaMemcpy(d_x, x, BATCH * NTTSIZE * sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_y, y, BATCH * NTTSIZE * sizeof(uint32_t), cudaMemcpyHostToDevice);
		
		bit_reverse_copy_tbl_Phi_gpu << <BATCH, NTTSIZE >> > (d_x, d_X);
		bit_reverse_copy_tbl_Phi_gpu << <BATCH, NTTSIZE >> > (d_y, d_Y);
		//wklee, INTT and NTT are using the same function, only differ by twiddle factors
		radix2NTT_gpu0 << <BATCH, 512 >> > (d_X, d_tf0, 2, 1);
		radix2NTT_gpu0 << <BATCH, 256 >> > (d_X, d_tf0, 4, 2);
		radix2NTT_gpu0 << <BATCH, 128 >> > (d_X, d_tf0, 8, 3);
		radix2NTT_gpu0 << <BATCH, 64 >> > (d_X, d_tf0, 16, 4);
		radix2NTT_gpu0 << <BATCH, 32 >> > (d_X, d_tf0, 32, 5);
		radix2NTT_gpu1 << <BATCH, 32 >> > (d_X, d_tf0, 32, 6);
		radix2NTT_gpu1 << <BATCH, 64 >> > (d_X, d_tf0, 64, 7);
		radix2NTT_gpu1 << <BATCH, 128 >> > (d_X, d_tf0, 128, 8);
		radix2NTT_gpu1 << <BATCH, 256 >> > (d_X, d_tf0, 256, 9);
		radix2NTT_gpu1 << <BATCH, 512 >> > (d_X, d_tf0, 512, 10);

		radix2NTT_gpu0 << <BATCH, 512 >> > (d_Y, d_tf0, 2, 1);
		radix2NTT_gpu0 << <BATCH, 256 >> > (d_Y, d_tf0, 4, 2);
		radix2NTT_gpu0 << <BATCH, 128 >> > (d_Y, d_tf0, 8, 3);
		radix2NTT_gpu0 << <BATCH, 64 >> > (d_Y, d_tf0, 16, 4);
		radix2NTT_gpu0 << <BATCH, 32 >> > (d_Y, d_tf0, 32, 5);
		radix2NTT_gpu1 << <BATCH, 32 >> > (d_Y, d_tf0, 32, 6);
		radix2NTT_gpu1 << <BATCH, 64 >> > (d_Y, d_tf0, 64, 7);
		radix2NTT_gpu1 << <BATCH, 128 >> > (d_Y, d_tf0, 128, 8);
		radix2NTT_gpu1 << <BATCH, 256 >> > (d_Y, d_tf0, 256, 9);
		radix2NTT_gpu1 << <BATCH, 512 >> > (d_Y, d_tf0, 512, 10);

		pointwise_mult << <BATCH, NTTSIZE >> > (d_X, d_Y, d_z);
		bit_reverse_copy_tbl_gpu << <BATCH, NTTSIZE >> > (d_z, d_Z);
		radix2INTT_gpu0 << <BATCH, 512 >> > (d_Z, d_ti0, 2, 1);
		radix2INTT_gpu0 << <BATCH, 256 >> > (d_Z, d_ti0, 4, 2);
		radix2INTT_gpu0 << <BATCH, 128 >> > (d_Z, d_ti0, 8, 3);
		radix2INTT_gpu0 << <BATCH, 64 >> > (d_Z, d_ti0, 16, 4);
		radix2INTT_gpu0 << <BATCH, 32 >> > (d_Z, d_ti0, 32, 5);
		radix2INTT_gpu1 << <BATCH, 32 >> > (d_Z, d_ti0, 32, 6);
		radix2INTT_gpu1 << <BATCH, 64 >> > (d_Z, d_ti0, 64, 7);
		radix2INTT_gpu1 << <BATCH, 128 >> > (d_Z, d_ti0, 128, 8);
		radix2INTT_gpu1 << <BATCH, 256 >> > (d_Z, d_ti0, 256, 9);
		radix2INTT_gpu2 << <BATCH, 512 >> > (d_Z, d_ti0, 512, 10);
		
		cudaMemcpy(z, d_Z, BATCH * NTTSIZE * sizeof(uint32_t), cudaMemcpyDeviceToHost);

		QueryPerformanceCounter(&t1);
		elapsed += ((double)((t1.QuadPart - t0.QuadPart) * 1000) / freq.QuadPart);
	}

	printf("Performance GPU CT-CT GPU \n Time\t\t: % .4f ms. \nThroughput\t: %.2f Multiplications per second\n", elapsed / NUM_AVE, (BATCH * NUM_AVE / elapsed) * 1000);
#ifdef DEBUG
	printf("z: "); for (int i = 0; i < BATCH * NTTSIZE; i++)
	{
		if (i % NTTSIZE == 0) printf("\n\n");
		printf("%u ", z[i]);
	}
#endif
	cudaFree(d_x);	cudaFree(d_X);
	cudaFree(d_y);	cudaFree(d_Y);
	cudaFree(d_z); cudaFree(d_Z);
}


void test_NTT_GS_GS_nega_gpu(uint32_t* x, uint32_t* y, uint32_t* z, uint32_t* X, uint32_t* Y, uint32_t* Z, uint32_t* tf0, uint32_t* ti0, uint32_t nfg0, uint32_t nig0, uint32_t Ni)
{
	for (int i = 0; i < BATCH * NTTSIZE; i++) { x[i] = 1; y[i] = 1; }
	printf("\n========================\n");
	printf("test_NTT_negacyclic GS-GS GPU. Batch Size is %d", BATCH);
	printf("\n========================\n");

	LARGE_INTEGER t0, t1, freq;
	uint32_t* d_x, * d_X, * d_y, * d_Y, * d_Z,* d_tf0, * d_ti0;
	double elapsed = 0.0;

	cudaMalloc((void**)& d_x, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMalloc((void**)& d_X, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMalloc((void**)& d_y, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMalloc((void**)& d_Y, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMalloc((void**)& d_Z, BATCH * NTTSIZE * sizeof(uint32_t));

	cudaMemset(d_x, 0, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMemset(d_y, 0, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMemset(d_X, 0, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMemset(d_Y, 0, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMemset(d_Z, 0, BATCH * NTTSIZE * sizeof(uint32_t));
	QueryPerformanceFrequency(&freq);

	for (int i = 0; i < NUM_AVE; i++)
	{
		QueryPerformanceCounter(&t0);
		cudaMemcpy(d_x, x, BATCH * NTTSIZE * sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_y, y, BATCH * NTTSIZE * sizeof(uint32_t), cudaMemcpyHostToDevice);

		GS_radix2NTT_gpu0 << <BATCH, 512 >> > (d_x, d_tf0, 0);
		GS_radix2NTT_gpu1 << <BATCH, 256 >> > (d_x, d_tf0, 1);
		GS_radix2NTT_gpu1 << <BATCH, 128 >> > (d_x, d_tf0, 2);
		GS_radix2NTT_gpu1 << <BATCH, 64 >> > (d_x, d_tf0, 3);
		GS_radix2NTT_gpu1 << <BATCH, 32 >> > (d_x, d_tf0, 4);
		GS_radix2NTT_gpu2 << <BATCH, 32 >> > (d_x, d_tf0, 5);
		GS_radix2NTT_gpu2 << <BATCH, 64 >> > (d_x, d_tf0, 6);
		GS_radix2NTT_gpu2 << <BATCH, 128 >> > (d_x, d_tf0, 7);
		GS_radix2NTT_gpu2 << <BATCH, 256 >> > (d_x, d_tf0, 8);
		GS_radix2NTT_gpu2 << <BATCH, 512 >> > (d_x, d_tf0, 9);

		GS_radix2NTT_gpu0 << <BATCH, 512 >> > (d_y, d_tf0, 0);
		GS_radix2NTT_gpu1 << <BATCH, 256 >> > (d_y, d_tf0, 1);
		GS_radix2NTT_gpu1 << <BATCH, 128 >> > (d_y, d_tf0, 2);
		GS_radix2NTT_gpu1 << <BATCH, 64 >> > (d_y, d_tf0, 3);
		GS_radix2NTT_gpu1 << <BATCH, 32 >> > (d_y, d_tf0, 4);
		GS_radix2NTT_gpu2 << <BATCH, 32 >> > (d_y, d_tf0, 5);
		GS_radix2NTT_gpu2 << <BATCH, 64 >> > (d_y, d_tf0, 6);
		GS_radix2NTT_gpu2 << <BATCH, 128 >> > (d_y, d_tf0, 7);
		GS_radix2NTT_gpu2 << <BATCH, 256 >> > (d_y, d_tf0, 8);
		GS_radix2NTT_gpu2 << <BATCH, 512 >> > (d_y, d_tf0, 9);
			   
		bit_reverse_copy_tbl_gpu << <BATCH, NTTSIZE >> > (d_x, d_X);
		bit_reverse_copy_tbl_gpu << <BATCH, NTTSIZE >> > (d_y, d_Y);

		pointwise_mult << <BATCH, NTTSIZE >> > (d_X, d_Y, d_Z);

		GS_radix2INTT_gpu0 << <BATCH, 512 >> > (d_Z, d_ti0, 0);
		GS_radix2INTT_gpu0 << <BATCH, 256 >> > (d_Z, d_ti0, 1);
		GS_radix2INTT_gpu0 << <BATCH, 128 >> > (d_Z, d_ti0, 2);
		GS_radix2INTT_gpu0 << <BATCH, 64 >> > (d_Z, d_ti0, 3);
		GS_radix2INTT_gpu0 << <BATCH, 32 >> > (d_Z, d_ti0, 4);
		GS_radix2INTT_gpu2 << <BATCH, 32 >> > (d_Z, d_ti0, 5);
		GS_radix2INTT_gpu2 << <BATCH, 64 >> > (d_Z, d_ti0, 6);
		GS_radix2INTT_gpu2 << <BATCH, 128 >> > (d_Z, d_ti0, 7);
		GS_radix2INTT_gpu2 << <BATCH, 256 >> > (d_Z, d_ti0, 8);
		GS_radix2INTT_gpu2 << <BATCH, 512 >> > (d_Z, d_ti0, 9);
		bit_reverse_copy_tbl_invPhi_gpu << <BATCH, NTTSIZE >> > (d_Z, d_x);
		cudaMemcpy(z, d_x, BATCH * NTTSIZE * sizeof(uint32_t), cudaMemcpyDeviceToHost);

		QueryPerformanceCounter(&t1);
		elapsed += ((double)((t1.QuadPart - t0.QuadPart) * 1000) / freq.QuadPart);
	}

	printf("Performance GPU GS-GS GPU \n Time\t\t: % .4f ms. \nThroughput\t: %.2f Multiplications per second\n", elapsed / NUM_AVE, (BATCH * NUM_AVE / elapsed) * 1000);
#ifdef DEBUG
	printf("z: "); for (int i = 0; i < BATCH * NTTSIZE; i++)
	{
		if (i % NTTSIZE == 0) printf("\n\n");
		printf("%u ", z[i]);
	}
#endif
	cudaFree(d_x);	cudaFree(d_X);
	cudaFree(d_y);	cudaFree(d_Y);
	cudaFree(d_Z);
}

void test_NTT_CT_GS_nega_gpu(uint32_t* x, uint32_t* y, uint32_t* z, uint32_t* X, uint32_t* Y, uint32_t* Z, uint32_t* tf0, uint32_t* ti0, uint32_t nfg0, uint32_t nig0, uint32_t Ni)
{
	for (int i = 0; i < BATCH * NTTSIZE; i++) { x[i] = 1; y[i] = 1; }
	printf("\n========================\n");
	printf("test_NTT_negacyclic CT-GS GPU. Batch Size is %d", BATCH);
	printf("\n========================\n");

	LARGE_INTEGER t0, t1, freq;
	uint32_t* d_x, * d_X, * d_y, * d_Y, * d_tf0, * d_Z, * d_ti0;
	double elapsed = 0.0;

	cudaMalloc((void**)& d_x, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMalloc((void**)& d_X, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMalloc((void**)& d_y, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMalloc((void**)& d_Y, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMalloc((void**)& d_Z, BATCH * NTTSIZE * sizeof(uint32_t));

	cudaMemset(d_x, 0, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMemset(d_y, 0, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMemset(d_X, 0, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMemset(d_Y, 0, BATCH * NTTSIZE * sizeof(uint32_t));
	cudaMemset(d_Z, 0, BATCH * NTTSIZE * sizeof(uint32_t));
	QueryPerformanceFrequency(&freq);

	for (int i = 0; i < NUM_AVE; i++)
	{
		QueryPerformanceCounter(&t0);
		cudaMemcpy(d_x, x, BATCH * NTTSIZE * sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_y, y, BATCH * NTTSIZE * sizeof(uint32_t), cudaMemcpyHostToDevice);

		bit_reverse_copy_tbl_Phi_gpu << <BATCH, NTTSIZE >> > (d_x, d_X);
		bit_reverse_copy_tbl_Phi_gpu << <BATCH, NTTSIZE >> > (d_y, d_Y);

		radix2NTT_gpu0 << <BATCH, 512 >> > (d_X, d_tf0, 2, 1);
		radix2NTT_gpu0 << <BATCH, 256 >> > (d_X, d_tf0, 4, 2);
		radix2NTT_gpu0 << <BATCH, 128 >> > (d_X, d_tf0, 8, 3);
		radix2NTT_gpu0 << <BATCH, 64 >> > (d_X, d_tf0, 16, 4);
		radix2NTT_gpu0 << <BATCH, 32 >> > (d_X, d_tf0, 32, 5);
		radix2NTT_gpu1 << <BATCH, 32 >> > (d_X, d_tf0, 32, 6);
		radix2NTT_gpu1 << <BATCH, 64 >> > (d_X, d_tf0, 64, 7);
		radix2NTT_gpu1 << <BATCH, 128 >> > (d_X, d_tf0, 128, 8);
		radix2NTT_gpu1 << <BATCH, 256 >> > (d_X, d_tf0, 256, 9);
		radix2NTT_gpu1 << <BATCH, 512 >> > (d_X, d_tf0, 512, 10);

		radix2NTT_gpu0 << <BATCH, 512 >> > (d_Y, d_tf0, 2, 1);
		radix2NTT_gpu0 << <BATCH, 256 >> > (d_Y, d_tf0, 4, 2);
		radix2NTT_gpu0 << <BATCH, 128 >> > (d_Y, d_tf0, 8, 3);
		radix2NTT_gpu0 << <BATCH, 64 >> > (d_Y, d_tf0, 16, 4);
		radix2NTT_gpu0 << <BATCH, 32 >> > (d_Y, d_tf0, 32, 5);
		radix2NTT_gpu1 << <BATCH, 32 >> > (d_Y, d_tf0, 32, 6);
		radix2NTT_gpu1 << <BATCH, 64 >> > (d_Y, d_tf0, 64, 7);
		radix2NTT_gpu1 << <BATCH, 128 >> > (d_Y, d_tf0, 128, 8);
		radix2NTT_gpu1 << <BATCH, 256 >> > (d_Y, d_tf0, 256, 9);
		radix2NTT_gpu1 << <BATCH, 512 >> > (d_Y, d_tf0, 512, 10);

		pointwise_mult << <BATCH, NTTSIZE >> > (d_X, d_Y, d_Z);

		GS_radix2INTT_gpu0 << <BATCH, 512 >> > (d_Z, d_ti0, 0);
		GS_radix2INTT_gpu0 << <BATCH, 256 >> > (d_Z, d_ti0, 1);
		GS_radix2INTT_gpu0 << <BATCH, 128 >> > (d_Z, d_ti0, 2);
		GS_radix2INTT_gpu0 << <BATCH, 64 >> > (d_Z, d_ti0, 3);
		GS_radix2INTT_gpu0 << <BATCH, 32 >> > (d_Z, d_ti0, 4);
		GS_radix2INTT_gpu2 << <BATCH, 32 >> > (d_Z, d_ti0, 5);
		GS_radix2INTT_gpu2 << <BATCH, 64 >> > (d_Z, d_ti0, 6);
		GS_radix2INTT_gpu2 << <BATCH, 128 >> > (d_Z, d_ti0, 7);
		GS_radix2INTT_gpu2 << <BATCH, 256 >> > (d_Z, d_ti0, 8);
		GS_radix2INTT_gpu2 << <BATCH, 512 >> > (d_Z, d_ti0, 9);
		bit_reverse_copy_tbl_invPhi_gpu << <BATCH, NTTSIZE >> > (d_Z, d_x);
		cudaMemcpy(z, d_x, BATCH * NTTSIZE * sizeof(uint32_t), cudaMemcpyDeviceToHost);

		QueryPerformanceCounter(&t1);
		elapsed += ((double)((t1.QuadPart - t0.QuadPart) * 1000) / freq.QuadPart);
	}

	printf("Performance GPU CT-GS GPU \n Time\t\t: % .4f ms. \nThroughput\t: %.2f Multiplications per second\n", elapsed / NUM_AVE, (BATCH * NUM_AVE / elapsed) * 1000);
#ifdef DEBUG
	printf("z: "); for (int i = 0; i < BATCH * NTTSIZE; i++)
	{
		if (i % NTTSIZE == 0) printf("\n\n");
		printf("%u ", z[i]);
	}
#endif
	cudaFree(d_x);	cudaFree(d_X);
	cudaFree(d_y);	cudaFree(d_Y);
	cudaFree(d_Z);
}

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <windows.h>
#define BATCH 2
#define NUM_AVE 1
#define DEBUG
// Parameters specified for qTESLA-III-speed
#define QTESLA
//#define SMALLPRIME
#ifdef QTESLA
#define P  8404993
#define PARAM_QINV 4034936831
#define NTTSIZE 1024 
#define N1 32	//Must be less than NTTSIZE, and power of 2
#define N2 NTTSIZE/N1
#define NSIZE_hlf NTTSIZE / 2
#define MIU  33489019;	// Barrett Reduction, 2^48 / P
#define MIU2 8372254;	// Barrett Reduction, 2^46 / P
#endif
#ifdef SMALLPRIME
#define P 65537
#define NTTSIZE 32 //32|16|8|4
#define N1 4 //Must less than NTTSIZE, and power of 2
#define N2 NTTSIZE/N1
#define NSIZE_hlf NTTSIZE / 2
#endif
#define RANDOM 0 //0: fixed operands; 1: random operands;


//====================================================
void init_operand(uint32_t *x);
void init_operands(uint32_t *x, uint32_t *y);
void clear_operand(uint32_t *x);
void reset(uint32_t *x, uint32_t *y, uint32_t *X, uint32_t *Y);

//====================================================
void NTT_naive(uint32_t *ip, uint32_t *op, uint32_t fg);
void INTT_naive(uint32_t *ip, uint32_t *op, uint32_t fg, uint32_t Ni);

//====================================================
void NTT_precom(uint32_t *ip, uint32_t *op, uint32_t *tf);
void INTT_precom(uint32_t *ip, uint32_t *op, uint32_t *ti, uint32_t Ni);

//====================================================
void NTT_CT2(uint32_t *ip, uint32_t *op, uint32_t *tf0, uint32_t *tf1, uint32_t *tf2);
void INTT_CT2(uint32_t *ip, uint32_t *op, uint32_t *ti0, uint32_t *ti1, uint32_t *ti2, uint32_t Ni);

//====================================================
void test_NTT_naive(uint32_t *x, uint32_t *y, uint32_t *z, uint32_t *X, uint32_t *Y, uint32_t *Z, uint32_t fg, uint32_t ig, uint32_t Ni);
void test_NTT_precom(uint32_t *x, uint32_t *y, uint32_t *z, uint32_t *X, uint32_t *Y, uint32_t *Z, uint32_t *tf, uint32_t *ti, uint32_t Ni);
void test_NTT_precompute_BATCH(uint32_t *x, uint32_t *y, uint32_t *z, uint32_t *X, uint32_t *Y, uint32_t *Z, uint32_t *tf, uint32_t *ti, uint32_t Ni);
void test_NTT_GS_CT_BATCH(uint32_t* x, uint32_t* y, uint32_t* z, uint32_t* X, uint32_t* Y, uint32_t* Z, uint32_t* tf0, uint32_t* ti0, uint32_t fg0, uint32_t ig0, uint32_t Ni);
void test_NTT_nega_GS(uint32_t* x, uint32_t* y, uint32_t* z, uint32_t* X, uint32_t* Y, uint32_t* Z, uint32_t* tf0, uint32_t* ti0, uint32_t nfg0, uint32_t nig0, uint32_t Ni);
void test_NTT_nega_CT(uint32_t* x, uint32_t* y, uint32_t* z, uint32_t* X, uint32_t* Y, uint32_t* Z, uint32_t* tf0, uint32_t* ti0, uint32_t nfg0, uint32_t nig0, uint32_t Ni);
void test_NTT_Stockham_nega(uint32_t* x, uint32_t* y, uint32_t* z, uint32_t* X, uint32_t* Y, uint32_t* Z, uint32_t* tf0, uint32_t* ti0, uint32_t fg0, uint32_t ig0, uint32_t Ni);
void test_NTT_CT2(uint32_t* x, uint32_t* y, uint32_t* z, uint32_t* X, uint32_t* Y, uint32_t* Z, uint32_t* tf0, uint32_t* tf1, uint32_t* tf2, uint32_t* ti0, uint32_t* ti1, uint32_t* ti2, uint32_t Ni);
void test_NTT_CT2_BATCH(uint32_t* x, uint32_t* y, uint32_t* z, uint32_t* X, uint32_t* Y, uint32_t* Z, uint32_t* tf0, uint32_t* tf1, uint32_t* tf2, uint32_t* ti0, uint32_t* ti1, uint32_t* ti2, uint32_t Ni);
void test_nussbaumer(uint32_t* x, uint32_t* y, uint32_t* z, uint32_t* X, uint32_t* Y, uint32_t* Z);

//====================================================
void test_NTT_precom_gpu(uint32_t *x, uint32_t *y, uint32_t *z, uint32_t *X, uint32_t *Y, uint32_t *Z, uint32_t *tf, uint32_t *ti, uint32_t Ni);
void test_NTT_CT2_gpu(uint32_t *x, uint32_t *y, uint32_t *z, uint32_t *X, uint32_t *Y, uint32_t *Z, uint32_t *tf0, uint32_t *tf1, uint32_t *tf2, uint32_t *ti0, uint32_t *ti1, uint32_t *ti2, uint32_t Ni);
void test_NTT_Stockham_nega_gpu(uint32_t* x, uint32_t* y, uint32_t* z, uint32_t* X, uint32_t* Y, uint32_t* Z, uint32_t* tf0, uint32_t* ti0, uint32_t fg0, uint32_t ig0, uint32_t Ni);
void test_NTT_GS_CT_nega_gpu(uint32_t* x, uint32_t* y, uint32_t* z, uint32_t* X, uint32_t* Y, uint32_t* Z, uint32_t* tf0, uint32_t* ti0, uint32_t nfg0, uint32_t nig0, uint32_t Ni);
void test_NTT_CT_CT_nega_gpu(uint32_t* x, uint32_t* y, uint32_t* z, uint32_t* X, uint32_t* Y, uint32_t* Z, uint32_t* tf0, uint32_t* ti0, uint32_t nfg0, uint32_t nig0, uint32_t Ni);
void test_NTT_GS_GS_nega_gpu(uint32_t* x, uint32_t* y, uint32_t* z, uint32_t* X, uint32_t* Y, uint32_t* Z, uint32_t* tf0, uint32_t* ti0, uint32_t nfg0, uint32_t nig0, uint32_t Ni);
void test_NTT_CT_GS_nega_gpu(uint32_t* x, uint32_t* y, uint32_t* z, uint32_t* X, uint32_t* Y, uint32_t* Z, uint32_t* tf0, uint32_t* ti0, uint32_t nfg0, uint32_t nig0, uint32_t Ni);
void test_reduction();
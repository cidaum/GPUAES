#include <stdio.h>
#include <stdint.h>
#include <sys/time.h>
#include "aes.h"

#define aes_mul(a, b) ((a)&&(b)?iLogTable[(logTable[(a)]+logTable[(b)])%0xff]:0)
#define caes_mul(a, b) ((a)&&(b)?CiLogTable[(ClogTable[(a)]+ClogTable[(b)])%0xff]:0)
#define GET(M,X,Y) ((M)[((Y) << 2) + (X)])

int const THREADS = 512;

__device__ void C2SubBytes(uint8_t *estado) {
		estado[threadIdx.x] = Csbox[estado[threadIdx.x]];
}

__device__ void C2InvSubBytes(uint8_t *estado) {
		estado[threadIdx.x] = CInvSbox[estado[threadIdx.x]];
}

__device__ void C2ShiftRows(uint8_t *estado) {
		unsigned int idx  = threadIdx.x;
		int row = idx % 4;
		uint8_t t;
	
		t = estado[((idx + 4*row) % 16) + ((idx >> 4 ) << 4)];
	
		__syncthreads();

		estado[idx] = t;
}

__device__ void C2InvShiftRows(uint8_t *estado) {
		unsigned int idx  = threadIdx.x;
		int row = idx % 4;
		uint8_t t;
	
		t = estado[((idx - 4*row) % 16) + ((idx >> 4 ) << 4)];
	
		__syncthreads();

		estado[idx] = t;
}

__device__ void C2MixColumns(uint8_t *estado) {
		unsigned int idx = threadIdx.x;
		int base = idx % 4;
		uint8_t t;

		if(base == 0) t = caes_mul(0x02, estado[idx]) ^ caes_mul(0x03, estado[idx+1]) ^ estado[idx+2] ^ estado[idx+3];
		if(base == 1) t = estado[idx-1] ^ caes_mul(0x02, estado[idx]) ^ caes_mul(0x03, estado[idx+1]) ^ estado[idx+2];
		if(base == 2) t = estado[idx-2] ^ estado[idx-1] ^ caes_mul(0x02, estado[idx]) ^ caes_mul(0x03, estado[idx+1]);
		if(base == 3) t = caes_mul(0x03, estado[idx-3]) ^ estado[idx-2] ^ estado[idx-1] ^ caes_mul(0x02, estado[idx]);
	
		__syncthreads();

		estado[idx] = t;
}

__device__ void C2InvMixColumns(uint8_t *estado) {
		unsigned int idx = threadIdx.x;
		int base = idx % 4;
		uint8_t t;

		if(base == 0) t = caes_mul(0x0e, estado[idx]) ^ caes_mul(0x0b, estado[idx+1]) ^ caes_mul(0x0d, estado[idx+2]) ^ caes_mul(0x09, estado[idx+3]);
		if(base == 1) t = caes_mul(0x09, estado[idx-1]) ^ caes_mul(0x0e, estado[idx]) ^ caes_mul(0x0b, estado[idx+1]) ^ caes_mul(0x0d, estado[idx+2]);
		if(base == 2) t = caes_mul(0x0d, estado[idx-2]) ^ caes_mul(0x09, estado[idx-1]) ^ caes_mul(0x0e, estado[idx]) ^ caes_mul(0x0b, estado[idx+1]);
		if(base == 3) t = caes_mul(0x0b, estado[idx-3]) ^ caes_mul(0x0d, estado[idx-2]) ^ caes_mul(0x09, estado[idx-1]) ^ caes_mul(0x0e, estado[idx]);
	
		__syncthreads();

		estado[idx] = t;
}

__device__ void C2AddRoundKey(uint8_t *estado, uint8_t *chave) {
		estado[threadIdx.x] ^= chave[threadIdx.x % 16];
}


//Substitui o estado pelas entradas da S_BOX
__global__ void CSubBytes(uint8_t *estado) {
	estado[(blockIdx.x*blockDim.x)+(blockIdx.y*gridDim.x*blockDim.x)+threadIdx.x] = Csbox[estado[(blockIdx.x*blockDim.x)+(blockIdx.y*gridDim.x*blockDim.x)+threadIdx.x]];
}

void SubBytes(uint8_t *estado, uint64_t offset) {
	for(uint64_t j=0; j<offset; j++) {
		for(register int i=0; i<16; i++){
			estado[i+(16*j)] = Sbox[estado[i+(16*j)]];
		}
	}
}

__global__ void CInvSubBytes(uint8_t *estado) {
		estado[(blockIdx.x*blockDim.x)+(blockIdx.y*gridDim.x*blockDim.x)+threadIdx.x] = CInvSbox[estado[(blockIdx.x*blockDim.x)+(blockIdx.y*gridDim.x*blockDim.x)+threadIdx.x]];
}

void InvSubBytes(uint8_t *estado, uint64_t offset) {
	for(uint64_t j=0; j<offset; j++) {
		for(register int i=0; i<16; i++){
			estado[i+(16*j)]= InvSbox[estado[i+(16*j)]];
		}
	}
}

__global__ void CShiftRows(uint8_t *estado) {
	uint64_t idx  = (blockIdx.x*blockDim.x)+(blockIdx.y*gridDim.x*blockDim.x)+threadIdx.x;
	int row = idx % 4;
	uint8_t t;

	t = estado[((idx + 4*row) % 16) + ((idx >> 4 ) << 4)];

	__syncthreads();

	estado[idx] = t;
}

void ShiftRows(uint8_t *estado, uint64_t offset) {
	for(uint64_t j=0; j<offset; j++) {
		uint8_t t[16];
		for(register int i=0; i<16; i++){
			uint64_t idx  = i+(16*j);
			int row = idx % 4;
			
			t[i] = estado[((idx + 4*row) % 16) + ((idx >> 4) << 4)];
		}
		for(register int i=0; i<16; i++) {
			estado[i+(16*j)] = t[i];
		}
	}
}

__global__ void CInvShiftRows(uint8_t *estado) {
	uint64_t idx  = (blockIdx.x*blockDim.x)+(blockIdx.y*gridDim.x*blockDim.x)+threadIdx.x;
	int row = idx % 4;
	uint8_t t;

	t = estado[((idx - 4*row) % 16) + ((idx >> 4 ) << 4)];

	__syncthreads();

	estado[idx] = t;
}


void InvShiftRows(uint8_t *estado, uint64_t offset) {
	for(uint64_t j=0; j<offset; j++) {
		uint8_t t[16];
		for(register int i=0; i<16; i++){
			uint64_t idx  = i+(16*j);
			int row = idx % 4;
			
			t[i] = estado[((idx - 4*row) % 16) + ((idx >> 4) << 4)];
		}
		for(register int i=0; i<16; i++) {
			estado[i+(16*j)] = t[i];
		}
	}
}

__global__ void CMixColumns(uint8_t *estado) {
	uint64_t idx = (blockIdx.x*blockDim.x)+(blockIdx.y*gridDim.x*blockDim.x)+threadIdx.x;
	uint8_t base = idx % 4;
	uint8_t t;

	if(base == 0) t = caes_mul(0x02, estado[idx]) ^ caes_mul(0x03, estado[idx+1]) ^ estado[idx+2] ^ estado[idx+3];
	if(base == 1) t = estado[idx-1] ^ caes_mul(0x02, estado[idx]) ^ caes_mul(0x03, estado[idx+1]) ^ estado[idx+2];
	if(base == 2) t = estado[idx-2] ^ estado[idx-1] ^ caes_mul(0x02, estado[idx]) ^ caes_mul(0x03, estado[idx+1]);
	if(base == 3) t = caes_mul(0x03, estado[idx-3]) ^ estado[idx-2] ^ estado[idx-1] ^ caes_mul(0x02, estado[idx]);

	__syncthreads();

	estado[idx] = t;
}

void MixColumns(uint8_t *estado, uint64_t offset) {
	for(uint64_t j=0; j<offset; j++) {
		uint8_t t[16];
		for(register int i=0; i<16; i++) {
			uint64_t idx = (i+(16*j));
			uint8_t base = idx % 4;
	
			if(base == 0) t[i] = aes_mul(0x02, estado[idx]) ^ aes_mul(0x03, estado[idx+1]) ^ estado[idx+2] ^ estado[idx+3];
			if(base == 1) t[i] = estado[idx-1] ^ aes_mul(0x02, estado[idx]) ^ aes_mul(0x03, estado[idx+1]) ^ estado[idx+2];
			if(base == 2) t[i] = estado[idx-2] ^ estado[idx-1] ^ aes_mul(0x02, estado[idx]) ^ aes_mul(0x03, estado[idx+1]);
			if(base == 3) t[i] = aes_mul(0x03, estado[idx-3]) ^ estado[idx-2] ^ estado[idx-1] ^ aes_mul(0x02, estado[idx]);
		}
		for(register int i=0; i<16; i++) {
			estado[i+(16*j)] = t[i];
		}
	}
}

__global__ void CInvMixColumns(uint8_t *estado) {
	uint64_t idx = (blockIdx.x*blockDim.x)+(blockIdx.y*gridDim.x*blockDim.x)+threadIdx.x;
	uint8_t base = idx % 4;
	uint8_t t;

	if(base == 0) t = caes_mul(0x0e, estado[idx]) ^ caes_mul(0x0b, estado[idx+1]) ^ caes_mul(0x0d, estado[idx+2]) ^ caes_mul(0x09, estado[idx+3]);
	if(base == 1) t = caes_mul(0x09, estado[idx-1]) ^ caes_mul(0x0e, estado[idx]) ^ caes_mul(0x0b, estado[idx+1]) ^ caes_mul(0x0d, estado[idx+2]);
	if(base == 2) t = caes_mul(0x0d, estado[idx-2]) ^ caes_mul(0x09, estado[idx-1]) ^ caes_mul(0x0e, estado[idx]) ^ caes_mul(0x0b, estado[idx+1]);
	if(base == 3) t = caes_mul(0x0b, estado[idx-3]) ^ caes_mul(0x0d, estado[idx-2]) ^ caes_mul(0x09, estado[idx-1]) ^ caes_mul(0x0e, estado[idx]);
	
	__syncthreads();

	estado[idx] = t;
}

void InvMixColumns(uint8_t *estado, uint64_t offset) {
	for(uint64_t j=0; j<offset; j++) {
		uint8_t t[16];
		for(register int i=0; i<16; i++) {
			uint64_t idx = (i+(16*j));
			uint8_t base = idx % 4;
		
			if(base == 0) t[i] = aes_mul(0x0e, estado[idx]) ^ aes_mul(0x0b, estado[idx+1]) ^ aes_mul(0x0d, estado[idx+2]) ^ aes_mul(0x09, estado[idx+3]);
			if(base == 1) t[i] = aes_mul(0x09, estado[idx-1]) ^ aes_mul(0x0e, estado[idx]) ^ aes_mul(0x0b, estado[idx+1]) ^ aes_mul(0x0d, estado[idx+2]);
			if(base == 2) t[i] = aes_mul(0x0d, estado[idx-2]) ^ aes_mul(0x09, estado[idx-1]) ^ aes_mul(0x0e, estado[idx]) ^ aes_mul(0x0b, estado[idx+1]);
			if(base == 3) t[i] = aes_mul(0x0b, estado[idx-3]) ^ aes_mul(0x0d, estado[idx-2]) ^ aes_mul(0x09, estado[idx-1]) ^ aes_mul(0x0e, estado[idx]);
		}
		for(register int i=0; i<16; i++) {
			estado[i+(16*j)] = t[i];
		}
	}
}

__global__ void CAddRoundKey(uint8_t *estado, uint8_t *chave) {
		estado[(blockIdx.x*blockDim.x)+(blockIdx.y*gridDim.x*blockDim.x)+threadIdx.x] ^= chave[((blockIdx.x*blockDim.x)+(blockIdx.y*gridDim.x*blockDim.x)+threadIdx.x) % 16];
}

void AddRoundKey(uint8_t *estado, uint8_t *chave, uint64_t offset) {
	for(uint64_t j=0; j<offset; j++) {	
		for(uint8_t i=0; i<16; i++) {
			estado[i+(16*j)] ^= chave[i];
		}
	}
}

__global__ void C2InvAes(uint8_t *cp, uint8_t *cW, uint8_t Nr) {
	__shared__ uint8_t estado[THREADS];
	register int i;
	estado[threadIdx.x] = cp[(blockIdx.x*blockDim.x)+(blockIdx.y*blockDim.x*gridDim.x)+threadIdx.x];
	__syncthreads();
	C2AddRoundKey(estado, cW+(Nr << 4));
	for(i=Nr; i>1; i--) {
		C2InvShiftRows(estado);
		C2InvSubBytes(estado);
		C2AddRoundKey(estado, cW+((i-1) << 4));
		C2InvMixColumns(estado);
	}
	C2InvShiftRows(estado);
	C2InvSubBytes(estado);
	C2AddRoundKey(estado, cW);
	__syncthreads();
	cp[(blockIdx.x*blockDim.x)+(blockIdx.y*blockDim.x*gridDim.x)+threadIdx.x] = estado[threadIdx.x];
}

__global__ void C2Aes(uint8_t *cp, uint8_t *cW, uint8_t Nr) {
	__shared__ uint8_t estado[THREADS];
	register int i;
	estado[threadIdx.x] = cp[(blockIdx.x*blockDim.x)+(blockIdx.y*blockDim.x*gridDim.x)+threadIdx.x];
	__syncthreads();
	C2AddRoundKey(estado, cW);
	for(i=1; i<Nr; i++) {
		C2SubBytes(estado);
		C2ShiftRows(estado);
		C2MixColumns(estado);
		C2AddRoundKey(estado, cW+(i << 4));
	}
	C2SubBytes(estado);
	C2ShiftRows(estado);
	C2AddRoundKey(estado, cW+(i << 4));
	__syncthreads();
	cp[(blockIdx.x*blockDim.x)+(blockIdx.y*blockDim.x*gridDim.x)+threadIdx.x] = estado[threadIdx.x];
}

void cinvAes(uint8_t *cp, uint8_t *cW, uint8_t Nr, dim3 numeroBlocos, uint16_t numeroThreads, uint64_t n) {
	
	register uint8_t i;
//	register uint64_t j;
//  	uint8_t tmp[16*n];
//	cudaMemcpy(tmp, cp, sizeof(uint8_t)*16*n, cudaMemcpyDeviceToHost);
//  	printf("0 str ");
//  	for(j=0; j < 16*n; j++) {
//  		printf("%02X", tmp[j]);
//  	}
//  	printf("\n");
	CAddRoundKey<<<numeroBlocos,numeroThreads>>>(cp, cW+(Nr << 4));
//  	cudaMemcpy(tmp, cp, sizeof(uint8_t)*16*n, cudaMemcpyDeviceToHost);
//  	printf("0 add ");
//  	for(j=0; j < 16*n; j++) {
//  		printf("%02X", tmp[j]);
//  	}
//  	printf("\n");
	for(i=Nr; i>1; i--) {
		CInvShiftRows<<<numeroBlocos,numeroThreads>>>(cp);
//  		cudaMemcpy(tmp, cp, sizeof(uint8_t)*16*n, cudaMemcpyDeviceToHost);
//  		printf("%d shi ",i);
//  		for(j=0; j < 16*n; j++) {
//  			printf("%02X", tmp[j]);
//  		}
//  		printf("\n");
		CInvSubBytes<<<numeroBlocos,numeroThreads>>>(cp);
//  		cudaMemcpy(tmp, cp, sizeof(uint8_t)*16*n, cudaMemcpyDeviceToHost);
//  		printf("%d sub ",i);
// 		for(j=0; j < 16*n; j++) {
//  			printf("%02X", tmp[j]);
//  		}
//  		printf("\n");
		CAddRoundKey<<<numeroBlocos,numeroThreads>>>(cp, cW+((i-1) << 4));
//  		cudaMemcpy(tmp, cp, sizeof(uint8_t)*16*n, cudaMemcpyDeviceToHost);
// 		printf("%d add ",i);
//  		for(j=0; j < 16*n; j++) {
//  			printf("%02X", tmp[j]);
//  		}
//		printf("\n");
		CInvMixColumns<<<numeroBlocos,numeroThreads>>>(cp);
//  		cudaMemcpy(tmp, cp, sizeof(uint8_t)*16*n, cudaMemcpyDeviceToHost);
//  		printf("%d mix ",i);
//  		for(j=0; j < 16*n; j++) {
//  			printf("%02X", tmp[j]);
//  		}
//  		printf("\n");
	}
	CInvShiftRows<<<numeroBlocos,numeroThreads>>>(cp);
	CInvSubBytes<<<numeroBlocos,numeroThreads>>>(cp);
	CAddRoundKey<<<numeroBlocos,numeroThreads>>>(cp, cW);
	
}

void caes(uint8_t *cp, uint8_t *cW, uint8_t Nr, dim3 numeroBlocos, uint16_t numeroThreads, uint64_t n) {

	register uint8_t i;
//	register uint64_t j;
//	uint8_t tmp[16*n];
	CAddRoundKey<<<numeroBlocos,numeroThreads>>>(cp, cW);
//	cudaMemcpy(tmp, cp, sizeof(uint8_t)*16*n, cudaMemcpyDeviceToHost);
//	printf("0 add ");
//	for(j=0; j < 16*n; j++) {
//		printf("%02X", tmp[j]);
//	}
//	printf("\n");
	for(i=1; i<Nr; i++) {
		CSubBytes<<<numeroBlocos,numeroThreads>>>(cp);
//		cudaMemcpy(tmp, cp, sizeof(uint8_t)*16*n, cudaMemcpyDeviceToHost);
//		printf("%d sub ",i);
//		for(j=0; j < 16*n; j++) {
//			printf("%02X", tmp[j]);
//		}
//		printf("\n");
		CShiftRows<<<numeroBlocos,numeroThreads>>>(cp);
//		cudaMemcpy(tmp, cp, sizeof(uint8_t)*16*n, cudaMemcpyDeviceToHost);
//		printf("%d shi ",i);
//		for(j=0; j < 16*n; j++) {
//			printf("%02X", tmp[j]);
//		}
//		printf("\n");
		CMixColumns<<<numeroBlocos,numeroThreads>>>(cp);
//		cudaMemcpy(tmp, cp, sizeof(uint8_t)*16*n, cudaMemcpyDeviceToHost);
//		printf("%d mix ",i);
//		for(j=0; j < 16*n; j++) {
//			printf("%02X", tmp[j]);
//		}
//		printf("\n");
		CAddRoundKey<<<numeroBlocos,numeroThreads>>>(cp, cW+(i << 4));
//		cudaMemcpy(tmp, cp, sizeof(uint8_t)*16*n, cudaMemcpyDeviceToHost);
//		printf("%d add ",i);
//		for(j=0; j < 16*n; j++) {
//			printf("%02X", tmp[j]);
//		}
//		printf("\n");
	}
	CSubBytes<<<numeroBlocos,numeroThreads>>>(cp);
	CShiftRows<<<numeroBlocos,numeroThreads>>>(cp);
	CAddRoundKey<<<numeroBlocos,numeroThreads>>>(cp, cW+(i << 4));
}

void aes(uint8_t *tp, uint8_t *W, uint8_t Nr, uint64_t n) {

	register uint8_t i;
//	uint64_t k;
//	printf("-1 add ");
//	for(k=0; k < 16*n; k++) {
//		printf("%02X", tp[k]);
//	}
//	printf("\n");
	AddRoundKey(tp, W, n);
//	printf("0 add ");
//	for(k=0; k < 16*n; k++) {
//		printf("%02X", tp[k]);
//	}
//	printf("\n");
	for(i=1; i<Nr; i++){
		SubBytes(tp, n);
//		printf("%d sub ",i);
//		for(k=0; k < 16*n; k++) {
//			printf("%02X", tp[k]);
//		}
//		printf("\n");
		ShiftRows(tp, n);
//		printf("%d shi ",i);
//		for(k=0; k < 16*n; k++) {
//			printf("%02X", tp[k]);
//		}
//		printf("\n");
		MixColumns(tp, n);
//		printf("%d mix ",i);
//		for(k=0; k < 16*n; k++) {
//			printf("%02X", tp[k]);
//		}
//		printf("\n");
		AddRoundKey(tp, W+(i << 4), n);
//		printf("%d add ",i);
//		for(k=0; k < 16*n; k++) {
//			printf("%02X", tp[k]);
//		}
//		printf("\n");
	}
	SubBytes(tp, n);
	ShiftRows(tp, n);
	AddRoundKey(tp, W+(i << 4), n);
//	printf("%d add ",i);
//	for(k=0; k < 16*n; k++) {
//		printf("%02X", tp[k]);
//	}
//	printf("\n");
}

void invAes(uint8_t *tp, uint8_t *W, uint8_t Nr, uint64_t n) {

	register uint8_t i;
//	uint64_t k;
//	printf("-1 add ");
//	for(k=0; k < 16*n; k++) {
//  		printf("%02X", tp[k]);
//	}
//  	printf("\n");
      	AddRoundKey(tp, W+(Nr << 4), n);
//  	printf("0 add ");
//	for(k=0; k < 16*n; k++) {
//  		printf("%02X", tp[k]);
//  	}
//  	printf("\n");
      	for(i=Nr; i>1; i--){
      		InvShiftRows(tp, n);
//  		printf("%d shi ",i);
//  		for(k=0; k < 16*n; k++) {
//  			printf("%02X", tp[k]);
//  		}
//  		printf("\n");
    		InvSubBytes(tp, n);
//  		printf("%d sub ",i);
//  		for(k=0; k < 16*n; k++) {
// 			printf("%02X", tp[k]);
//		}
//		printf("\n");
      		AddRoundKey(tp, W+((i-1) << 4), n);
//		printf("%d add ",i);
//		for(k=0; k < 16*n; k++) {
//			printf("%02X", tp[k]);
//  		}
//  		printf("\n");
      		InvMixColumns(tp, n);
//  		printf("%d mix ",i);
//  		for(k=0; k < 16*n; k++) {
// 			printf("%02X", tp[k]);
//  		}
//  		printf("\n");
	}
	InvShiftRows(tp, n);
	InvSubBytes(tp, n);
	AddRoundKey(tp, W, n);
//	printf("%d add ",i);
//	for(k=0; k < 16*n; k++) {
//		printf("%02X", tp[k]);
//	}
//	printf("\n");
	
}

void ExpandKeys(uint8_t *key, uint8_t keysize, uint8_t *W, uint8_t Nk, uint8_t Nr) {
	uint8_t i, j, cols, temp, tmp[4];
	cols = (Nr + 1) << 2;

	memcpy(W, key, (keysize >> 3)*sizeof(uint8_t));

	for(i=Nk; i<cols; i++) {
		for(j=0; j<4; j++)
			tmp[j] = GET(W, j, i-1);
		if(Nk > 6) {
			if(i % Nk == 0) {
				temp   = Sbox[tmp[0]] ^  (Rcon[i/Nk] & 0x000000ff);
				tmp[0] = Sbox[tmp[1]] ^ ((Rcon[i/Nk] & 0xff000000) >> 24);
				tmp[1] = Sbox[tmp[2]] ^ ((Rcon[i/Nk] & 0x00ff0000) >> 16);
				tmp[2] = Sbox[tmp[3]] ^ ((Rcon[i/Nk] & 0x0000ff00) >>  8);
				tmp[3] = temp;
			} else if(i % Nk == 4) {
				tmp[0] = Sbox[tmp[0]];
				tmp[1] = Sbox[tmp[1]];
				tmp[2] = Sbox[tmp[2]];
				tmp[3] = Sbox[tmp[3]];
			}
		} else {
			if(i % Nk == 0) {
				temp   = Sbox[tmp[0]] ^  (Rcon[i/Nk] & 0x000000ff);
				tmp[0] = Sbox[tmp[1]] ^ ((Rcon[i/Nk] & 0xff000000) >> 24);
				tmp[1] = Sbox[tmp[2]] ^ ((Rcon[i/Nk] & 0x00ff0000) >> 16);
				tmp[2] = Sbox[tmp[3]] ^ ((Rcon[i/Nk] & 0x0000ff00) >>  8);
				tmp[3] = temp;
			}
		}
		for(j=0; j<4; j++)
			GET(W, j, i) = GET(W, j, i-Nk) ^ tmp[j];
	}
}

void aes_serial(uint8_t *in, uint8_t *chave, uint8_t *out, uint8_t tamanhoChave, uint64_t offset, uint8_t acao) {
	uint8_t *W, Nk, Nr;
	Nk = tamanhoChave >> 5;
	Nr = Nk + 6;
	uint64_t size = 4*4*offset*sizeof(uint8_t);
	uint64_t s = ((Nr+1) * sizeof(uint8_t)) << 4;
	W = (uint8_t *)malloc(s);
	ExpandKeys(chave, tamanhoChave, W, Nk, Nr);
	memcpy(out, in, size);
	if(acao) {
		aes(out, W, Nr, offset);
	} else {
		invAes(out, W, Nr, offset);
	}
	//printHexArray(out,sizeof(out));
	//for(register uint8_t i=0; i<(size/sizeof(uint8_t)); i++) {
	//	printf("%d:", out[i]);
	//}
	//printf("\n");
}

void aes_cuda(uint8_t *in, uint8_t *chave, uint8_t *out, uint8_t tamanhoChave, uint64_t offset, dim3 numeroBlocos, uint16_t numeroThreads, uint8_t acao) {
	uint8_t *cp, *W, *cW, Nk, Nr;
	Nk = tamanhoChave >> 5;
	Nr = Nk + 6;
	long size = 4*4*offset*sizeof(uint8_t);
	uint64_t s = ((Nr+1) * sizeof(uint8_t)) << 4;
	W = (uint8_t *)malloc(s);
	cudaMalloc((void**)&cW, s);
	ExpandKeys(chave, tamanhoChave, W, Nk, Nr);
	cudaMemcpyAsync(cW, W, s, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&cp, size);
	cudaMemcpyAsync(cp, in, size, cudaMemcpyHostToDevice);
	if(acao) {
		caes(cp, cW, Nr, numeroBlocos, numeroThreads, offset);
	} else {
		cinvAes(cp, cW, Nr, numeroBlocos, numeroThreads, offset);
	}
	cudaMemcpy(out, cp, size, cudaMemcpyDeviceToHost);
	cudaFree(&cW);
	cudaFree(&cp);
	//printHexArray(out,(size/sizeof(uint8_t)));
	//for(register uint8_t i=0; i<(size/sizeof(uint8_t)); i++) {
	//	printf("%d:", out[i]);
	//}
	//printf("\n");
}

void aes_cuda2(uint8_t *in, uint8_t *chave, uint8_t *out, uint8_t tamanhoChave, uint64_t offset, dim3 numeroBlocos, uint8_t acao) {
	uint8_t *cp, *W, *cW, Nk, Nr;
	Nk = tamanhoChave >> 5;
	Nr = Nk + 6;
	long size = 4*4*offset*sizeof(uint8_t);
	uint64_t s = ((Nr+1) * sizeof(uint8_t)) << 4;
	W = (uint8_t *)malloc(s);
	cudaMalloc((void**)&cW, s);
	ExpandKeys(chave, tamanhoChave, W, Nk, Nr);
	cudaMemcpyAsync(cW, W, s, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&cp, size);
	cudaMemcpyAsync(cp, in, size, cudaMemcpyHostToDevice);
	if(acao) {
		C2Aes<<<numeroBlocos, THREADS>>>(cp, cW, Nr);
	} else {
		C2InvAes<<<numeroBlocos, THREADS>>>(cp, cW, Nr);
	}
	cudaMemcpy(out, cp, size, cudaMemcpyDeviceToHost);
	cudaFree(&cW);
	cudaFree(&cp);
}

//Transforma a entrada em um array de char
uint8_t stringToByteArray(char *str, uint8_t *array[]) {
	register uint8_t i;
	uint8_t len  = strlen(str) >> 1;
	*array = (uint8_t *)malloc(len * sizeof(uint8_t));

	for(i=0; i<len; i++)
		sscanf(str + i*2, "%02X", *array+i);

	return len;
}

//Imprime a saída em hexa TODO gravar em um arquivo.
void printHexArray(uint8_t *array, uint64_t size) {
	register uint8_t i;
	for(i=0; i<size; i++)
		printf("%02X", array[i]);
	printf("\n");
}

//Popula uma entrada aleaória
void aleatorio(uint8_t *entrada, uint64_t size) {
	for(uint64_t i = 0; i < size; i++)
		entrada[i] = (uint8_t)(rand() % 0xff);
}

//calcula diferença de tempo
double time_diff(struct timeval * prior, struct timeval * latter) {
  double x =
   (double)(latter->tv_usec - prior->tv_usec) / 1000.0L +
   (double)(latter->tv_sec - prior->tv_sec) * 1000.0L;
  return x;
}

int main(int argc, char **argv){
	struct timeval inicio, fim, inicioc, fimc, inicioc2, fimc2, inicios, fims;
	gettimeofday(&inicio,NULL);
	double tempo, totalc, totalc2, totals;
	uint8_t *chave, *outs, *outc, *outc2, *in;
	uint64_t blocos;

        if(argc < 4) {
                printf("Número de parâmetros errados\nUse: aes BLOCOS THREADSPORBLOCO TAMANHOCHAVE TAMANHOENTRADA\n");
		return 1;
        }
	
	dim3 numeroBlocos(atoi(argv[1]), atoi(argv[2]));
	printf("\n x %d y %d z %d \n", numeroBlocos.x, numeroBlocos.y, numeroBlocos.z);
	int numeroThreads = atoi(argv[3]);
	uint8_t tamanhoChave = atoi(argv[4]);
	uint64_t tamanhoIn = atoi(argv[5]);
	
        if(tamanhoChave != 16 && tamanhoChave != 24 && tamanhoChave != 32) {
                printf("Tamanho da chave inválido\n");
                return 1;
        }
	if(tamanhoIn == 0) {
		char *chavein = "000102030405060708090a0b0c0d0e0f";
		char *inin = "3243f6a8885a308d313198a2e037073400112233445566778899aabbccddeeff";	
	        tamanhoChave = stringToByteArray(chavein, &chave);
	        tamanhoIn  = stringToByteArray(inin, &in);
	} else {
	       if(tamanhoIn % 16 != 0) {
			printf("Tamanho de bloco inválido\n Deve ser múltiplo de 16\n resto: %d \n", (tamanhoIn % 16));
			return 1;
	        } else {
			srand(time(NULL));
			chave = (uint8_t *)malloc(tamanhoChave * sizeof(uint8_t));
			in = (uint8_t *)malloc(tamanhoIn * sizeof(uint8_t));
			aleatorio(chave, tamanhoChave);
			aleatorio(in, tamanhoIn);
		}
	}		
	blocos = tamanhoIn / 16;
	printf("%d\n", tamanhoIn);
	printf("Entrada : ");
	printHexArray(in, 32);
	printf("Chave : ");
	printHexArray(chave, tamanhoChave);
	outs = (uint8_t *)malloc(tamanhoIn * sizeof(uint8_t));
	memset(outs, 0, tamanhoIn);
	outc = (uint8_t *)malloc(tamanhoIn * sizeof(uint8_t));
	memset(outc, 0, tamanhoIn);
	outc2 = (uint8_t *)malloc(tamanhoIn * sizeof(uint8_t));
	memset(outc2, 0, tamanhoIn);
	gettimeofday(&fim, NULL);
	tempo = time_diff(&inicio, &fim);
	printf("Tempo de inicialização em ms %f\n",  tempo); 
	
	printf("Criptografa CUDA\n");
	gettimeofday(&inicioc, NULL);
	aes_cuda(in, chave, outc, tamanhoChave << 3, blocos, numeroBlocos, numeroThreads, 1);
	gettimeofday(&fimc, NULL);
	totalc = tempo = time_diff(&inicioc, &fimc);
	printf("Tempo em ms %f\n",  tempo); 
//	printHexArray(outc, 32);

	printf("Criptografa CUDA Otimizado\n");
	gettimeofday(&inicioc2, NULL);
	aes_cuda2(in, chave, outc2, tamanhoChave << 3, blocos, numeroBlocos, 1);
	gettimeofday(&fimc2, NULL);
	totalc2 = tempo = time_diff(&inicioc2, &fimc2);
	printf("Tempo em ms %f\n",  tempo);
//	printHexArray(outc2, 32);

	printf("Criptografa Serial\n");
	gettimeofday(&inicios, NULL);
	aes_serial(in, chave, outs, tamanhoChave << 3, blocos, 1);
	gettimeofday(&fims, NULL);
	totals = tempo = time_diff(&inicios, &fims);
	printf("Tempo em ms %f\n",  tempo); 
//	printHexArray(outs, 32);
	
	printf("Verificando consistencia entre CUDA e Serial: ");
	!memcmp(outs, outc, tamanhoIn)?printf("OK\n"):printf("Falha. Verifique o algoritmo\n");
	printf("Verificando consistencia entre CUDA Otimizado e Serial: ");
	!memcmp(outs, outc2, tamanhoIn)?printf("OK\n"):printf("Falha. Verifique o algoritmo\n");

	printf("Descriptografa CUDA\n");
	gettimeofday(&inicioc, NULL);
	aes_cuda(outc, chave, outc, tamanhoChave << 3, blocos, numeroBlocos, numeroThreads, 0);
	gettimeofday(&fimc, NULL);
	tempo = time_diff(&inicioc, &fimc);
	totalc += tempo;
	printf("Tempo em ms %f\n",  tempo); 
//	printHexArray(outc, 32);
	printf("Verificando algoritmo CUDA: ");
	!memcmp(in, outc, tamanhoIn)?printf("OK\n"):printf("Falha. Verifique o algoritmo\n");

	printf("Descriptografa CUDA Otimizado\n");
	gettimeofday(&inicioc2, NULL);
	aes_cuda2(outc2, chave, outc2, tamanhoChave << 3, blocos, numeroBlocos, 0);
	gettimeofday(&fimc2, NULL);
	tempo = time_diff(&inicioc2, &fimc2);
	totalc2 += tempo;
	printf("Tempo em ms %f\n",  tempo); 
//	printHexArray(outc2, 32);
	printf("Verificando algoritmo CUDA Otimizado: ");
	!memcmp(in, outc2, tamanhoIn)?printf("OK\n"):printf("Falha. Verifique o algoritmo\n");
	
	printf("Descriptografa Serial\n");
	gettimeofday(&inicios, NULL);
	aes_serial(outs, chave, outs, tamanhoChave << 3, blocos, 0);
	gettimeofday(&fims, NULL);
	tempo = time_diff(&inicios, &fims);
	totals += tempo;
	printf("Tempo em ms %f\n",  tempo);
//	printHexArray(outs, 32);
	printf("Verificando algoritmo Serial: ");
	!memcmp(in, outs, tamanhoIn)?printf("OK\n"):printf("Falha. Verifique o algoritmo\n");
	printf("\n");

	printf("Tempo total cuda: %f Tempo total cuda o: %f Tempo total serial: %f", totalc, totalc2, totals); 

        return EXIT_SUCCESS;
}

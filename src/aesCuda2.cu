#include <stdio.h>
#include <time.h>
#include "aes2.h"

#define caes_mul(a, b) ((a)&&(b)?CiLogTable[(ClogTable[(a)]+ClogTable[(b)])%0xff]:0)
#define GET(M,X,Y) ((M)[((Y) << 2) + (X)])
int const THREADS = 512;

__device__ void SubBytes(uint8_t *estado) {
		estado[threadIdx.x] = Sbox[estado[threadIdx.x]];
}

__device__ void InvSubBytes(uint8_t *estado) {
		estado[threadIdx.x] = InvSbox[estado[threadIdx.x]];
}

__device__ void ShiftRows(uint8_t *estado) {
		unsigned int idx  = threadIdx.x;
		int row = idx % 4;
		uint8_t t;
	
		t = estado[((idx + 4*row) % 16) + ((idx >> 4 ) << 4)];
	
		__syncthreads();

		estado[idx] = t;
}

__device__ void InvShiftRows(uint8_t *estado) {
		unsigned int idx  = threadIdx.x;
		int row = idx % 4;
		uint8_t t;
	
		t = estado[((idx - 4*row) % 16) + ((idx >> 4 ) << 4)];
	
		__syncthreads();

		estado[idx] = t;
}

__device__ void MixColumns(uint8_t *estado) {
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

__device__ void InvMixColumns(uint8_t *estado) {
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

__device__ void AddRoundKey(uint8_t *estado, uint8_t *chave) {
		estado[threadIdx.x] ^= chave[threadIdx.x % 16];
}

__global__ void InvAes(uint8_t *cp, uint8_t *cW, uint8_t Nr) {
	__shared__ uint8_t estado[THREADS];
	register int i;
	estado[threadIdx.x] = cp[(blockIdx.x*blockDim.x)+(blockIdx.y*blockDim.x*gridDim.x)+threadIdx.x];
	__syncthreads();
	AddRoundKey(estado, cW+(Nr << 4));
	for(i=Nr; i>1; i--) {
		InvShiftRows(estado);
		InvSubBytes(estado);
		AddRoundKey(estado, cW+((i-1) << 4));
		InvMixColumns(estado);
	}
	InvShiftRows(estado);
	InvSubBytes(estado);
	AddRoundKey(estado, cW);
	__syncthreads();
	cp[(blockIdx.x*blockDim.x)+(blockIdx.y*blockDim.x*gridDim.x)+threadIdx.x] = estado[threadIdx.x];
}

__global__ void Aes(uint8_t *cp, uint8_t *cW, uint8_t Nr) {
	__shared__ uint8_t estado[THREADS];
	register int i;
	estado[threadIdx.x] = cp[(blockIdx.x*blockDim.x)+(blockIdx.y*blockDim.x*gridDim.x)+threadIdx.x];
	__syncthreads();
	AddRoundKey(estado, cW);
	for(i=1; i<Nr; i++) {
		SubBytes(estado);
		ShiftRows(estado);
		MixColumns(estado);
		AddRoundKey(estado, cW+(i << 4));
	}
	SubBytes(estado);
	ShiftRows(estado);
	AddRoundKey(estado, cW+(i << 4));
	__syncthreads();
	cp[(blockIdx.x*blockDim.x)+(blockIdx.y*blockDim.x*gridDim.x)+threadIdx.x] = estado[threadIdx.x];
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
				temp   = KeySbox[tmp[0]] ^  (Rcon[i/Nk] & 0x000000ff);
				tmp[0] = KeySbox[tmp[1]] ^ ((Rcon[i/Nk] & 0xff000000) >> 24);
				tmp[1] = KeySbox[tmp[2]] ^ ((Rcon[i/Nk] & 0x00ff0000) >> 16);
				tmp[2] = KeySbox[tmp[3]] ^ ((Rcon[i/Nk] & 0x0000ff00) >>  8);
				tmp[3] = temp;
			} else if(i % Nk == 4) {
				tmp[0] = KeySbox[tmp[0]];
				tmp[1] = KeySbox[tmp[1]];
				tmp[2] = KeySbox[tmp[2]];
				tmp[3] = KeySbox[tmp[3]];
			}
		} else {
			if(i % Nk == 0) {
				temp   = KeySbox[tmp[0]] ^  (Rcon[i/Nk] & 0x000000ff);
				tmp[0] = KeySbox[tmp[1]] ^ ((Rcon[i/Nk] & 0xff000000) >> 24);
				tmp[1] = KeySbox[tmp[2]] ^ ((Rcon[i/Nk] & 0x00ff0000) >> 16);
				tmp[2] = KeySbox[tmp[3]] ^ ((Rcon[i/Nk] & 0x0000ff00) >>  8);
				tmp[3] = temp;
			}
		}
		for(j=0; j<4; j++)
			GET(W, j, i) = GET(W, j, i-Nk) ^ tmp[j];
	}
}

void aes_cuda(uint8_t *in, uint8_t *chave, uint8_t *out, uint8_t tamanhoChave, uint64_t offset, dim3 numeroBlocos, uint8_t acao) {
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
		Aes<<<numeroBlocos, THREADS>>>(cp, cW, Nr);
	} else {
		InvAes<<<numeroBlocos, THREADS>>>(cp, cW, Nr);
	}
	cudaMemcpy(out, cp, size, cudaMemcpyDeviceToHost);
	cudaFree(&cW);
	cudaFree(&cp);
}

uint8_t stringToByteArray(char *str, uint8_t *array[]) {
	register uint8_t i;
	uint8_t len  = strlen(str) >> 1;
	*array = (uint8_t *)malloc(len * sizeof(uint8_t));

	for(i=0; i<len; i++)
		sscanf(str + i*2, "%02X", *array+i);

	return len;
}

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

int main(int argc, char **argv){
	clock_t passo;
	passo = clock();
	uint8_t *chave, *out, *in;
	uint64_t blocos;

        if(argc < 4) {
                printf("Número de parâmetros errados\nUse: aes BLOCOS THREADSPORBLOCO TAMANHOCHAVE TAMANHOENTRADA\n");
		return 1;
        }

	dim3 numeroBlocos = (atoi(argv[1]), atoi(argv[2]));
	uint8_t tamanhoChave = atoi(argv[3]);
	uint64_t tamanhoIn = atoi(argv[4]);
	
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
	out = (uint8_t *)malloc(tamanhoIn * sizeof(uint8_t));
	memset(out, 0, tamanhoIn);
	printf("Tempo de inicialização em ms %f\n",  (clock() - passo) / (double)CLOCKS_PER_SEC/1000); 
	printf("Criptografa CUDA\n");
	passo = clock();
	aes_cuda(in, chave, out, tamanhoChave << 3, blocos, numeroBlocos, 1);
	printf("Tempo em ms %f\n",  (clock() - passo) / (double)CLOCKS_PER_SEC); 
	printHexArray(out, 32);
	!memcmp(in, out, tamanhoIn)?printf("Falha\n"):printf("Ok\n");
	printf("Descriptografa CUDA\n");
	passo = clock();
	aes_cuda(out, chave, out, tamanhoChave << 3, blocos, numeroBlocos, 0);
	printf("Tempo em ms %f\n",  (clock() - passo) / (double)CLOCKS_PER_SEC); 
	printHexArray(out, 32);
	printf("Verificando algoritmo CUDA: ");
	!memcmp(in, out, tamanhoIn)?printf("OK\n"):printf("Falha. Verifique o algoritmo\n");
	printf("\n");

        return EXIT_SUCCESS;
}

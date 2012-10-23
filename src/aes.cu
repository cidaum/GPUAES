#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "cuda.h"
#include "aes.h"

#define xtime(x) ((x<<1) ^ (((x>>7) & 1) * 0x1b))
#define GET(M,X,Y) ((M)[((Y) << 2) + (X)])
const uint size = 4*4*sizeof(uint);

//Substitui o estado pelas entradas da S_BOX
__global__ void SubBytes(uint *estado) {
	estado[threadIdx.x] = sbox[estado[threadIdx.x]];
}


__global__ void ShiftRows(uint *estado) {
	uint8_t estadoTemp[4];
	register unsigned int i;
	for(i=0; i<4; i++)
		//Salva a linha com shift em uma tabela temporaria
		estadoTemp[i] = estadoTemp[threadIdx.x + 4*(i+threadIdx.x) % 16];
	for(i=0; i<4; i++)
		//Transfere para o estado as linhas com shift
		estado[threadIdx.x + 4*i] = estadoTemp[i];
}

__global__ void MixColumns(uint *estado) {
	uint base = threadIdx.x << 2;
	uint t, Tmp, Tm;

	t   = estado[base];
	Tmp = estado[base] ^ estado[base + 1] ^ estado[base + 2] ^ estado[base + 3];
	Tm  = estado[base    ] ^ estado[base + 1]; Tm = xtime(Tm) & 0xff; estado[base    ] ^= Tm ^ Tmp;
	Tm  = estado[base + 1] ^ estado[base + 2]; Tm = xtime(Tm) & 0xff; estado[base + 1] ^= Tm ^ Tmp;
	Tm  = estado[base + 2] ^ estado[base + 3]; Tm = xtime(Tm) & 0xff; estado[base + 2] ^= Tm ^ Tmp;
	Tm  = estado[base + 3] ^ t;               Tm = xtime(Tm) & 0xff; estado[base + 3] ^= Tm ^ Tmp;
}

__global__ void AddRoundKey(uint *estado, uint *chave) {
	estado[threadIdx.x] ^= chave[threadIdx.x];
}

void ExpandKeys(uint *key, uint keysize, uint *W, uint Nk, uint Nr) {
	uint i, j, cols, temp, tmp[4];
	cols = (Nr + 1) << 2;

	memcpy(W, key, (keysize >> 3)*sizeof(uint));

	for(i=Nk; i<cols; i++) {
		for(j=0; j<4; j++)
			tmp[j] = GET(W, j, i-1);
		if(Nk > 6) {
			if(i % Nk == 0) {
				temp   = hsbox[tmp[0]] ^  (Rcon[i/Nk] & 0x000000ff);
				tmp[0] = hsbox[tmp[1]] ^ ((Rcon[i/Nk] & 0xff000000) >> 24);
				tmp[1] = hsbox[tmp[2]] ^ ((Rcon[i/Nk] & 0x00ff0000) >> 16);
				tmp[2] = hsbox[tmp[3]] ^ ((Rcon[i/Nk] & 0x0000ff00) >>  8);
				tmp[3] = temp;
			} else if(i % Nk == 4) {
				tmp[0] = hsbox[tmp[0]];
				tmp[1] = hsbox[tmp[1]];
				tmp[2] = hsbox[tmp[2]];
				tmp[3] = hsbox[tmp[3]];
			}
		} else {
			if(i % Nk == 0) {
				temp   = hsbox[tmp[0]] ^  (Rcon[i/Nk] & 0x000000ff);
				tmp[0] = hsbox[tmp[1]] ^ ((Rcon[i/Nk] & 0xff000000) >> 24);
				tmp[1] = hsbox[tmp[2]] ^ ((Rcon[i/Nk] & 0x00ff0000) >> 16);
				tmp[2] = hsbox[tmp[3]] ^ ((Rcon[i/Nk] & 0x0000ff00) >>  8);
				tmp[3] = temp;
			}
		}
		for(j=0; j<4; j++)
			GET(W, j, i) = GET(W, j, i-Nk) ^ tmp[j];
	}
}

//Transforma a entrada em um array de char
uint stringToByteArray(char *str, uint *array[]) {
	register uint i;
	uint len  = strlen(str) >> 1;
	*array = (uint *)malloc(len * sizeof(uint));

	for(i=0; i<len; i++)
		sscanf(str + i*2, "%02X", *array+i);

	return len;
}

void aes(uint *cp, uint *cW, uint Nr) {

	register uint i;

	AddRoundKey<<<1,16>>>(cp, cW);
	for(i=1; i<Nr; i++) {
		SubBytes<<<1,16>>>(cp);
		ShiftRows<<<1,4>>>(cp);
		MixColumns<<<1,4>>>(cp);
		AddRoundKey<<<1,16>>>(cp, cW+(i << 4));
	}
	SubBytes<<<1,16>>>(cp);
	ShiftRows<<<1,4>>>(cp);
	AddRoundKey<<<1,16>>>(cp, cW+(i << 4));
}

void encriptar(uint *tp, uint *chave, uint *tc, uint tamanhoChave) {
	uint *cp, *W, *cW, Nk, Nr;
	Nk = tamanhoChave >> 5;
	Nr = Nk + 6;

	uint s = ((Nr+1) * sizeof(uint)) << 4;
	W = (uint *)malloc(s);
	cudaMalloc((void**)&cW, s);
	ExpandKeys(chave, tamanhoChave, W, Nk, Nr);
	cudaMemcpy(cW, W, s, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&cp, size);
	cudaMemcpy(cp, tp, size, cudaMemcpyHostToDevice);

	aes(cp, cW, Nr);

	cudaMemcpy(tc, cp, size, cudaMemcpyDeviceToHost);

}

void printHexArray(uint *array, uint size) {
	uint i;
	for(i=0; i<size; i++)
		printf("%02X", array[i]);
	printf("\n");
}

int main(int argc, char *argv[]){
	
	if(argc < 4) {
		printf("Número de parâmetros errados\nUse: aes -e CHAVE TEXTO para encriptar\n     aes -d CHAVE TEXTO para decriptar");
	}

	uint *chave, *out, *in;
	uint tamanhoChave = stringToByteArray(argv[2], &chave);
	uint tamanhoIn  = stringToByteArray(argv[3], &in);

	if(tamanhoChave != 16 && tamanhoChave != 24 && tamanhoChave != 32) {
		printf("Tamanho da chave inválido\n");
		return 1;
	}

	if(tamanhoIn % 16 != 0) {
		//TODO Fazer algo para completar o tamanho dos blocos
		return 1;
	}

	if (strcmp(argv[1], "-e")) {
		encriptar(in, chave, out, tamanhoChave << 3);
	} else
		if (strcmp(argv[1], "-d")) {
			decriptar(in, chave, out, tamanhoChave << 3);
		} else
			printf("Parâmetro inválido\n Use -e para encriptar e -d para decriptar");

	printHexArray(out, tamanhoIn);

	return EXIT_SUCCESS;
}

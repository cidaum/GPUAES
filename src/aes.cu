#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "aes.h"

//#define xtime(x) ((x<<1) ^ (((x>>7) & 1) * 0x1b))
#define aes_mul(a, b) ((a)&&(b)?iLogTable[(logTable[(a)]+logTable[(b)])%0xff]:0)
#define caes_mul(a, b) ((a)&&(b)?CiLogTable[(ClogTable[(a)]+ClogTable[(b)])%0xff]:0)
#define GET(M,X,Y) ((M)[((Y) << 2) + (X)])

//const uint8_t size = 4*4*sizeof(uint8_t);

//Substitui o estado pelas entradas da S_BOX
__global__ void CSubBytes(uint8_t *estado) {
	estado[threadIdx.x] = Csbox[estado[threadIdx.x]];
}

void SubBytes(uint8_t *estado, ulong offset) {
	for(register int i=0; i<16; i++){
		estado[i*offset] = Sbox[estado[i*offset]];
	}
}

__global__ void CInvSubBytes(uint8_t *estado) {
	estado[threadIdx.x] = CInvSbox[estado[threadIdx.x]];
}

void InvSubBytes(uint8_t *estado, ulong offset) {
	for(register int i=0; i<16; i++){
		estado[i*offset] = InvSbox[estado[i*offset]];
	}
}

__global__ void CShiftRows(uint8_t *estado) {
	unsigned int row  = threadIdx.x;
	uint8_t tmp[4];

	tmp[0] = estado[row + 4*(0+row) % 16];
	tmp[1] = estado[row + 4*(1+row) % 16];
	tmp[2] = estado[row + 4*(2+row) % 16];
	tmp[3] = estado[row + 4*(3+row) % 16];

	estado[row + 4*0] = tmp[0];
	estado[row + 4*1] = tmp[1];
	estado[row + 4*2] = tmp[2];
	estado[row + 4*3] = tmp[3];
}

void ShiftRows(uint8_t *estado, uint8_t offset) {
	for(register int i=0; i<16; i++){
		uint8_t row  = i*offset;
		uint8_t tmp[4];
	
		tmp[0] = estado[row + 4*(0+row) % 16];
		tmp[1] = estado[row + 4*(1+row) % 16];
		tmp[2] = estado[row + 4*(2+row) % 16];
		tmp[3] = estado[row + 4*(3+row) % 16];

		estado[row + 4*0] = tmp[0];
		estado[row + 4*1] = tmp[1];
		estado[row + 4*2] = tmp[2];
		estado[row + 4*3] = tmp[3];
	}
}

__global__ void CInvShiftRows(uint8_t *estado) {
	unsigned int row  = threadIdx.x;
	uint8_t tmp[4];

	tmp[0] = estado[row + 4*(0-row) % 16];
	tmp[1] = estado[row + 4*(1-row) % 16];
	tmp[2] = estado[row + 4*(2-row) % 16];
	tmp[3] = estado[row + 4*(3-row) % 16];

	estado[row + 4*0] = tmp[0];
	estado[row + 4*1] = tmp[1];
	estado[row + 4*2] = tmp[2];
	estado[row + 4*3] = tmp[3];
}


void InvShiftRows(uint8_t *estado, ulong offset) {
	for(register int i=0; i<16; i++){
		uint8_t row  = i*offset;
		uint8_t tmp[4];
	
		tmp[0] = estado[row + 4*(0-row) % 16];
		tmp[1] = estado[row + 4*(1-row) % 16];
		tmp[2] = estado[row + 4*(2-row) % 16];
		tmp[3] = estado[row + 4*(3-row) % 16];

		estado[row + 4*0] = tmp[0];
		estado[row + 4*1] = tmp[1];
		estado[row + 4*2] = tmp[2];
		estado[row + 4*3] = tmp[3];
	}
}
__global__ void CMixColumns(uint8_t *estado) {
	uint8_t base = threadIdx.x << 2;
	uint8_t t[4];
	
	t[0] = caes_mul(0x02, estado[base]) ^ caes_mul(0x03, estado[base+1]) ^ estado[base+2] ^ estado[base+3];
	t[1] = estado[base] ^ caes_mul(0x02, estado[base+1]) ^ caes_mul(0x03, estado[base+2]) ^ estado[base+3];
	t[2] = estado[base] ^ estado[base+1] ^ caes_mul(0x02, estado[base+2]) ^ caes_mul(0x03, estado[base+3]);
	t[3] = caes_mul(0x03, estado[base]) ^ estado[base+1] ^ estado[base+2] ^ caes_mul(0x02, estado[base+3]);

	estado[base] = t[0];
	estado[base+1] = t[1];
	estado[base+2] = t[2];
	estado[base+3] = t[3];	
}

void MixColumns(uint8_t *estado, ulong offset) {
	for(register int i=0; i<16; i++) {
		uint8_t base = (i*offset) << 2;
		uint8_t t[4];
	
		t[0] = aes_mul(0x02, estado[base]) ^ aes_mul(0x03, estado[base+1]) ^ estado[base+2] ^ estado[base+3];
		t[1] = estado[base] ^ aes_mul(0x02, estado[base+1]) ^ aes_mul(0x03, estado[base+2]) ^ estado[base+3];
		t[2] = estado[base] ^ estado[base+1] ^ aes_mul(0x02, estado[base+2]) ^ aes_mul(0x03, estado[base+3]);
		t[3] = aes_mul(0x03, estado[base]) ^ estado[base+1] ^ estado[base+2] ^ aes_mul(0x02, estado[base+3]);

		estado[base] = t[0];
		estado[base+1] = t[1];
		estado[base+2] = t[2];
		estado[base+3] = t[3];
	}	
}

__global__ void CInvMixColumns(uint8_t *estado) {
	uint8_t base = threadIdx.x << 2;
	uint8_t t[4];
	
	t[0] = caes_mul(0x0e, estado[base]) ^ caes_mul(0x0b, estado[base+1]) ^ caes_mul(0x0d, estado[base+2]) ^ caes_mul(0x09, estado[base+3]);
	t[1] = caes_mul(0x09, estado[base]) ^ caes_mul(0x0e, estado[base+1]) ^ caes_mul(0x0b, estado[base+2]) ^ caes_mul(0x0d, estado[base+3]);
	t[2] = caes_mul(0x0d, estado[base]) ^ caes_mul(0x09, estado[base+1]) ^ caes_mul(0x0e, estado[base+2]) ^ caes_mul(0x0b, estado[base+3]);
	t[3] = caes_mul(0x0b, estado[base]) ^ caes_mul(0x0d, estado[base+1]) ^ caes_mul(0x09, estado[base+2]) ^ caes_mul(0x0e, estado[base+3]);

	estado[base] = t[0];
	estado[base+1] = t[1];
	estado[base+2] = t[2];
	estado[base+3] = t[3];	
}

void InvMixColumns(uint8_t *estado, ulong offset) {
	for(uint8_t i=0; i<16; i++) {
		uint8_t base = (i*offset) << 2;
		uint8_t t[4];
	
		t[0] = aes_mul(0x0e, estado[base]) ^ aes_mul(0x0b, estado[base+1]) ^ aes_mul(0x0d, estado[base+2]) ^ aes_mul(0x09, estado[base+3]);
		t[1] = aes_mul(0x09, estado[base]) ^ aes_mul(0x0e, estado[base+1]) ^ aes_mul(0x0b, estado[base+2]) ^ aes_mul(0x0d, estado[base+3]);
		t[2] = aes_mul(0x0d, estado[base]) ^ aes_mul(0x09, estado[base+1]) ^ aes_mul(0x0e, estado[base+2]) ^ aes_mul(0x0b, estado[base+3]);
		t[3] = aes_mul(0x0b, estado[base]) ^ aes_mul(0x0d, estado[base+1]) ^ aes_mul(0x09, estado[base+2]) ^ aes_mul(0x0e, estado[base+3]);

		estado[base] = t[0];
		estado[base+1] = t[1];
		estado[base+2] = t[2];
		estado[base+3] = t[3];
	}
}

__global__ void CAddRoundKey(uint8_t *estado, uint8_t *chave) {
	estado[threadIdx.x] ^= chave[threadIdx.x % 16];
}

void AddRoundKey(uint8_t *estado, uint8_t *chave, ulong offset) {
	for(uint8_t i=0; i<16; i++) {
		estado[i*offset] ^= chave[i];
	}
}

void cinvAes(uint8_t *cp, uint8_t *cW, uint8_t Nr, ulong n) {
	
	register uint8_t i;
	register uint8_t j;
	uint8_t tmp[16*n];
	cudaMemcpy(tmp, cp, sizeof(uint8_t)*16*n, cudaMemcpyDeviceToHost);
	printf("0 str ");
	for(j=0; j < 16*n; j++) {
		printf("%02X", tmp[j]);
	}
	printf("\n");
	CAddRoundKey<<<1,16*n>>>(cp, cW+(Nr << 4));
	cudaMemcpy(tmp, cp, sizeof(uint8_t)*16*n, cudaMemcpyDeviceToHost);
	printf("0 add ");
	for(j=0; j < 16*n; j++) {
		printf("%02X", tmp[j]);
	}
	printf("\n");
	for(i=Nr; i>1; i--) {
		CInvShiftRows<<<1,16*n>>>(cp);
		cudaMemcpy(tmp, cp, sizeof(uint8_t)*16*n, cudaMemcpyDeviceToHost);
		printf("%d shi ",i);
		for(j=0; j < 16*n; j++) {
			printf("%02X", tmp[j]);
		}
		printf("\n");
		CInvSubBytes<<<1,16*n>>>(cp);
		cudaMemcpy(tmp, cp, sizeof(uint8_t)*16*n, cudaMemcpyDeviceToHost);
		printf("%d sub ",i);
		for(j=0; j < 16*n; j++) {
			printf("%02X", tmp[j]);
		}
		printf("\n");
		CAddRoundKey<<<1,16*n>>>(cp, cW+((i-1) << 4));
		cudaMemcpy(tmp, cp, sizeof(uint8_t)*16*n, cudaMemcpyDeviceToHost);
		printf("%d add ",i);
		for(j=0; j < 16*n; j++) {
			printf("%02X", tmp[j]);
		}
		printf("\n");
		CInvMixColumns<<<1,16*n>>>(cp);
		cudaMemcpy(tmp, cp, sizeof(uint8_t)*16*n, cudaMemcpyDeviceToHost);
		printf("%d mix ",i);
		for(j=0; j < 16*n; j++) {
			printf("%02X", tmp[j]);
		}
		printf("\n");
	}
	CInvShiftRows<<<1,16*n>>>(cp);
	CInvSubBytes<<<1,16*n>>>(cp);
	CAddRoundKey<<<1,16*n>>>(cp, cW);
	
}

void caes(uint8_t *cp, uint8_t *cW, uint8_t Nr, ulong n) {

	register uint8_t i;
//	register uint8_t j;
//	uint8_t tmp[16*n];
	CAddRoundKey<<<1,16*n>>>(cp, cW);
//	cudaMemcpy(tmp, cp, sizeof(uint8_t)*16*n, cudaMemcpyDeviceToHost);
//	printf("0 add ");
//	for(j=0; j < 16*n; j++) {
//		printf("%02X", tmp[j]);
//	}
//	printf("\n");
	for(i=1; i<Nr; i++) {
		CSubBytes<<<1,16*n>>>(cp);
//		cudaMemcpy(tmp, cp, sizeof(uint8_t)*16*n, cudaMemcpyDeviceToHost);
//		printf("%d sub ",i);
//		for(j=0; j < 16*n; j++) {
//			printf("%02X", tmp[j]);
//		}
//		printf("\n");
		CShiftRows<<<1,16*n>>>(cp);
//		cudaMemcpy(tmp, cp, sizeof(uint8_t)*16*n, cudaMemcpyDeviceToHost);
//		printf("%d shi ",i);
//		for(j=0; j < 16*n; j++) {
//			printf("%02X", tmp[j]);
//		}
//		printf("\n");
		CMixColumns<<<1,16*n>>>(cp);
//		cudaMemcpy(tmp, cp, sizeof(uint8_t)*16*n, cudaMemcpyDeviceToHost);
//		printf("%d mix ",i);
//		for(j=0; j < 16*n; j++) {
//			printf("%02X", tmp[j]);
//		}
//		printf("\n");
		CAddRoundKey<<<1,16*n>>>(cp, cW+(i << 4));
//		cudaMemcpy(tmp, cp, sizeof(uint8_t)*16*n, cudaMemcpyDeviceToHost);
//		printf("%d add ",i);
//		for(j=0; j < 16*n; j++) {
//			printf("%02X", tmp[j]);
//		}
//		printf("\n");
	}
	CSubBytes<<<1,16*n>>>(cp);
	CShiftRows<<<1,16*n>>>(cp);
	CAddRoundKey<<<1,16*n>>>(cp, cW+(i << 4));
}

void aes(uint8_t *tp, uint8_t *W, uint8_t Nr, ulong n) {
	register uint8_t i,j;
	for(j=1; j<=n; j++){
		AddRoundKey(tp, W, j);
		for(i=1; i<Nr; i++){
			SubBytes(tp, j);
			ShiftRows(tp, j);
			MixColumns(tp, j);
			AddRoundKey(tp, W+(i << 4), j);
		}
		SubBytes(tp, j);
		ShiftRows(tp, j);
		AddRoundKey(tp, W+(i << 4), j);
	}
}

void invAes(uint8_t *tp, uint8_t *W, uint8_t Nr, ulong n) {
	//TODO
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

void aes_serial(uint8_t *in, uint8_t *chave, uint8_t **out, uint8_t tamanhoChave, ulong offset, uint8_t acao) {
	uint8_t *W, Nk, Nr;
	Nk = tamanhoChave >> 5;
	Nr = Nk + 6;
	ulong size = 4*4*offset*sizeof(uint8_t);
	uint8_t s = ((Nr+1) * sizeof(uint8_t)) << 4;
	W = (uint8_t *)malloc(s);
	ExpandKeys(chave, tamanhoChave, W, Nk, Nr);
	if(acao) {
		printf("Criptgrafa serial\n");
		aes(in, W, Nr, offset);
	} else {
		printf("Descriptografa serial\n");
		invAes(in, W, Nr, offset);
	}
	*out = (uint8_t *)malloc(size*16*sizeof(uint8_t));
	memcpy(*out, in, sizeof(in));
	//printHexArray(out,(size/sizeof(uint8_t)));
	//for(register uint8_t i=0; i<(size/sizeof(uint8_t)); i++) {
	//	printf("%d:", out[i]);
	//}
	//printf("\n");
}

void aes_cuda(uint8_t *in, uint8_t *chave, uint8_t **out, uint8_t tamanhoChave, ulong offset, uint8_t acao) {
	uint8_t *cp, *W, *cW, Nk, Nr;
	Nk = tamanhoChave >> 5;
	Nr = Nk + 6;
	ulong size = 4*4*offset*sizeof(uint8_t);
	uint8_t s = ((Nr+1) * sizeof(uint8_t)) << 4;
	W = (uint8_t *)malloc(s);
	cudaMalloc((void**)&cW, s);
	ExpandKeys(chave, tamanhoChave, W, Nk, Nr);
	cudaMemcpy(cW, W, s, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&cp, size);
	cudaMemcpy(cp, in, size, cudaMemcpyHostToDevice);
	if(acao) {
		printf("Criptgrafa\n");
		caes(cp, cW, Nr, offset);
	} else {
		printf("Descriptografa\n");
		cinvAes(cp, cW, Nr, offset);
	}
	*out = (uint8_t *)malloc(size*16*sizeof(uint8_t));
	cudaMemcpy(*out, cp, size, cudaMemcpyDeviceToHost);
	//printHexArray(out,(size/sizeof(uint8_t)));
	//for(register uint8_t i=0; i<(size/sizeof(uint8_t)); i++) {
	//	printf("%d:", out[i]);
	//}
	//printf("\n");
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
void printHexArray(uint8_t *array, ulong size) {
	register uint8_t i;
	for(i=0; i<size; i++)
		printf("%02X", array[i]);
	printf("\n");
}

//Popula uma entrada aleaória
uint8_t *aleatorio(uint8_t size) {
	uint8_t *entrada = (uint8_t *)malloc(size * sizeof(uint8_t));
	for(long i = 0; i < size; i++)
		entrada[i] = (uint8_t)(rand() % 0xff);
	return entrada;

}

int main(int argc, char **argv){

/*        if(argc < 4) {
                printf("Número de parâmetros errados\nUse: aes enc CHAVE TEXTO para encriptar\n     aes dec CHAVE TEXTO para decriptar");
		return 1;
        }
*/      uint8_t *chave, *out = 0, *in, offset;
	time_t inicio, fim;
/*	srand(time(NULL));
	chave = aleatorio(16);
	printf("\nchave ");
	for(register int i=0; i< 16; i++){
		printf("%02X", chave[i]);
	}
	in = aleatorio(16);
	printf("\nconteudo ");
	for(register int i=0; i< 16; i++){
		printf("%02X", in[i]);
	}
	printf("\n");
*/	offset = 16 / 16;
	//uint8_t tamanhoChave = 16;
	//uint8_t tamanhoIn = 16;
	char *chavein = "000102030405060708090a0b0c0d0e0f";
	char *inin = "00112233445566778899aabbccddeeff";	
        uint8_t tamanhoChave = stringToByteArray(chavein, &chave);
        ulong tamanhoIn  = stringToByteArray(inin, &in);

        if(tamanhoChave != 16 && tamanhoChave != 24 && tamanhoChave != 32) {
                printf("Tamanho da chave inválido\n");
                return 1;
        }

        if(tamanhoIn % 16 != 0) {
                printf("Tamanho de bloco inválido\n Deve ser múltiplo de 16\n resto: %d \n", (tamanhoIn % 16));
                return 1;
        }
	inicio = time(NULL);
	printf("\ninicio %f\n", inicio);
	printHexArray(in, tamanhoIn);
//	aes_cuda(in, chave, &out, tamanhoChave << 3, offset, 1);
//      printHexArray(out, tamanhoIn);
//      aes_cuda(out, chave, &in, tamanhoChave << 3, offset, 0);
//      printHexArray(in, tamanhoIn);
	aes_serial(in, chave, &out, tamanhoChave << 3, offset, 1);
	printHexArray(out, tamanhoIn);
	fim = time(NULL);
	printf("\n Tempo gasto: %f\n", difftime(fim, inicio));

        return EXIT_SUCCESS;
}

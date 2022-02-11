// multiattack.c
// Starts multiple instances checking if a password can be found in the dictionnary.
// usage : nb_of_processus dictionnary_file shasum_file

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <string.h>

#define DEBUG false
#define WL_BLOCK 1000
#define MAX_LINE_LENGTH 200
#define MAX_SHADOW_LENGTH 5000
#define MAX_HASH_LENGTH 50

__global__ void check_hash(char **wordlist_block_plain, char **wordlist_block_hash, int lines, char **shadow_db, int shadow_count)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < shadow_count)
	{
		char *current_hash = shadow_db[index];

		#if DEBUG
			printf("[Thread - (%d,%d)] Cracking hash %s (%d)\n", blockIdx.x, threadIdx.x, current_hash, index);
		#endif
		for (int i = 0; i < lines; i++)
		{
			#if DEBUG
				printf("[Thread - (%d,%d)] Testing '%s' (%s)\n", blockIdx.x, threadIdx.x, wordlist_block_plain[i], wordlist_block_hash[i]);
			#endif
			bool ok = true;
			for (int v = 0; v < MAX_HASH_LENGTH; v++)
			{
				if (current_hash[v] == '\0' && wordlist_block_hash[i][v] == '\0')
					break;
				if (current_hash[v] != wordlist_block_hash[i][v] || current_hash[v] == '\0' || wordlist_block_hash[i][v] == '\0')
				{
					ok = false;
					break;
				}
			}
			if (ok) {
				printf("[+] FOUND %s for hash %s (shadow_index = %d)\n", wordlist_block_plain[i], current_hash, index);
				break;
			}
		}
	}
}

int main(int argc, char *argv[])
{
	// parsing arguments
	if (argc < 3)
		fprintf(stderr, "Usage: '%s' dictionnary_file shasum_file\n", argv[0]), exit(EXIT_FAILURE);
	char *dict_file = argv[1];
	char *shasum_file = argv[2];

	// opening files
	FILE *shadow_file = fopen(shasum_file, "r");
	FILE *wordlist_file = fopen(dict_file, "r");
	if (shadow_file == NULL || wordlist_file == NULL)
	{
		printf("Error while opening %s file\n", shadow_file == NULL ? "shadow" : "wordlist");
		exit(EXIT_FAILURE);
	}

	/* ------------- Loading shadow db into device ------------- */
	char *shadow_db[MAX_SHADOW_LENGTH];
	char **shadow_dbGPU;
	char buf[MAX_LINE_LENGTH];

	int shadow_count = 0;

	while ((fgets(buf, MAX_LINE_LENGTH, shadow_file)) != NULL)
	{
		buf[strlen(buf) - 1] = '\0'; // remove the trailing newline
#if DEBUG
		printf("address:%p -> %s\n", buf, buf);
#endif
		cudaMallocManaged(&shadow_db[shadow_count], strlen(buf));
		cudaMemcpy(shadow_db[shadow_count], buf, strlen(buf), cudaMemcpyHostToDevice);
		shadow_count++;
	}

	cudaMallocManaged(&shadow_dbGPU, shadow_count * sizeof(char *));
	cudaMemcpy(shadow_dbGPU, shadow_db, shadow_count * sizeof(char *), cudaMemcpyHostToDevice);

	// #if DEBUG
	// 	printf("[DEBUG] Shadow content - head : \n");
	// 	for (int i = 0; i < 10; i++)
	// 	{
	// 		printf("[%i] : %s\n", i, shadow_db[i]);
	// 	}
	// #endif

	/* ------- Optimizing number of threads & blocks based on 0.5 ratio ------ */
	int M = ceil((double)shadow_count / sqrt((shadow_count / 0.5)));
	int T = ceil((double)shadow_count / (double)M);

#if DEBUG
	printf("[DEBUG] Computed values : M=%d ; T=%d\n", M, T);
#endif

	/* ------------- Creating first parrallelisation by dividing wordlist into several blocks (divide & conquer strategy) ------------- */

	int block_counter = 0;
	while (true)
	{
		size_t lines = 0; /** next index to be used with lineBuffer
					(and number of lines already stored)*/
		char *lineBuffer_plain[WL_BLOCK];
		char *lineBuffer_hash[WL_BLOCK];
		char buf[MAX_LINE_LENGTH];
		while (lines < WL_BLOCK && fgets(buf, sizeof(buf), wordlist_file) != NULL)
		{
			char *plain, *hash;
			buf[strlen(buf) - 1] = '\0'; // remove trailing newline
			for (int v = 0; v < MAX_LINE_LENGTH; v++)
			{
				if (buf[v] == '\0')
				{
					printf("ERROR: invalid input line \"%s\" in wordlist\n", buf);
					exit(EXIT_FAILURE);
				}
				if (buf[v] == ' ')
				{
					buf[v] = '\0';
					plain = buf;
					hash = buf + v + 1;
					break;
				}
			}

			cudaMallocManaged(&lineBuffer_plain[lines], strlen(plain) * sizeof(char));
			cudaMemcpy(lineBuffer_plain[lines], plain, strlen(plain), cudaMemcpyHostToDevice);

			cudaMallocManaged(&lineBuffer_hash[lines], strlen(hash) * sizeof(char));
			cudaMemcpy(lineBuffer_hash[lines], hash, strlen(hash), cudaMemcpyHostToDevice);
			lines++;
		}
		if (lines == 0)
			break;

		block_counter++;
#if DEBUG
		printf("[+] Assigned block %d (read %zd lines)\n", block_counter, lines);
#endif
		char **lineBuffer_plainGPU, **lineBuffer_hashGPU;
		cudaMallocManaged(&lineBuffer_plainGPU, lines * sizeof(char *));
		cudaMemcpy(lineBuffer_plainGPU, lineBuffer_plain, lines * sizeof(char *), cudaMemcpyHostToDevice);

		cudaMallocManaged(&lineBuffer_hashGPU, lines * sizeof(char *));
		cudaMemcpy(lineBuffer_hashGPU, lineBuffer_hash, lines * sizeof(char *), cudaMemcpyHostToDevice);

		check_hash<<<M, T>>>(lineBuffer_plainGPU, lineBuffer_hashGPU, lines, shadow_dbGPU, shadow_count);

		// for (int i = 0; i < lines; i++)
		// {
		//     free(lineBuffer[i]);
		// }

		cudaDeviceSynchronize();
	}

	return 0;
}

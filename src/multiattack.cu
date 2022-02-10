// multiattack.c
// Starts multiple instances checking if a password can be found in the dictionnary.
// usage : nb_of_processus dictionnary_file shasum_file

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <string.h>

#define DEBUG 0
#define WL_BLOCK 1000
#define MAX_LINE_SIZE 50
#define MAX_SHADOW_LENGTH 5000
#define MAX_HASH_LENGTH 50

// Default maximum number of simultaneous process
int MAX_FILS = 5;

// wraup for readline
char *readline(FILE *f)
{
	char *line = NULL;

	size_t len = 0;
	ssize_t read;
	if ((read = getline(&line, &len, f)) != -1)
	{
		line[read - 2] = '\0';

		return line;
	}
	return NULL;
}

__global__ void check_hash(char **wordlist_block, int lines, char **shadow_db)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	// printf("%p index=%d\n", wordlist_block, index);
	char *current_hash = shadow_db[index];

	printf("[Block - %d] Cracking hash %s (%d)\n", current_hash, wordlist_block[index], index);
	// for (int i = 0; i < lines; i++)
	// {
	// 	bool ok = true;
	// 	for (int v = 0; v < MAX_HASH_LENGTH; v++)
	// 	{
	// 		if (current_hash[v] == '\0' || wordlist_block_hash[i][v] == '\0' || current_hash[v] != wordlist_block_hash[i][v])
	// 		{
	// 			ok = false;
	// 			break;
	// 		}
	// 	}
	// 	if (ok)
	// 		printf("[+] FOUND %s\n", wordlist_block_plain);
	// 	break;
	// }
}

// bool check_hash(const char *hash, char **wordlist_block, int wordlist_block_size)
// {

// 	for (int i = 0; i < wordlist_block_size; i++)
// 	{
// 		char *plain_passwd_test = wordlist_block[i];
// 		unsigned char test_hash[SHA_DIGEST_LENGTH];
// 		size_t plain_length = strlen(wordlist_block[i]);

// 		SHA1((const unsigned char *)plain_passwd_test, plain_length, test_hash);

// 		if (strcmp((const char *)test_hash, hash) == 0)
// 		{
// 			return true;
// 		};
// 	}
// 	return false;
// }

int main(int argc, char *argv[])
{
	if (argc < 3)
		fprintf(stderr, "Usage: '%s' dictionnary_file shasum_file\n", argv[0]), exit(EXIT_FAILURE);
	// int M = strtol(argv[1], NULL, 10);
	// int T = strtol(argv[2], NULL, 10);
	char *dict_file = argv[3];
	char *shasum_file = argv[4];

	// opening files
	FILE *shadow_fd = fopen(shasum_file, "r");
	FILE *wordlist_fd = fopen(dict_file, "r");
	if (shadow_fd == NULL || wordlist_fd == NULL)
		exit(EXIT_FAILURE);

	/* ------------- Loading shadow db into device ------------- */
	char shadow_db[MAX_SHADOW_LENGTH][MAX_LINE_SIZE];
	char **shadow_dbGPU;
	char *line = NULL;
	size_t len = 0;
	int shadow_count = 0;

	while ((getline(&line, &len, shadow_fd)) != -1)
	{
		// sprintf(shadow_db[i], "%s", line);
		strcpy(shadow_db[shadow_count], line);
		shadow_count++;
	}

	cudaMallocManaged(shadow_dbGPU, MAX_SHADOW_LENGTH * MAX_LINE_SIZE * sizeof(char));
	cudaMemcpy(shadow_dbGPU, shadow_db, MAX_SHADOW_LENGTH * MAX_LINE_SIZE * sizeof(char), cudaMemcpyHostToDevice);

	/* ------- Optimizing number of threads & blocks based on 0.5 ratio ------ */
	int M = ceil((double)shadow_count / sqrt((shadow_count / 0.5)));
	int T = ceil((double)shadow_count / (double)M);
	// printf("[DEBUG] Shadow content - head : \n");
	// for (int i = 0; i < 10; i++)
	// {
	// 	printf("[%i] : %s\n", i, shadow_db[i]);
	// }

	/* ------------- Creating first parrallelisation by dividing wordlist (divide&conquer strategy) ------------- */

	int block_counter = 0;
	while (true)
	{
		size_t lines = 0; /** next index to be used with lineBuffer
					(and number of lines already stored)*/
		char *lineBuffer[WL_BLOCK];
		char buf[MAX_LINE_SIZE];
		while (lines < WL_BLOCK && fgets(buf, sizeof(buf), wordlist_fd) != NULL)
		{
			buf[strlen(buf) - 1] = '\0';
			cudaMallocManaged(&lineBuffer[lines], strlen(buf) * sizeof(char));
			cudaMemcpy(lineBuffer[lines], buf, strlen(buf), cudaMemcpyHostToDevice);
			lines++;
		}
		if (lines == 0)
			break;

		block_counter++;
		printf("[+] Assigned block %d (read %zd lines)\n", block_counter, lines);

		char **lineBufferGPU;
		cudaMallocManaged(&lineBufferGPU, lines * MAX_LINE_SIZE * sizeof(char));
		cudaMemcpy(lineBufferGPU, lineBuffer, sizeof(char *) * lines, cudaMemcpyHostToDevice);
		check_hash<<<M, T>>>(lineBufferGPU, lines, shadow_dbGPU);

		// for (int i = 0; i < lines; i++)
		// {
		//     free(lineBuffer[i]);
		// }

		cudaDeviceSynchronize();
	}

	return 0;
}

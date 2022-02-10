// multiattack.c
// Starts multiple instances checking if a password can be found in the dictionnary.
// usage : nb_of_processus dictionnary_file shasum_file

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <openssl/sha.h>
#include <string.h>

#define DEBUG 0
#define WL_BLOCK 1000
#define MAX_LINE_SIZE 1024
#define MAX_SHADOW_LENGTH 5000

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

__global__ void print_myself(const int ID, char **wordlist, int lines, char **shadow_db)
{

	printf("[Block - %d] Cracking hash %s\n", ID);
	for (int i = 0; i < lines; i++)
	{
		printf("[%d] Wordlist : %s\n", ID, wordlist[i]);
	}
}

// __global__ bool check_hash(const char *hash, char **wordlist_block, int wordlist_block_size)
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
	if (argc < 4)
		fprintf(stderr, "Usage: '%s' nb_blocks nb_threads dictionnary_file shasum_file\n", argv[0]), exit(EXIT_FAILURE);
	int M = strtol(argv[1], NULL, 10);
	int T = strtol(argv[2], NULL, 10);
	char *dict_file = argv[3];
	char *shasum_file = argv[4];

	// opening files
	FILE *shadow_fd = fopen(shasum_file, "r");
	FILE *wordlist_fd = fopen(dict_file, "r");
	if (shadow_fd == NULL || wordlist_fd == NULL)
		exit(EXIT_FAILURE);

	/* ------------- Loading shadow db ------------- */
	char shadow_db[MAX_SHADOW_LENGTH][MAX_LINE_SIZE];
	char shadow_dbGPU[MAX_SHADOW_LENGTH][MAX_LINE_SIZE];
	char *line = NULL;
	ssize_t read;
	size_t len = 0;
	int shadow_count = 0;

	while ((read = getline(&line, &len, shadow_fd)) != -1)
	{
		// sprintf(shadow_db[i], "%s", line);
		strcpy(shadow_db[shadow_count], line);
		shadow_count++;
	}
	// cudaMallocManaged(&lineBufferGPU, lines * MAX_LINE_SIZE * sizeof(char));
	// cudaMemcpy(lineBufferGPU, lineBuffer, sizeof(char *) * lines, cudaMemcpyHostToDevice);

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
		printf("[+] Assigned block %d (read %ld lines)\n", block_counter, lines);

		char **lineBufferGPU;
		cudaMallocManaged(&lineBufferGPU, lines * MAX_LINE_SIZE * sizeof(char));
		cudaMemcpy(lineBufferGPU, lineBuffer, sizeof(char *) * lines, cudaMemcpyHostToDevice);
		print_myself<<<M, T>>>(block_counter, lineBufferGPU, lines, (char**) shadow_db);

		// for (int i = 0; i < lines; i++)
		// {
		// 	free(lineBuffer[i]);
		// }

		cudaDeviceSynchronize();
	}

	return 0;
}

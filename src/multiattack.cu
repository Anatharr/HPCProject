// multiattack.c
// Starts multiple instances checking if a password can be found in the dictionnary.
// usage : nb_of_processus dictionnary_file shasum_file

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <openssl/sha.h>
#include <string.h>

#define DEBUG 0
#define WL_BLOCK 1000
#define MAX_LINE_SIZE 1024

__global__ void print_myself(const int ID, char **wordlist, int lines)
{
	printf("[%d] Cracking hash (testing against %d lines)\n", ID, lines);
	// for (int i = 0; i < lines; i++)
	// {
	// 	printf("[%d] Wordlist : %s\n", ID, wordlist[i]);
	// }
}

// __global__ bool check_hash(const char *hash, FILE *wordlist_fd)
// {
// 	// char *readlineGPU(FILE * f)	{
// 	// 	char *line = NULL;

// 	// 	size_t len = 0;
// 	// 	ssize_t read;
// 	// 	if ((read = getline(&line, &len, f)) != -1)
// 	// 	{
// 	// 		line[read - 2] = '\0';

// 	// 		return line;
// 	// 	}
// 	// 	return NULL;
// 	// }

// 	char *plain_text;
// 	while ((plain_text = readline(wordlist_fd)) != NULL)
// 	{
// 		size_t plain_length = strlen(plain_text);
// 		unsigned char test_hash[SHA_DIGEST_LENGTH];
// 		SHA1((const unsigned char *)plain_text, plain_length, test_hash);
// 		if (strcmp((const char *)test_hash, hash) == 0)
// 		{
// 			return true;
// 		};
// 	}
// 	return false;
// }

int main(int argc, char *argv[])
{
	if (argc < 5)
		fprintf(stderr, "Usage: '%s' nb_blocks nb_threads dictionnary_file shasum_file\n", argv[0]), exit(EXIT_FAILURE);
	int M = strtol(argv[1], NULL, 10);
	int T = strtol(argv[2], NULL, 10);
	char *dict_file = argv[3];
	char *shasum_file = argv[4];

	// opening file
	FILE *shadow_fd = fopen(shasum_file, "r");
	FILE *wordlist_fd = fopen(dict_file, "r");
	if (shadow_fd == NULL || wordlist_fd == NULL)
		exit(EXIT_FAILURE);

	/* ---------------- PART 1 ---------------- */
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
		cudaMemcpy(lineBufferGPU, lineBuffer, sizeof(char *)*lines, cudaMemcpyHostToDevice);
		print_myself<<<M, T>>>(block_counter, lineBufferGPU, lines);

		// for (int i = 0; i < lines; i++)
		// {
		// 	free(lineBuffer[i]);
		// }
	}

	cudaDeviceSynchronize();

	/* ---------------- PART 2 ---------------- */
	// char *current_password;
	// while ((current_password = readline(shadow_fd)) != NULL)
	// {
	// 	check_hash<<<M, T>>>(current_password, wordlist_fd);
	// }
	return 0;
}

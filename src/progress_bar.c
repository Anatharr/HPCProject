#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60
#define TOTAL_NUM_OF_BLOCK 20

void updatePBar(int current_block)
{
    double percentage = (double) current_block / TOTAL_NUM_OF_BLOCK; 
    int val = (int)(percentage * 100);
    int pb_width = strlen(PBSTR);
    int lpad = (int)(percentage * pb_width);
    int rpad = pb_width - lpad;
    printf("\r\x1b[33;1m Testing wordlist blocks ... [%d/%d] [%.*s%*s]\x1b[0m",current_block, TOTAL_NUM_OF_BLOCK, lpad, PBSTR, rpad, "");
    fflush(stdout);
}

int main(int argc, char const *argv[])
{
    for (int i = 1; i <= TOTAL_NUM_OF_BLOCK; i++)
    {
        updatePBar(i);
        sleep(1);
    }

    return 0;
}

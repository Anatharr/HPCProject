#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    char *shadow_file = argv[1];
    FILE *shadow_fd = fopen(shadow_file, "r");

    if (shadow_fd == NULL)
    {
        exit(EXIT_FAILURE);
    }

    char *line = NULL;
    ssize_t read;
    size_t len = 0;

    while ((read = getline(&line, &len, shadow_fd)) != -1)
    {
        printf("%s", line);
    }

    return 0;
}

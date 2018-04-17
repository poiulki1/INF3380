#include <stdio.h>
#include <stdlib.h>

typedef struct {

    int m, n, c;
    float **image_data;

} image;

void export_JPEG_file(const char* filename, const unsigned char* image_chars,
               int image_height, int image_width,
               int num_components, int quality);
void import_JPEG_file(const char* filename, unsigned char** image_chars,
               int* image_height, int* image_width,
               int* num_components);


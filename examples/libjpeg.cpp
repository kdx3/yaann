#include <bits/stdc++.h>

#include "jpeglib.h"

int main(int argc, char* argv[])
{
    if (argc < 3) {
        printf("Usage: ./libjpeg IN_BIN_FILE OUT_JPEG_FILE\n");
        return 0;
    }

    FILE *bf;
    const char *bfn = argv[1];
    if ( (bf = fopen(bfn, "rb")) == NULL) {
        printf("can't open %s\n", bfn);
        return 0;
    }
    double *b = (double*)malloc(784*sizeof(double));
    fread(b, sizeof(double), 784, bf);
    fclose(bf);

    FILE *jpegf;
    const char *jpegfn = argv[2];
    if ( (jpegf = fopen(jpegfn, "wb")) == NULL) {
        printf("can't open %s\n", jpegfn);
        return 0;
    }

    jpeg_compress_struct cinfo;
    jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    jpeg_stdio_dest(&cinfo, jpegf);

    cinfo.image_width = 28;
    cinfo.image_height = 28;
    cinfo.input_components = 1;
    cinfo.in_color_space = JCS_GRAYSCALE;
    jpeg_set_defaults(&cinfo);

    jpeg_start_compress(&cinfo, TRUE);
    for (unsigned int i = 0; i < cinfo.image_height; ++i) {
        uint8_t *pimg[1];
        pimg[0] = (uint8_t*)malloc(cinfo.image_width);
        for (unsigned int j = 0; j < cinfo.image_width; ++j) {
            pimg[0][j] = (uint8_t)(b[i*cinfo.image_width + j]*0xff);
        }
        jpeg_write_scanlines(&cinfo, pimg, 1);
        free(pimg[0]);
    }

    jpeg_finish_compress(&cinfo);
    fclose(jpegf);

    free(b);

    jpeg_destroy_compress(&cinfo);

}

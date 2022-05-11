/*
  As mentioned in the instructions, the code for reading and writing 
  png files is taken from: https://gist.github.com/niw/5963798
*/

#include <bits/stdc++.h>
#include <omp.h>
#include <png.h>

using namespace std;

int cols, rows, nthreads;
png_byte type;
png_byte num_bits;
png_bytep *img = NULL;

//The write_png writes an image in rgba format.
void write_png(char *filename) 
{
      int row;
      FILE *fp = fopen(filename, "wb");
      if(!fp) abort();
      png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
      if (!png) abort();
      png_infop info = png_create_info_struct(png);
      if (!info) abort();
      if (setjmp(png_jmpbuf(png))) abort();
      png_init_io(png, fp);
      // Output is 8bit depth, RGBA format.
      png_set_IHDR(png, info, cols, rows, 8, PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
      png_write_info(png, info);
      if (!img) abort();
      png_write_image(png, img);
      png_write_end(png, NULL);
      for(int row = 0; row < rows; row++) {
        free(img[row]);
      }
      free(img);
      fclose(fp);
      png_destroy_write_struct(&png, &info);
}

//The read_png reads an image and extracts rgba matrix. 
void read_png(char *filename) 
{
      FILE *fp = fopen(filename, "rb");
      png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
      if(!png) abort();
      png_infop info = png_create_info_struct(png);
      if(!info) abort();
      if(setjmp(png_jmpbuf(png))) abort();
      png_init_io(png, fp);
      png_read_info(png, info);
      cols  = png_get_image_width(png, info);
      rows  = png_get_image_height(png, info);
      type = png_get_color_type(png, info);
      num_bits  = png_get_bit_depth(png, info);
      if(num_bits == 16)
          png_set_strip_16(png);
      if(type == PNG_COLOR_TYPE_PALETTE)
          png_set_palette_to_rgb(png);
      // PNG_COLOR_TYPE_GRAY_ALPHA is always 8 or 16bit depth.
      if(type == PNG_COLOR_TYPE_GRAY && num_bits < 8)
           png_set_expand_gray_1_2_4_to_8(png);
      if(png_get_valid(png, info, PNG_INFO_tRNS))
           png_set_tRNS_to_alpha(png);
      // These type don't have an alpha channel then fill it with 0xff.
      if(type == PNG_COLOR_TYPE_RGB || type == PNG_COLOR_TYPE_GRAY || type == PNG_COLOR_TYPE_PALETTE)
         png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
      if(type == PNG_COLOR_TYPE_GRAY || type == PNG_COLOR_TYPE_GRAY_ALPHA)
         png_set_gray_to_rgb(png);
      png_read_update_info(png, info);
      if (img) abort();
      img = (png_bytep*)malloc(sizeof(png_bytep) * rows);
      for(int row = 0; row < rows; row++) {
          img[row] = (png_byte*)malloc(png_get_rowbytes(png,info));
      }
      png_read_image(png, img);
      fclose(fp);
      png_destroy_read_struct(&png, &info, NULL);
}


/* 
   This function has two main steps:
   1 => Calculate the pdf(probability distributed function) of histogram of image.
   2 => Calculate cdf(cummulative distributed function) using pdf and thereby a transformation function.
*/
void histogram_equalization() 
{
      const int K = pow(2, num_bits); //Number of intensity Levels
     
      // pdfs for 4 channels red, green, blue and alpha of the image(size = intensity levels)
      int f_r[K], f_g[K], f_b[K], f_alpha[K]; 
      float pdf_r[K], pdf_g[K], pdf_b[K], pdf_alpha[K];  
      for(int i=0; i<K; i++)
      {
          f_r[i] = 0; pdf_r[i] = 0.0; 
          f_g[i] = 0; pdf_g[i] = 0.0;
          f_b[i] = 0; pdf_b[i] = 0.0;
          f_alpha[i] = 0; pdf_alpha[i] = 0.0;
      }
    
      /* Transformation functions for each channel*/
      vector<int> T_r(K, 0), T_g(K, 0), T_b(K, 0), T_alpha(K, 0);
      float sum_r = 0.0, sum_g = 0.0, sum_b=0.0, sum_alpha=0.0;
      
      //Histogram Equalization starts

      double start_time = omp_get_wtime();
      // STEP 1: Obtain the Histogram/PDF
     
      #pragma omp parallel 
      {
          // Calculating frequencies for 4 channels using the best possible parallelization.
          #pragma omp for reduction(+:f_r, f_g, f_b, f_alpha)
          for(int row = 0; row < rows; row++) {
              for(int col = 0; col < cols; col++){
                  png_bytep px = &(img[row][4*col]);
                  f_r[px[0]]++;
                  f_g[px[1]]++;
                  f_b[px[2]]++;
                  f_alpha[px[3]]++;
              }
          }

          // obtaining pdf by normalising final frequencies for each channel
          #pragma omp for schedule(static, 1)
          for(int i=0; i<K; i++)
          {
              pdf_r[i] = (float)(f_r[i])/(rows * cols);
              pdf_g[i] = (float)(f_g[i])/(rows * cols);
              pdf_b[i] = (float)(f_b[i])/(rows * cols);
              pdf_alpha[i] = (float)(f_alpha[i])/(rows * cols);
          }
      }        

      /* STEP 2: Obtain the cumulative distribution function CDF and Calculate the 
      transformation function(per channel) to map the old intensity values to new intensity values.*/
 
      /* cannot parallelise due to data dependencies. This isn't an overhead considering 
         the number of intensity levels are far lesser than actual image size */
      for (int i=0; i<K; i++)
      {
            sum_r += pdf_r[i];
            sum_g += pdf_g[i];
            sum_b += pdf_b[i];
            sum_alpha += pdf_alpha[i];
            T_r[i] = floor((K-1)*sum_r);
            T_g[i] = floor((K-1)*sum_g);
            T_b[i] = floor((K-1)*sum_b);   
            T_alpha[i] = floor((K-1)*sum_alpha);
      }
      
     /* for(int i=0; i<K; i++)
     {
        cout << i << " => (" << T_r[i] << ", " << T_g[i] << ", " << T_b[i] << ", " << T_alpha[i] << ")" << endl; 
     }*/
     // Transforming image using the transformation function 
     #pragma omp parallel for default(none) shared(img, T_r, T_g, T_alpha, T_b, rows, cols) 
     for(int row = 0; row < rows; row++) {
          for(int col = 0; col < cols; col++){
               png_bytep px = &(img[row][4*col]);
               int offset = col * 4;
               img[row][offset] = T_r[px[0]];
               img[row][offset+1] = T_g[px[1]];
               img[row][offset+2] = T_b[px[2]];
               img[row][offset+3] = T_alpha[px[3]];
         }
     }
     double end_time = omp_get_wtime();
     cout << "Elapsed Wall Time for Histogram Equalization: " << (end_time-start_time)  << endl;
}


int main(int argc, char *argv[]) {
      char const * p = argv[3];
      std::stringstream ss(p);
      ss >> nthreads;
      omp_set_num_threads(nthreads);
      read_png(argv[1]);
      cout << "Image Size : " << cols << "x" << rows << endl;
      cout << "Number of threads: " << nthreads << endl;    
      histogram_equalization();
      write_png(argv[2]);
      return 0;
}
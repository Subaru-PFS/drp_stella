/*
*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

#include "fitsio2.h"

int main(int argc, char *argv[])
{
  int              i,j;
  char             infname[255];

  if (argc<2)
  {
    printf("usage: fits-nozero fitsfile1 [fitsfile2 ...]\n");
    printf("\n      Sets all zero values of image to 0.000000001!\n");
    exit(0);
  }
  for (i=1; i<argc; i++)
  {

    strcpy(infname, argv[i]);

    /*      printf("%s\n", infname); */

    {
      int bitpix;
      int anynul, extend, simple;
      int naxis;
      long pcount, gcount;
      long naxes[2];
      long fpixel, nelements;
      fitsfile *ffile;
      int      count,status;
      float * array;
      float nullval;
      char strbuf[256];

      status=0;
      fits_open_file(&ffile, infname, READWRITE, &status);
      fits_read_imghdr(ffile, 2, &simple , &bitpix, &naxis, naxes,
                       &pcount, &gcount, &extend, &status);
      if (status !=0)
        printf("Error %d in file %s\n", status, infname);

      nelements =  naxes[0] * naxes[1];
      array = malloc(sizeof(float) * nelements);
      fpixel = 1; nullval = 0.;
      fits_read_img(ffile, TFLOAT, fpixel, nelements,
                    &nullval, array, &anynul, &status);
      if (status !=0)
        printf("Error %d in file %s\n", status, infname);

      /* set all negative pixels to 0 */
      count=0;
      for (j=0; j<nelements; j++)
        if ( fabs(array[j]) < 0.00000001)
        {
          array[j]=0.00000001;
          count++;
        }
      printf("%d pixels set to 0.00000001\n", count);

      fits_write_img(ffile, TFLOAT, fpixel, nelements,
                     array, &status);

      if (status !=0)
        printf("Error %d in file %s\n", status, infname);
      fits_close_file(ffile, &status);
      free(array);

      if (status !=0)
        printf("Error %d in file %s\n", status, infname);
    }

  }
}

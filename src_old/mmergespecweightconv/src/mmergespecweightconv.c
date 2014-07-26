/*
author:        Andreas Ritter
created:       01/08/2007
last edited:   01/08/2007
compiler:      gcc 4.0
basis machine: Ubuntu Linux 6.06 LTS
*/

#include "mmergespecweightconv.h"

double calcnoisefromoverlap(const double* warr, const double* varra, const double* varrb, const long starta, const long enda, const long endb, const char* out)
{
  double mean, thismean, thisrms, rms;
  long i;

#ifdef __DEBUG_MERGESPECWEIGHTCONV__
  printf("CalcNoiseFromOverlap: warr[enda=%d] = <%.7f>\n", enda, warr[enda]);
#endif

  mean = 0.;
  rms = 0.;
  for (i = starta; i <= enda; i++)
  {
    mean += varra[i];
  }
  mean = mean / (enda - starta + 1.);
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
  printf("MBasis.calcnoisefromoverlap: mean of first order = %.7f\n", mean);
#endif

  FILE *ffile;
  ffile = fopen(out, "w");
  if (ffile == NULL)
  {
    printf("mbasis.calcnoisefromoverlap: Failed to open file fname (=<%s>)\n", out);
    return 0;
  }
  for (i = 0; i < enda - starta + 1; i++)
  {
    thismean = (varra[starta+i] + varrb[i]) / (2.);
    thisrms = sqrt(pow(thismean - (varra[starta + i]), 2) + pow(thismean - (varrb[i]), 2));
    //thisrms = sqrt(pow(1. - (varra[starta + i]/thismean), 2) + pow(1. - (varrb[i]/thismean), 2));
    rms += pow(thisrms, 2);
    fprintf(ffile, "%.8f %.7f\n", warr[starta+i], thisrms);
  }
  fclose(ffile);
  rms = sqrt(rms) / (enda - starta + 1);
//  printf("Hello, world!\n");
  return (rms);
}

int main(int argc, char *argv[])
{
  char *tmpdatfilestr = "/yoda/feros/077.D0235B_F/2006-04-11/lq_hya5_2006-04-12T00-21-05.674_840s_botzfxs_ec1_bldtnRb_text.list";
  char *tmperrfilestr = "/yoda/feros/077.D0235B_F/2006-04-11/lq_hya5_2006-04-12T00-21-05.674_840s_err_obtzx_ec1_bldtnRb_text.list";
  char *tempstra;
  char ***filelist;
  char indatalist[250], inerrlist[250], tempchararr[255], rmsfile[255], rmsoutfile[255], ConvOutFile[255];
  char tempdirarr[255], dirarr[255], dataoutfilename[255], erroutfilename[255], filenamesuffix[255];
  char *line;
  char *tempstr;
  char  *slash = "/";
  char  *tempdir = "";
  char oneword[200];
  long i, j, k, length, dirlength, nextpos, startpos, dataoutfilelength, erroutfilelength;
  long overlapstartpos, overlapstarta, overlapenda, overlapendb, rmsfilelength, suffixlength;
  long rmsoutfilelength, ConvOutFileLength;
  long *nelements;
  int norders, pointpos, overlaplength,centerpos;
  FILE *indatalistfile;
  FILE *inerrlistfile;
  FILE *fdataoutfile;
  FILE *ferroutfile;
  FILE *f_rmsoutfile;
//  FILE *P_ConvOutFile;
  double **wdataarr, **werrarr, **vdataarr, **verrarr, *newwarr, *newvdataarr;//, *convarr;
  double tempwave, dlambda, overlapcenter, convolution, tempdbl;// errij, erri1nextpos, dlambda;
  double ConvA, ConvD, ConvF;

  // --- check if input-file name is given
  if (argc<3)
  {
    printf("mergespecweightconv.main: NOT ENOUGH PARAMETERS SPECIFIED!\n");
    printf("mergespecweightconv.main: USAGE:\n");
    printf("mergespecweightconv.main: mergespecweightconv (char*)datafiles.list (char*)snrfiles.list\n");
    printf("\n datafiles.list:\n");
    printf("    science20060704-0207_botzfxs_ec_bld_001_rb.text\n");
    printf("    science20060704-0207_botzfxs_ec_bld_002_rb.text\n");
    printf("    ...\n");
    printf("\n snrfiles.list:\n");
    printf("    science20060704-0207_botzfxs_ec_bld_001_rb_snr.text\n");
    printf("    science20060704-0207_botzfxs_ec_bld_002_rb_snr.text\n");
    printf("    ...\n");
    printf(" PRE: *_rb.text are outfiles of fitsrebin\n");
    exit(0);
/*    argv[1] = (char*)malloc(sizeof(char)*(strlen(tmpdatfilestr)+1));
    if (argv[1] == NULL)
    {
      printf("MMergeSpecWeightConv.main: NOT ENOUGH MEMORY FOR argv[1]\n");
      exit (EXIT_FAILURE);
    }
    strcpy(argv[1],tmpdatfilestr);
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
    printf("mergespecweightconv.main: argv[1] = %s\n",argv[1]);
#endif
    argv[2] = (char*)malloc(sizeof(char)*(strlen(tmperrfilestr)+1));
    if (argv[2] == NULL)
    {
      printf("MMergeSpecWeightConv.main: NOT ENOUGH MEMORY FOR argv[1]\n");
      exit (EXIT_FAILURE);
    }
    strcpy(argv[2],tmperrfilestr);
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
    printf("mergespecweightconv.main: argv[2] = %s\n",argv[2]);
#endif
    argc = 3;*/
  }
  indatalist[0] = '\0';
  inerrlist[0] = '\0';
  ChArrCat(indatalist, argv[1], 0);
  ChArrCat(inerrlist, argv[2], 0);
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
  printf("MMergeSpecWeightConv.main: argc = %d\n",argc);
#endif

  norders = CountLines(indatalist);
  if (norders != CountLines(inerrlist))
  {
    printf("MMergeSpecWeightConv.main: INFILES DON'T HAVE SAME NUMBER OF ORDERS\n");
    exit (EXIT_FAILURE);
  }
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
  printf("MMergeSpecWeightConv.main: norders = %d\n",norders);
#endif

  nelements = (long*)malloc(sizeof(long) * norders);
  if (nelements == NULL)
  {
    printf("MMergeSpecWeightConv.main: NOT ENOUGH MEMORY FOR nelements\n");
    exit (EXIT_FAILURE);
  }

  /*  convarr = (long*)malloc(sizeof(double) * norders);
    if (convarr == NULL)
    {
    printf("MMergeSpecWeightConv.main: NOT ENOUGH MEMORY FOR convarr\n");
    exit (EXIT_FAILURE);
  }*/

  filelist = (char***)malloc(sizeof(char**) * 2);
  if (filelist == NULL)
  {
    printf("MMergeSpecWeightConv.main: NOT ENOUGH MEMORY FOR filelist\n");
    exit (EXIT_FAILURE);
  }
  for (i=0; i<2; i++)
  {
    filelist[i] = (char**)malloc(sizeof(char*) * norders);
    if (filelist[i] == NULL)
    {
      printf("MMergeSpecWeightConv.main: NOT ENOUGH MEMORY FOR filelist\n");
      exit (EXIT_FAILURE);
    }
  }
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
  printf("MMergeSpecWeightConv.main: memory for filelist allocated\n");
#endif

  strcpy(tempchararr, indatalist);

#ifdef __DEBUG_MERGESPECWEIGHTCONV__
  printf("mergespecweightconv.main: tempchararr = %s\n",tempchararr);
#endif
  
  tempstra = strtok(tempchararr,"/");

#ifdef __DEBUG_MERGESPECWEIGHTCONV__
  printf("mergespecweightconv.main: tempstra = %s\n",tempstra);
#endif

  strcpy(tempdirarr,slash);
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
  printf("mergespecweightconv.main: tempdirarr = %s\n",tempdirarr);
#endif

  length = 1;

#ifdef __DEBUG_MERGESPECWEIGHTCONV__
  printf("mergespecweightconv.main: length = %d\n",length);
#endif

  length = ChArrCat(tempdirarr, tempstra, length);
  strcpy(dirarr,slash);

#ifdef __DEBUG_MERGESPECWEIGHTCONV__
  printf("mergespecweightconv.main: dirarr = %s\n",dirarr);
#endif

  dirlength = 1;
  while(tempstra != NULL)
  {
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
    printf("mergespecweightconv.main\n");
#endif

    length = ChArrCat(tempdirarr,slash,length);

#ifdef __DEBUG_MERGESPECWEIGHTCONV__
    printf("mergespecweightconv.main: tempdirarr = <%s>, length = %d\n",tempdir,length);
#endif

    strcpy(tempchararr,tempstra);

#ifdef __DEBUG_MERGESPECWEIGHTCONV__
    printf("mergespecweightconv.main: tempchararr = <%s>\n",tempchararr);
#endif

    tempstra = strtok(NULL,"/");

#ifdef __DEBUG_MERGESPECWEIGHTCONV__
    printf("mergespecweightconv.main: tempstra = <%s>\n",tempstra);
#endif

    if (tempstra != NULL)
    {
      length = ChArrCat(tempdirarr,tempstra,length);
      dirlength = ChArrCat(dirarr,tempchararr,dirlength);
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
      printf("mergespecweightconv.main: dir = <%s>, dirlength = %d\n",dirarr,dirlength);
#endif
      dirlength = ChArrCat(dirarr,slash,dirlength);
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
      printf("mergespecweightconv.main: dirarr = <%s>, dirlength = %d\n",dirarr,dirlength);
#endif

    }
    //      else
    //  break;
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
    printf("mergespecweightconv.main: tempdirarr = <%s>\n",tempdirarr);
    printf("mergespecweightconv.main: dirarr = <%s>\n",dirarr);
    printf("mergespecweightconv.main: tempstra = <%s>\n",tempstra);
#endif

  }
  //    dirarr[dirlength] = '\0';//strncat(dir,dirarr,dirlength);
  //    dirlength++;
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
  printf("mergespecweightconv.main: dirarr = <%s>, dirlength = %d ready\n",dirarr,dirlength);
#endif

  // --- read input datafiles to filelist
  // --- open file <indatalistfile> with name <indatalist> for reading
  indatalistfile = fopen(indatalist, "r");
  inerrlistfile = fopen(inerrlist, "r");
  if (indatalistfile == NULL)
  {
    printf("mergespecweightconv.main: Failed to open file indatalist =(<%s>)\n", indatalist);
    exit (EXIT_FAILURE);
  }
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
  printf("mergespecweightconv.main: File indatalist =(<%s>) opened\n", indatalist);
#endif
  if (inerrlistfile == NULL)
  {
    printf("mergespecweightconv.main: Failed to open file inerrlist =(<%s>)\n", inerrlist);
    exit (EXIT_FAILURE);
  }
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
  printf("mergespecweightconv.main: File inerrlist =(<%s>) opened\n", inerrlist);
#endif
  // --- read file <indatalistfile> named <indatalist> in <filelist[0]>
  for (i=0; i<norders; i++)
  {
    filelist[0][i] = (char*)malloc(sizeof(char) * 255);
    if (filelist[0][i] == NULL)
    {
      printf("MMergeSpecWeightConv.main: NOT ENOUGH MEMORY FOR filelist\n");
      exit (EXIT_FAILURE);
    }
    filelist[1][i] = (char*)malloc(sizeof(char) * 255);
    if (filelist[1][i] == NULL)
    {
      printf("MMergeSpecWeightConv.main: NOT ENOUGH MEMORY FOR filelist\n");
      exit (EXIT_FAILURE);
    }
    filelist[0][i][0] = '\0';
    filelist[1][i][0] = '\0';
    dirlength = ChArrCat(filelist[0][i], dirarr, 0);
    ChArrCat(filelist[1][i], dirarr, 0);
    // --- read line of indatalist
    line = fgets(oneword, 200, indatalistfile);
    if (line == NULL)
    {
      printf("mergespecweightconv.main: Failed to read line %d of file indatalist =(<%s>)\n", i, indatalist);
      exit (EXIT_FAILURE);
    }
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
    printf("mergespecweightconv.main: oneword No i=%d of file indatalist(=%s) = <%s>\n", i, indatalist, oneword);
#endif

    tempstr = strtok(line," \t\n");
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
    printf("mergespecweightconv.main: tempstr = <%s>\n", tempstr);
#endif
    if (tempstr[0] == '/')
      ChArrCat(filelist[0][i], tempstr, 0);
    else
      ChArrCat(filelist[0][i], tempstr, dirlength);

#ifdef __DEBUG_MERGESPECWEIGHTCONV__
    printf("mergespecweightconv.main: filelist[0][%d] = <%s>\n", i, filelist[0][i]);
#endif
    // --- read line of inerrlist
    line = fgets(oneword, 200, inerrlistfile);
    if (line == NULL)
    {
      printf("mergespecweightconv.main: Failed to read line %d of file inerrlist =(<%s>)\n", i, inerrlist);
      exit (EXIT_FAILURE);
    }
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
    printf("mergespecweightconv.main: oneword No i=%d of file inerrlist(=%s) = <%s>\n", i, inerrlist, oneword);
#endif

    tempstr = strtok(line," \t\n");
    ChArrCat(filelist[1][i], tempstr, dirlength);

#ifdef __DEBUG_MERGESPECWEIGHTCONV__
    printf("mergespecweightconv.main: filelist[1][%d] = <%s>\n", i, filelist[1][i]);
#endif

  }
  fclose(indatalistfile);
  fclose(inerrlistfile);

  // allocate memory for data arrays
  wdataarr = (double**)malloc(sizeof(double*) * norders);
  if (wdataarr == NULL)
  {
    printf("MMergeSpecWeightConv.main: NOT ENOUGH MEMORY FOR wdataarr\n");
    exit (EXIT_FAILURE);
  }
  werrarr = (double**)malloc(sizeof(double*) * norders);
  if (werrarr == NULL)
  {
    printf("MMergeSpecWeightConv.main: NOT ENOUGH MEMORY FOR werrarr\n");
    exit (EXIT_FAILURE);
  }
  vdataarr = (double**)malloc(sizeof(double*) * norders);
  if (vdataarr == NULL)
  {
    printf("MMergeSpecWeightConv.main: NOT ENOUGH MEMORY FOR vdataarr\n");
    exit (EXIT_FAILURE);
  }
  verrarr = (double**)malloc(sizeof(double*) * norders);
  if (verrarr == NULL)
  {
    printf("MMergeSpecWeightConv.main: NOT ENOUGH MEMORY FOR verrarr\n");
    exit (EXIT_FAILURE);
  }
  for (i = 0; i < norders; i++)
  {
    nelements[i] = CountLines(filelist[0][i]);
    if (nelements[i] != CountLines(filelist[1][i]))
    {
      printf("MMergeSpecWeightConv.main: %s and %s don't have same number of elements! EXITING\n", filelist[0][i], filelist[1][i]);
      exit (EXIT_FAILURE);
    }
    wdataarr[i] = (double*)malloc(sizeof(double) * nelements[i]);
    if (wdataarr[i] == NULL)
    {
      printf("MMergeSpecWeightConv.main: NOT ENOUGH MEMORY FOR wdataarr[%d]\n", i);
      exit (EXIT_FAILURE);
    }
    werrarr[i] = (double*)malloc(sizeof(double) * nelements[i]);
    if (werrarr[i] == NULL)
    {
      printf("MMergeSpecWeightConv.main: NOT ENOUGH MEMORY FOR werrarr[%d]\n", i);
      exit (EXIT_FAILURE);
    }
    vdataarr[i] = (double*)malloc(sizeof(double) * nelements[i]);
    if (vdataarr[i] == NULL)
    {
      printf("MMergeSpecWeightConv.main: NOT ENOUGH MEMORY FOR vdataarr[%d]\n", i);
      exit (EXIT_FAILURE);
    }
    verrarr[i] = (double*)malloc(sizeof(double) * nelements[i]);
    if (verrarr[i] == NULL)
    {
      printf("MMergeSpecWeightConv.main: NOT ENOUGH MEMORY FOR verrarr[%d]\n", i);
      exit (EXIT_FAILURE);
    }
  }

#ifdef __DEBUG_MERGESPECWEIGHTCONV__
  printf("mergespecweightconv.main: Memory for data arrays allocated\n");
#endif

  // read data files to data arrays
  for (i = 0; i < norders; i++)
  {
    //#ifdef __DEBUG_MERGESPECWEIGHTCONV__
    //    printf("mergespecweightconv.main: i = %d\n", i);
    //#endif
    indatalistfile = fopen(filelist[0][i], "r");
    inerrlistfile = fopen(filelist[1][i], "r");
    if (indatalistfile == NULL)
    {
      printf("mergespecweightconv.main: Failed to open file filelist[0][%d] =(<%s>)\n", i, indatalist);
      exit (EXIT_FAILURE);
    }
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
    printf("mergespecweightconv.main: File filelist[0][%d] =(<%s>) opened\n", i, filelist[0][i]);
#endif
    if (inerrlistfile == NULL)
    {
      printf("mergespecweightconv.main: Failed to open file filelist[1][%d] =(<%s>)\n", i, filelist[1][i]);
      exit (EXIT_FAILURE);
    }
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
    printf("mergespecweightconv.main: File filelist[1][%d] =(<%s>) opened\n", i, filelist[1][i]);
#endif

    for (j = 0; j < nelements[i]; j++)
    {
      //#ifdef __DEBUG_MERGESPECWEIGHTCONV__
      //      printf("mergespecweightconv.main: j = %d\n", j);
      //#endif
      // --- read line of indatalist
      line = fgets(oneword, 200, indatalistfile);
      if (line == NULL)
      {
        printf("mergespecweightconv.main: Failed to read line %d of file filelist[0][%d] =(<%s>)\n", j, i, filelist[0][i]);
        exit (EXIT_FAILURE);
      }
      //#ifdef __DEBUG_MERGESPECWEIGHTCONV__
      //      printf("mergespecweightconv.main: oneword No i=%d of file filelist[0][%d](=%s) = <%s>\n", j, i, filelist[0][i], oneword);
      //#endif
      wdataarr[i][j] = atof(strtok(line," "));
      vdataarr[i][j] = atof(strtok(NULL," "));

      // --- read line of inerrlist
      line = fgets(oneword, 200, inerrlistfile);
      if (line == NULL)
      {
        printf("mergespecweightconv.main: Failed to read line %d of file inerrlist =(<%s>)\n", i, inerrlist);
        exit (EXIT_FAILURE);
      }
      //#ifdef __DEBUG_MERGESPECWEIGHTCONV__
      //      printf("mergespecweightconv.main: oneword No i=%d of file inerrlist(=%s) = <%s>\n", i, inerrlist, oneword);
      //#endif

      werrarr[i][j] = atof(strtok(line," "));
      verrarr[i][j] = atof(strtok(NULL," "));
    }
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
    printf("mergespecweightconv.main: wdataarr[%d][%d] = %.7f\n", i, nelements[i]-1, wdataarr[i][nelements[i]-1]);
    printf("mergespecweightconv.main: vdataarr[%d][%d] = %.7f\n", i, nelements[i]-1, vdataarr[i][nelements[i]-1]);
    printf("mergespecweightconv.main: werrarr[%d][%d] = %.7f\n", i, nelements[i]-1, werrarr[i][nelements[i]-1]);
    printf("mergespecweightconv.main: verrarr[%d][%d] = %.7f\n", i, nelements[i]-1, verrarr[i][nelements[i]-1]);
#endif
    fclose(indatalistfile);
    fclose(inerrlistfile);
  }

  // --- merge orders to single file
  // --- allocate memory for new arrays
  newwarr = (double*)malloc(sizeof(double));
  newvdataarr = (double*)malloc(sizeof(double));
  //newverrarr = (double*)malloc(sizeof(double));
  length = 0;
  // --- set first elements for new arrays
  tempwave = wdataarr[0][0];
  newwarr[0] = tempwave;
  //newverrarr[0] = verrarr[0][0];

  // --- reset starting position in overlapping region for next order
  startpos = 0;
  overlapstartpos = 0;

  // set rmsoutfile
  pointpos = LastCharPosInChArr(indatalist, '.');
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
  printf("mergespecweightconv.main: pointpos(=LastCharPosInChArr(indatalist=%s)) = %d\n", indatalist, pointpos);
#endif
  if (pointpos == -1)
  {
    printf("mergespecweightconv.main: pointpos == 0 => exiting\n");
    exit (EXIT_FAILURE);
  }
  rmsoutfile[0] = '\0';
  for (k = 0; k < pointpos; k++)
  {
    rmsoutfile[k] = indatalist[k];
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
    printf("mergespecweightconv.main: rmsoutfile = <%s>\n", rmsoutfile);
#endif

  }
  rmsoutfile[k+1] = '\0';
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
  printf("mergespecweightconv.main: rmsoutfile = <%s>\n", rmsoutfile);
#endif
  rmsoutfilelength = k;

  suffixlength = ChArrCat(filenamesuffix, "_rms.dat", 0);
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
  printf("mergespecweightconv.main: starting ChArrCat(rmsoutfile = %s, filenamesuffix = %s, rmsoutfilelength = %d), suffixlength = %d\n", rmsoutfile, filenamesuffix, rmsoutfilelength, suffixlength);
#endif
  rmsoutfilelength = ChArrCat(rmsoutfile, filenamesuffix, rmsoutfilelength);
  if (rmsoutfilelength == 0)
  {
    printf("mergespecweightconv.main: ChArrCat(rmsoutfile, filenamesuffix, rmsoutfilelength) returned 0 => exiting\n");
    exit (EXIT_FAILURE);
  }

  f_rmsoutfile = fopen(rmsoutfile, "w");
  if (f_rmsoutfile == NULL)
  {
    printf("mergespecweightconv.main: Failed to open file rmsoutfile =(<%s>)\n", rmsoutfile);
    exit (EXIT_FAILURE);
  }
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
  printf("mergespecweightconv.main: File rmsoutfile =(<%s>) opened\n", rmsoutfile);
#endif


  // set ConvOutFile
  pointpos = LastCharPosInChArr(indatalist, '.');
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
  printf("mergespecweightconv.main: pointpos(=LastCharPosInChArr(indatalist=%s)) = %d\n", indatalist, pointpos);
#endif
  if (pointpos == -1)
  {
    printf("mergespecweightconv.main: pointpos == 0 => exiting\n");
    exit (EXIT_FAILURE);
  }
  ConvOutFile[0] = '\0';
  for (k = 0; k < pointpos; k++)
  {
    ConvOutFile[k] = indatalist[k];
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
    printf("mergespecweightconv.main: ConvOutFile = <%s>\n", ConvOutFile);
#endif
  }
  ConvOutFile[k+1] = '\0';
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
  printf("mergespecweightconv.main: ConvOutFile = <%s>\n", ConvOutFile);
#endif
  ConvOutFileLength = k;

  suffixlength = ChArrCat(filenamesuffix, "_conv.dat", 0);
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
  printf("mergespecweightconv.main: starting ChArrCat(ConvOutFile = %s, filenamesuffix = %s, ConvOutFileLength = %d), suffixlength = %d\n", ConvOutFile, filenamesuffix, ConvOutFileLength, suffixlength);
#endif
  ConvOutFileLength = ChArrCat(ConvOutFile, filenamesuffix, ConvOutFileLength);
  if (ConvOutFileLength == 0)
  {
    printf("mergespecweightconv.main: ChArrCat(ConvOutFile, filenamesuffix, ConvOutFileLength) returned 0 => exiting\n");
    exit (EXIT_FAILURE);
  }

//  P_ConvOutFile = fopen(ConvOutFile, "w");
//  if (P_ConvOutFile == NULL)
//  {
//    printf("mergespecweightconv.main: Failed to open file ConvOutFile =(<%s>)\n", ConvOutFile);
//    exit (EXIT_FAILURE);
//  }
//#ifdef __DEBUG_MERGESPECWEIGHTCONV__
//  printf("mergespecweightconv.main: File ConvOutFile =(<%s>) opened\n", ConvOutFile);
//#endif

  // --- for every order
  for (i = 0; i < norders; i++)
  {
    // --- reset position in overlapping region for next order
    nextpos = 0;
    overlaplength = 0;
    for (j = startpos; j < nelements[i]; j++)
    {
      length++;
      // --- reallocate memory for new arrays
      newwarr = (double*)realloc(newwarr, sizeof(double) * length);
      newvdataarr = (double*)realloc(newvdataarr, sizeof(double) * length);
      //newverrarr = (double*)realloc(newverrarr, sizeof(double) * length);
      // --- jump to position next to overlapping region
      //      do
      //      {
      //        tempwave = wdataarr[i][j];
      //        j++;
      //      }
      //      while ((tempwave < newwarr[length - 1]) && (j < nelements[i]));
      //      newwarr[length - 1] = tempwave;
      newwarr[length - 1] = wdataarr[i][j];

      //#ifdef __DEBUG_MERGESPECWEIGHTCONV__
      //      printf("mergespecweightconv.main: newwarr[%d] = %.7f\n", length -1, newwarr[length - 1]);
      //#endif

      // --- until 2nd last order
      if (i < norders - 1)
      {
        // --- inside overlapping region?
        if (wdataarr[i+1][nextpos] == newwarr[length - 1])
        {
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
          printf("mergespecweightconv.main: wdataarr[i=%d][j=%d] = %.7f, wdataarr[i+1=%d][nextpos=%d] = %.7f, newwarr[length-1=%d] = %.7f\n", i, j, wdataarr[i][j], i+1, nextpos, wdataarr[i+1][nextpos], length-1, newwarr[length - 1]);
          printf("mergespecweightconv.main: vdataarr[i=%d][j=%d] = %.7f, verrarr[i=%d][j=%d] = %.7f\n", i, j, vdataarr[i][j], i, j, verrarr[i][j]);
          printf("mergespecweightconv.main: vdataarr[i+1=%d][nextpos=%d] = %.7f, verrarr[i+1=%d][nextpos=%d] = %.7f\n", i+1, nextpos, vdataarr[i+1][nextpos], i+1, nextpos, verrarr[i+1][nextpos]);
#endif

          if (overlaplength == 0)
          {
            overlapstartpos = j;
            overlaplength = nelements[i] - j;
            centerpos = j + (overlaplength / 2);
            overlapcenter = wdataarr[i][centerpos];
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
            printf("mergespecweightconv.main: order %d: overlapstartpos = %d, overlaplength = %d, centerpos = %d, overlapcenter = %.7f\n", i, overlapstartpos, overlaplength, centerpos, overlapcenter);
#endif
          }

          // --- calculate convolution function
          if (newwarr[length - 1] < overlapcenter)
          {
            // f(x) = ConvA(x)²
            // x = j - overlapstartpos
            ConvA = 2. / pow(overlaplength, 2);
            convolution = ConvA * pow(j - overlapstartpos, 2);//(4. / pow(overlaplength,2)) * (j - overlapstartpos);
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
            printf("mergespecweightconv.main: order %d: ConvA = %.7f, convolution = %.7f\n", i, ConvA, convolution);
#endif
          }
          else
          {
            // f(x) = ConvD(x + ConvF)² + 1.
            ConvD = 0. - (2. / pow(overlaplength, 2));
            ConvF = 0. - overlaplength;
            convolution = (ConvD * pow((j - overlapstartpos) + ConvF, 2)) + 1.;
          }

//          fprintf(P_ConvOutFile, "%.7f %.7f\n", newwarr[length-1], convolution);

          // --- calculate new flux value by adding the snr-weighted flux values from both orders involved in overlapping region
          newvdataarr[length - 1] = vdataarr[i][j] * verrarr[i][j] * (1. - convolution);
          newvdataarr[length - 1] += vdataarr[i+1][nextpos] * verrarr[i+1][nextpos] * convolution;
          newvdataarr[length - 1] = newvdataarr[length - 1] / ((verrarr[i][j] * (1. - convolution)) + (verrarr[i+1][nextpos] * convolution));

#ifdef __DEBUG_MERGESPECWEIGHTCONV__
          printf("MMergeSpecWeightConv: j = %d, nelements[i=%d]=%d\n", j, i, nelements[i]);
#endif
          if (j == (nelements[i] - 1))
          {
            overlapstarta = overlapstartpos;
            overlapenda = j;
            overlapendb = nextpos;
            // --- build output-file names
            pointpos = LastCharPosInChArr(filelist[0][i], '.');
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
            printf("mergespecweightconv.main: pointpos(=LastCharPosInChArr(filelist[0][%d]=%s)) = %d\n", i, filelist[0][i], pointpos);
#endif
            if (pointpos == -1)
            {
              printf("mergespecweightconv.main: pointpos == 0 => exiting\n");
              exit (EXIT_FAILURE);
            }
            rmsfile[0] = '\0';
            // --- set dataoutfilename to indatalist until position of last point
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
            printf("mergespecweightconv.main: rmsfile = <%s>\n", rmsfile);
#endif
            for (k = 0; k < pointpos; k++)
            {
              rmsfile[k] = filelist[0][i][k];
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
              printf("mergespecweightconv.main: rmsfile = <%s>\n", rmsfile);
#endif

            }
            rmsfile[k+1] = '\0';
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
            printf("mergespecweightconv.main: rmsfile = <%s>\n", rmsfile);
#endif
            rmsfilelength = k;

            suffixlength = ChArrCat(filenamesuffix, "_rms.text", 0);
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
            printf("mergespecweightconv.main: starting ChArrCat(rmsfile = %s, filenamesuffix = %s, rmsfilelength = %d), suffixlength = %d\n", rmsfile, filenamesuffix, rmsfilelength, suffixlength);
#endif
            rmsfilelength = ChArrCat(rmsfile, filenamesuffix, rmsfilelength);
            if (rmsfilelength == 0)
            {
              printf("mergespecweightconv.main: ChArrCat(rmsfile, filenamesuffix, rmsfilelength) returned 0 => exiting\n");
              exit (EXIT_FAILURE);
            }
            tempdbl = calcnoisefromoverlap(wdataarr[i], vdataarr[i], vdataarr[i+1], overlapstarta, overlapenda, overlapendb, rmsfile);
            fprintf(f_rmsoutfile, "%d %.7f\n", i, tempdbl);
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
            printf("MMergeSpecWeightConv: tempdbl = %.7f\n", tempdbl);
#endif
            //          exit(0);
          }

          // --- increase nextpos by 1
          nextpos++;
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
          printf("mergespecweightconv.main: nextpos = %d\n", nextpos);
#endif

        } // end if (wdataarr[i+1][nextpos] == newwarr[length - 1])
        else
        {
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
          printf("mergespecweightconv.main: nextpos = %d\n", nextpos);
          if (nextpos > 0)
          {
            printf("mergespecweightconv.main: no overlapping\n");
            printf("mergespecweightconv.main: wdataarr[i=%d][j=%d] = %.7f, wdataarr[i+1=%d][nextpos=%d] = %.7f, newwarr[length-1=%d] = %.7f\n", i, j, wdataarr[i][j], i+1, nextpos, wdataarr[i+1][nextpos], length-1, newwarr[length - 1]);
            printf("mergespecweightconv.main: vdataarr[i=%d][j=%d] = %.7f, verrarr[i=%d][j=%d] = %.7f\n", i, j, vdataarr[i][j], i, j, verrarr[i][j]);
          }
#endif
          newvdataarr[length - 1] = vdataarr[i][j];
          //newverrarr[length - 1] = verrarr[i][j];

          // --- fill gaps with zeros
          if (j == nelements[i] - 1)
          {
            dlambda = newwarr[length - 1] - newwarr[length - 2];
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
            printf("mergespecweightconv.main: dlambda = %.7f\n", dlambda);
#endif
            while(newwarr[length - 1] + dlambda < wdataarr[i+1][0])
            {
              length++;
              // --- reallocate memory for new arrays
              newwarr = (double*)realloc(newwarr, sizeof(double) * length);
              newvdataarr = (double*)realloc(newvdataarr, sizeof(double) * length);
              //newverrarr = (double*)realloc(newverrarr, sizeof(double) * length);
              newwarr[length - 1] = newwarr[length - 2] + dlambda;
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
              printf("mergespecweightconv.main: gap-newwarr[%d] = %.7f\n", length - 1, newwarr[length - 1]);
              printf("mergespecweightconv.main: wdataarr[%d][0] = %.7f\n", i+1, wdataarr[i+1][0]);
#endif
              newvdataarr[length - 1] = 0.;
              //newverrarr[length - 1] = 0.;
            }
          }
        }
      } // end if (i < norders - 1)
      else// last order
      {
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
        printf("mergespecweightconv.main: nextpos = %d\n", nextpos);
        if (nextpos > 0)
        {
          printf("mergespecweightconv.main: last order\n");
          printf("mergespecweightconv.main: wdataarr[i=%d][j=%d] = %.7f, wdataarr[i+1=%d][nextpos=%d] = %.7f, newwarr[length-1=%d] = %.7f\n", i, j, wdataarr[i][j], i+1, nextpos, wdataarr[i+1][nextpos], length-1, newwarr[length - 1]);
          printf("mergespecweightconv.main: vdataarr[i=%d][j=%d] = %.7f, verrarr[i=%d][j=%d] = %.7f\n", i, j, vdataarr[i][j], i, j, verrarr[i][j]);
        }
#endif
        // --- last order
        newvdataarr[length - 1] = vdataarr[i][j];
        //newverrarr[length - 1] = verrarr[i][j];
      }
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
      printf("mergespecweightconv.main: newwarr[%d] = %.7f, newvdataarr[%d] = %.7f\n", length - 1, newwarr[length - 1], length - 1, newvdataarr[length - 1]);
#endif

    }// end for (j = startpos; j < nelements[i]; j++)
    startpos = nextpos;
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
    printf("mergespecweightconv.main: startpos nextorder = %d\n", startpos);
#endif

  }// end for (i = 0; i < norders; i++)
  fclose(f_rmsoutfile);
//  fclose(P_ConvOutFile);

  // --- write results to outfiles
  // --- build output-file names
  pointpos = LastCharPosInChArr(indatalist, '.');
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
  printf("mergespecweightconv.main: pointpos(=LastCharPosInChArr(indatalist=%s)) = %d\n", indatalist, pointpos);
#endif
  if (pointpos == -1)
  {
    printf("mergespecweightconv.main: pointpos == 0 => exiting\n");
    exit (EXIT_FAILURE);
  }
  dataoutfilename[0] = '\0';
  // --- set dataoutfilename to indatalist until position of last point
  for (i = 0; i < pointpos; i++)
  {
    dataoutfilename[i] = indatalist[i];
  }
  dataoutfilename[i+1] = '\0';
  dataoutfilelength = i;

  // --- error image
  pointpos = LastCharPosInChArr(inerrlist, '.');
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
  printf("mergespecweightconv.main: pointpos(=LastCharPosInChArr(inerrlist=%s)) = %d\n", inerrlist, pointpos);
#endif
  if (pointpos == -1)
  {
    printf("mergespecweightconv.main: pointpos == -1 => exiting\n");
    exit (EXIT_FAILURE);
  }
  /*erroutfilename[0] = '\0';
  // --- set dataoutfilename to indatalist until position of last point
  for (i = 0; i < pointpos; i++)
  {
    erroutfilename[i] = inerrlist[i];
  }
  erroutfilename[i+1] = '\0';

  erroutfilelength = i;*/
  // --- set filename suffix for output files to _m
  filenamesuffix[0] = '_';
  filenamesuffix[1] = 'm';
  filenamesuffix[2] = '.';
  filenamesuffix[3] = 't';
  filenamesuffix[4] = 'e';
  filenamesuffix[5] = 'x';
  filenamesuffix[6] = 't';
  filenamesuffix[7] = '\0';
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
  printf("mergespecweightconv.main: starting ChArrCat(dataoutfilename = %s, filenamesuffix = %s, dataoutfilelength = %d)\n", dataoutfilename, filenamesuffix, dataoutfilelength);
#endif
  dataoutfilelength = ChArrCat(dataoutfilename, filenamesuffix, dataoutfilelength);
  if (dataoutfilelength == 0)
  {
    printf("mergespecweightconv.main: ChArrCat(dataoutfilename, '_rb', dataoutfilelength) returned 0 => exiting\n");
    exit (EXIT_FAILURE);
  }

#ifdef __DEBUG_MERGESPECWEIGHTCONV__
  printf("mergespecweightconv.main: dataoutfilelength = %d, pointpos = %d\n", dataoutfilelength, pointpos);
  printf("mergespecweightconv.main: strlen(indatalist(=%s)) returned %d\n", indatalist, strlen(indatalist));
#endif

#ifdef __DEBUG_MERGESPECWEIGHTCONV__
  printf("mergespecweightconv.main: dataoutfilename = <%s>\n", dataoutfilename);
#endif
  /*// --- erroutfile
  #ifdef __DEBUG_MERGESPECWEIGHTCONV__
  printf("mergespecweightconv.main: starting ChArrCat(erroutfilename = %s, filenamesuffix = %s, i = %d)\n", erroutfilename, filenamesuffix, i);
  #endif
  erroutfilelength = ChArrCat(erroutfilename, filenamesuffix, erroutfilelength);
  if (erroutfilelength == 0)
  {
    printf("mergespecweightconv.main: ChArrCat(erroutfilename, filenamesuffix, erroutfilelength) returned 0 => exiting\n");
    exit (EXIT_FAILURE);
  }
  #ifdef __DEBUG_MERGESPECWEIGHTCONV__
  printf("mergespecweightconv.main: erroutfilename = <%s>\n", erroutfilename);
  #endif*/

  // --- open fdataoutfile
  fdataoutfile = fopen(dataoutfilename, "w");
  if (fdataoutfile == NULL)
  {
    printf("mergespecweightconv.main: Failed to open file dataoutfilename (=<%s>)\n", dataoutfilename);
    exit (EXIT_FAILURE);
  }
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
  printf("mergespecweightconv.main: dataoutfilename = <%s> opened\n", dataoutfilename);
  printf("mergespecweightconv.main: length - 1 = <%d>\n", length - 1);
#endif
  /* --- open ferroutfile
  ferroutfile = fopen(erroutfilename, "w");
  if (ferroutfile == NULL)
  {
    printf("mergespecweightconv.main: Failed to open file erroutfilename (=<%s>)\n", erroutfilename);
    exit (EXIT_FAILURE);
  }
  #ifdef __DEBUG_MERGESPECWEIGHTCONV__
  printf("mergespecweightconv.main: erroutfilename = <%s> opened\n", erroutfilename);
  #endif*/
  for (i = 0; i < length - 1; i++)
  {
    if (i < 3)
    {
      printf("mergespecweightconv.main: writing outfiles: i = %d: newwarr[%d] = %.7f, newvdataarr[%d] = %.7f\n", i, i, newwarr[i], i, newvdataarr[i]);
      //printf("mergespecweightconv.main: writing outfiles: i = %d: newwarr[%d] = %.7f, newverrarr[%d] = %.7f\n", i, i, newwarr[i], i, newverrarr[i]);
    }
    fprintf(fdataoutfile, "%.8f %.7f\n", newwarr[i], newvdataarr[i]);
    //fprintf(ferroutfile, "%.8f %.7f\n", newwarr[i], newverrarr[i]);
  }
  fclose(fdataoutfile);
#ifdef __DEBUG_MERGESPECWEIGHTCONV__
  printf("mergespecweightconv.main: dataoutfilename = <%s> closed\n", dataoutfilename);
#endif
  /*fclose(ferroutfile);
  #ifdef __DEBUG_MERGESPECWEIGHTCONV__
  printf("mergespecweightconv.main: erroutfilename = <%s> closed\n", erroutfilename);
  #endif*/

  printf("mergespecweightconv ready\n");

  return EXIT_SUCCESS;
}

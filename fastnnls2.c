/* fastnnls.c implementation Jonas F and Wayne Broucher */

#ifndef MAXFLOAT
#define MAXFLOAT 3.40282347e+38F
#endif

#include "headerfiles/common.h"
#include "headerfiles/fastnnls.h"
#include "headerfiles/util.h"

//#include <math.h>
//#include <gsl/gsl_linalg.h>
//#include "util.h"

/* Heuristic value  */
//#define EPS  1.0e-12
#define EPS 2.22045E-16 // from numpy.finfo(flaot) 2.2204460492503131e-16

// from eps.c:
//#define EPS 1.19209e-07

/* amd46 redhat: did not find it in /usr/include/math.h */
//#define MAXFLOAT 3.40282347e+38F
//#define MAXFLOAT 3.40282347e+38F

/*
static void printMat(gsl_matrix *matrix, char *msg)
{
    printf("%s\n", msg);
    gsl_matrix_fprintf(stdout, matrix, "%7.5f");
}

static void printVec(gsl_vector *vector, char *msg)
{
    printf("%s\n", msg);
    gsl_vector_fprintf(stdout, vector, "%7.5f");
}

static void printInt(int n, int *p, char *msg)
{
    int i;

    printf("%s: \n", msg);
    for (i = 0; i < n; i++)
	printf(" %d", p[i]);
    printf("\n");
}
*/

static void solveEquation(int n, gsl_vector *z, int nzp, int *PP,
	gsl_permutation *perm, gsl_matrix *XtX, gsl_vector *Xty,
	gsl_matrix *workMat, gsl_vector *workVec)
{
    int i, j, s;
    double v;

    gsl_vector_set_zero(z);

    if (nzp == 0)
	return;
    
    workMat->size1 = workMat->size2 = nzp;
    workVec->size = nzp;
    perm->size = nzp;
    z->size = nzp;
    
    for (i=0; i<nzp; i++) {
      for (j=0; j<nzp; j++) 
	gsl_matrix_set(workMat,i,j,gsl_matrix_get(XtX,*(PP+i),*(PP+j)));
      gsl_vector_set(workVec,i,gsl_vector_get(Xty,*(PP+i)));
    }

    /* gsl_permutation_init(perm); */
    /* P(nzp) */
    
    gsl_linalg_LU_decomp(workMat,perm,&s);
    gsl_linalg_LU_solve(workMat,perm,workVec,z);
    //  below is much worse, for some reason (39 vs 22 secs on 1000 runs)
 /*    gsl_linalg_cholesky_decomp(workMat); */
/*     gsl_linalg_cholesky_solve(workMat, workVec, z); */

    workMat->size1=workMat->size2=n;
    workVec->size=n;
    perm->size=n;
    z->size=n;

/*  need to shuffle output so that have values at indices */
/*  this involves moving values up and zeroing what was left behind */

    for (i=nzp-1; i>=0; i--) { // note >= -> >
      v=gsl_vector_get(z,i);
      gsl_vector_set(z,*(PP+i),v);
      if (*(PP+i)>i)
	gsl_vector_set(z,i,0);
    }
}
// x goes to inf. 
static void determineW(gsl_matrix *XtX,gsl_vector *Xty,gsl_vector *x,gsl_vector *w) {
  gsl_blas_dgemv(CblasNoTrans,-1.0,XtX,x,0.0,w);
  gsl_vector_add(w,Xty);
}

void fastnnls(gsl_matrix *XtX, gsl_vector *Xty, gsl_vector *x, gsl_vector *w,double tol) {

  //  int nq;
  int i,n, nzz, nzp, iter, itmax, t;
  int *P, *Z, *PP, *ZZ;
  double v, vv, alpha;
  gsl_vector *z, *workVec;
  gsl_matrix *workMat;
  gsl_permutation *perm;

  //  printf("eps: %e\n",geteps());
  
  n=XtX->size1;

  if(n!=XtX->size2) {
    printf("XtX must be a square matrix (got %d x %d)\n",n,(int) XtX->size2);
    return;
  }

  if (n != Xty->size) {
    printf("XtX must be same size as Xty (got %d vs %d)\n", n,(int) Xty->size);
    return;
  }

  if (n != x->size) {
    printf("XtX must be same size as x (got %d vs %d)\n", n, (int) x->size);
    return;
  }

  if (n != w->size) {
    printf("XtX must be same size as w (got %d vs %d)\n", n, (int) w->size);
    return;
  }

//    float eps=geteps();
  float limit=1.0;
  if (tol <= limit)
    tol=10*EPS*normMatrix(XtX) * XtX->size1; // from function
  
  P = (int *) malloc(n*sizeof(int));
  PP = (int *) malloc(n*sizeof(int));
  Z = (int *) malloc(n*sizeof(int));
  ZZ = (int *) malloc(n*sizeof(int));
  perm = gsl_permutation_calloc(n);
  z = gsl_vector_calloc(n);
  workVec = gsl_vector_alloc(n);
  workMat = gsl_matrix_alloc(n,n);
  gsl_vector *temp1=gsl_vector_calloc(n);
  gsl_vector *temp2=gsl_vector_calloc(n);

  for(i=0; i<n; i++)
    P[i]=0;

  for(i=0; i<n; i++)
    ZZ[i]=Z[i]=i;
  
  gsl_vector_set_zero(x);

  nzz=n;

  gsl_blas_dgemv(CblasNoTrans,1.0,XtX,x,0.0,temp1); // jf
  gsl_vector_memcpy(temp2,Xty);
  gsl_vector_sub(temp2,temp1);
  gsl_vector_memcpy(w,temp2);

  iter=0;
  itmax=1; // jf

/* matlab: outer loop: while any(Z) && any(w(ZZ) > tol) */

  while(anyL(Z,n) && anyGtTolVector2(w,nzz,ZZ,tol)) {
     
    t=maxIndexVector2(w,nzz,ZZ);
    //    printf("1 before t: %d ZZ[t]: %d nzz: %d nzp: %d\n",t,ZZ[t],nzz,nzp);
    t=ZZ[t];
    P[t]=t;
    Z[t]=0;

    findNonNeg2(n,P,&nzp,PP);
    findNonNeg2(n,Z,&nzz,ZZ);
    //    printf("1 after t: %d ZZ[t]: %d nzz: %d nzp: %d\n",t,ZZ[t],nzz,nzp);

    solveEquation(n,z,nzp,PP,perm,XtX,Xty,workMat,workVec);
    //    gsl_vector_fprintf(stdout,z,"%f ");
/* matlab: inner loop: while any((z(PP) <= tol) & iter < itmax; iter = iter + 1 */
    for(; iter<itmax && anyLeTolVector(z,nzp,PP,tol); iter++) {
      alpha=MAXFLOAT; /* a bit dangerous */
      for(i=0; i<n; i++) {
	vv=gsl_vector_get(z,i);
	if((P[i]!=0) && (vv<=tol)) {
	  v=gsl_vector_get(x,i);
	  v=v/(v-vv);
	  if(v<alpha)
	    alpha=v;
	}
      }

      for(i=0; i<n; i++) {
      	v=gsl_vector_get(x,i);
      	vv=gsl_vector_get(z,i);
      	v+=alpha*(vv-v);
      	gsl_vector_set(x,i,v);
      }

      /* gsl_vector_memcpy(temp1,z); */
      /* gsl_vector_sub(temp1,x); */
      /* gsl_vector_scale(temp1,alpha); */
      /* gsl_vector_add(x,temp1); */

      for(i=0; i<n; i++) {
	v=gsl_vector_get(x,i);
	if(abs(v)<tol && P[i]!=0) { // note abs
	  Z[i]=i;
	  P[i]=0;
	}
	//	printf("2 before t: %d ZZ[t]: %d nzz: %d nzp: %d\n",t,ZZ[t],nzz,nzp);
	findNonNeg2(n,P,&nzp,PP);
	findNonNeg2(n,Z,&nzz,ZZ);
	//	printf("2 after t: %d ZZ[t]: %d nzz: %d nzp: %d\n",t,ZZ[t],nzz,nzp);
	solveEquation(n,z,nzp,PP,perm,XtX,Xty,workMat,workVec);
      }
    }

    gsl_vector_memcpy(x,z);

/* matlab: w = Xty - XtX*x */
    //    matrixVectorMultiply(w,XtX,x);
    gsl_blas_dgemv(CblasNoTrans,1.0,XtX,x,0.0,temp1);
    gsl_vector_memcpy(temp2,Xty);
    gsl_vector_sub(temp2,temp1);
    gsl_vector_memcpy(w,temp2);
    // gsl_vector_scale(w,-1); // ?
    //    determineW(XtX,Xty,x,w);
  }
  
  gsl_vector_free(workVec);
  gsl_matrix_free(workMat);
  gsl_vector_free(z);
  gsl_vector_free(temp1);
  gsl_vector_free(temp2);
  gsl_permutation_free(perm);
  free(P);
  free(PP);
  free(Z);
  free(ZZ);
}

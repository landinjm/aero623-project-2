#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/******************************************************************/
//   FUNCTION Definition: shapeL
void
shapeL(double* xref, int p, double** pphi)
{
  /* Returns Lagrange shape functions on a triangle, for a given order
     0<=p<=3 and reference coordinates in xref.  pphi is a pointer to
     the phi vector, which gets reallocated. */
  int prank;
  double x, y, *phi;

  prank = (p + 1) * (p + 2) / 2;

  if ((*pphi) == NULL)
    (*pphi) = (double*)malloc(prank * sizeof(double));
  else
    (*pphi) = (double*)realloc((void*)(*pphi), prank * sizeof(double));

  x = xref[0];
  y = xref[1];
  phi = (*pphi);

  switch (p) {

    case 0:
      phi[0] = 1.0;
      break;

    case 1:
      phi[0] = 1 - x - y;
      phi[1] = x;
      phi[2] = y;
      break;

    case 2:
      phi[0] =
        1.0 - 3.0 * x - 3.0 * y + 2.0 * x * x + 4.0 * x * y + 2.0 * y * y;
      phi[1] = -x + 2.0 * x * x;
      phi[2] = -y + 2.0 * y * y;
      phi[3] = 4.0 * x * y;
      phi[4] = 4.0 * y - 4.0 * x * y - 4.0 * y * y;
      phi[5] = 4.0 * x - 4.0 * x * x - 4.0 * x * y;
      break;

    case 3:
      phi[0] = 1.0 - 11.0 / 2.0 * x - 11.0 / 2.0 * y + 9.0 * x * x +
               18.0 * x * y + 9.0 * y * y - 9.0 / 2.0 * x * x * x -
               27.0 / 2.0 * x * x * y - 27.0 / 2.0 * x * y * y -
               9.0 / 2.0 * y * y * y;
      phi[1] = x - 9.0 / 2.0 * x * x + 9.0 / 2.0 * x * x * x;
      phi[2] = y - 9.0 / 2.0 * y * y + 9.0 / 2.0 * y * y * y;
      phi[3] = -9.0 / 2.0 * x * y + 27.0 / 2.0 * x * x * y;
      phi[4] = -9.0 / 2.0 * x * y + 27.0 / 2.0 * x * y * y;
      phi[5] = -9.0 / 2.0 * y + 9.0 / 2.0 * x * y + 18.0 * y * y -
               27.0 / 2.0 * x * y * y - 27.0 / 2.0 * y * y * y;
      phi[6] = 9.0 * y - 45.0 / 2.0 * x * y - 45.0 / 2.0 * y * y +
               27.0 / 2.0 * x * x * y + 27.0 * x * y * y +
               27.0 / 2.0 * y * y * y;
      phi[7] = 9.0 * x - 45.0 / 2.0 * x * x - 45.0 / 2.0 * x * y +
               27.0 / 2.0 * x * x * x + 27.0 * x * x * y +
               27.0 / 2.0 * x * y * y;
      phi[8] = -9.0 / 2.0 * x + 18.0 * x * x + 9.0 / 2.0 * x * y -
               27.0 / 2.0 * x * x * x - 27.0 / 2.0 * x * x * y;
      phi[9] = 27.0 * x * y - 27.0 * x * x * y - 27.0 * x * y * y;
      break;

    default:
      printf("Unrecognized p in shape.\n");
      break;
  }
}

/******************************************************************/
//   FUNCTION Definition: shape
void
shape(double* xref, int p, double** pphi)
{
  /* wrapper for the Lagrange shape function */
  shapeL(xref, p, pphi);
}

/******************************************************************/
//   FUNCTION Definition: gradientL
int
gradientL(double* xref, int p, double** pgphi)
{
  /* Returns gradients (x and y derivatives) of the Lagrange shape
     functions for a given order p and reference coordinates.  pghi is
     a pointer to the vector that gets reallocated.  In pgphi, all x
     derivatives are stored first, then the y derivatives. */

  int prank;
  double x, y;
  double* gphi;

  prank = (p + 1) * (p + 2) / 2;

  if ((*pgphi) == NULL)
    (*pgphi) = (double*)malloc(2 * prank * sizeof(double));
  else
    (*pgphi) = (double*)realloc((void*)(*pgphi), 2 * prank * sizeof(double));

  gphi = *pgphi;

  x = xref[0];
  y = xref[1];

  switch (p) {

    case 0:
      gphi[0] = 0.0;
      gphi[1] = 0.0;
      break;

    case 1:
      gphi[0] = -1.0;
      gphi[1] = 1.0;
      gphi[2] = 0.0;
      gphi[3] = -1.0;
      gphi[4] = 0.0;
      gphi[5] = 1.0;

      break;

    case 2:
      gphi[0] = -3.0 + 4.0 * x + 4.0 * y;
      gphi[1] = -1.0 + 4.0 * x;
      gphi[2] = 0.0;
      gphi[3] = 4.0 * y;
      gphi[4] = -4.0 * y;
      gphi[5] = 4.0 - 8.0 * x - 4.0 * y;
      gphi[6] = -3.0 + 4.0 * x + 4.0 * y;
      gphi[7] = 0.0;
      gphi[8] = -1.0 + 4.0 * y;
      gphi[9] = 4.0 * x;
      gphi[10] = 4.0 - 4.0 * x - 8.0 * y;
      gphi[11] = -4.0 * x;

      break;

    case 3:
      gphi[0] = -11.0 / 2.0 + 18.0 * x + 18.0 * y - 27.0 / 2.0 * x * x -
                27.0 * x * y - 27.0 / 2.0 * y * y;
      gphi[1] = 1.0 - 9.0 * x + 27.0 / 2.0 * x * x;
      gphi[2] = 0.0;
      gphi[3] = -9.0 / 2.0 * y + 27.0 * x * y;
      gphi[4] = -9.0 / 2.0 * y + 27.0 / 2.0 * y * y;
      gphi[5] = 9.0 / 2.0 * y - 27.0 / 2.0 * y * y;
      gphi[6] = -45.0 / 2.0 * y + 27.0 * x * y + 27.0 * y * y;
      gphi[7] = 9.0 - 45.0 * x - 45.0 / 2.0 * y + 81.0 / 2.0 * x * x +
                54.0 * x * y + 27.0 / 2.0 * y * y;
      gphi[8] = -9.0 / 2.0 + 36.0 * x + 9.0 / 2.0 * y - 81.0 / 2.0 * x * x -
                27.0 * x * y;
      gphi[9] = 27.0 * y - 54.0 * x * y - 27.0 * y * y;
      gphi[10] = -11.0 / 2.0 + 18.0 * x + 18.0 * y - 27.0 / 2.0 * x * x -
                 27.0 * x * y - 27.0 / 2.0 * y * y;
      gphi[11] = 0.0;
      gphi[12] = 1.0 - 9.0 * y + 27.0 / 2.0 * y * y;
      gphi[13] = -9.0 / 2.0 * x + 27.0 / 2.0 * x * x;
      gphi[14] = -9.0 / 2.0 * x + 27.0 * x * y;
      gphi[15] = -9.0 / 2.0 + 9.0 / 2.0 * x + 36.0 * y - 27.0 * x * y -
                 81.0 / 2.0 * y * y;
      gphi[16] = 9.0 - 45.0 / 2.0 * x - 45.0 * y + 27.0 / 2.0 * x * x +
                 54.0 * x * y + 81.0 / 2.0 * y * y;
      gphi[17] = -45.0 / 2.0 * x + 27.0 * x * x + 27.0 * x * y;
      gphi[18] = 9.0 / 2.0 * x - 27.0 / 2.0 * x * x;
      gphi[19] = 27.0 * x - 27.0 * x * x - 54.0 * x * y;

      break;
    default:
      return -1;
      break;
  }

  return 0;
} // gradientL

/******************************************************************/
//   FUNCTION Definition: gradient
int
gradient(double* xref, int p, double** pgphi)
{
  /* Wrapper for Lagrange gradient function */
  return gradientL(xref, p, pgphi);
}

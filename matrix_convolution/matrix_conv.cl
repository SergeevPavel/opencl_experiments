__kernel void matrix_conv(__global float * A, __global float * B, __global float * C, int N, int M)
{
   int i = get_global_id(0);
   int j = get_global_id(1);

   if (i >= N || j >= N)
     return;

   int HM = (M - 1) / 2;

   C[i * N + j] = 0;
   for (int k = -HM; k <= HM; k++) {
     for (int l = -HM; l <= HM; l++) {
       int a_i = i + k;
       int a_j = j + l;

       if (a_i < 0 || a_j < 0 || a_i >= N || a_j >= N) {
         continue;
       }

       int b_i = k + HM;
       int b_j = l + HM;

       C[i * N + j] += A[a_i * N + a_j] * B[b_i * M + b_j];
     }
   }
}

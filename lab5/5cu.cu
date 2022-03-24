
#include <iostream>
#include <stdio.h>
#include <limits>
#include <cfloat>
#include <math.h>
#include <time.h>
#include <fstream>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>



__global__ void histogramcalc(float* arr, int n, int* res, float min, float max, int scnt) {
   int i=blockDim.x*blockIdx.x+threadIdx.x;
   int set=gridDim.x*blockDim.x;
   for (;i<n;i += set) {
        atomicAdd(&(res[(int)((arr[i] - min) / (max - min) * (scnt - 1))]), 1);
    }}

__global__ void histogramsplit(float* arrdev, int n, float* sdev,
   int* firstsplit,
   unsigned int* splitsize,    float min,float max,int scnt){
   int i=blockDim.x*blockIdx.x+threadIdx.x;
   int set=gridDim.x*blockDim.x;
   for (;i < n;i +=set) {
       int sid = ((arrdev[i] - min) / (max - min) * (scnt - 1));
       sdev[firstsplit[sid] + atomicAdd(&(splitsize[sid]), 1)] = arrdev[i];
    }}

__global__ void scan(int *data, int n, int *s, int *res) {
   __shared__ int sharr[2 * 32+((2 * 32)>>5)];

   int set = 1;
       int ai = threadIdx.x;
    int bi = threadIdx.x+(n/2); 
    int set_A=(ai>>5);
     int set_B=(bi>>5);
   sharr[ai+ set_A]= data[ai+ 2 * 32 * blockIdx.x];
   sharr[bi+ set_B]= data[bi+ 2 * 32 * blockIdx.x];
    for (int i= n >> 1;i>0;i >>= 1) {
      __syncthreads();
       if (threadIdx.x<i) {
                   int a=set * (2 * threadIdx.x + 1) - 1 + ((set * (2 * threadIdx.x + 1) - 1) >> 5);
                   int b=set * (2 * threadIdx.x + 2) - 1 + ((set * (2 * threadIdx.x + 2) - 1) >> 5);
                   sharr[b]+=sharr[a];}
                   set <<= 1;}
    if (threadIdx.x==0) {
          int idx=n - 1 + ((n - 1)>>5);
          s[blockIdx.x]=sharr[idx];
          sharr[idx]=0;
    }
    for (int i=1;i < n;i <<= 1) {
       set >>= 1;
       __syncthreads();
       if (threadIdx.x < i) {
           int a=set * (2 * threadIdx.x + 1) - 1 + ((set * (2 * threadIdx.x + 1) - 1)>>5);
           int b=set * (2 * threadIdx.x + 2) - 1 + ((set * (2 * threadIdx.x + 2) - 1)>>5);
           int t=sharr[a];
           sharr[a]=sharr[b];
           sharr[b]+=t;
        }   }
   __syncthreads();
   set = 2 * 32 * blockIdx.x;
   res[ai + set]=sharr[ai + set_A];
   res[bi + set]=sharr[bi + set_B];
}

__global__ void scandistr(int* arr, int* s) {
    arr[threadIdx.x+blockIdx.x*2*32] += s[blockIdx.x];
}
__host__ void scanrec(int* arr, int n, int* res) {
   int block=n/(2 * 32) + 1;
   int* s=NULL;

   cudaMalloc((void**)&s, block * sizeof(int));
   int* s1=NULL;
       cudaMalloc((void**)&s1, block * sizeof(int));
   dim3 threads(32, 1, 1);
   dim3 blocks(block, 1, 1);
   scan <<<blocks, threads >>> (arr, 2 * 32, s, res);
   if (n>=2*32) 
           scanrec(s, block, s1);
   else 
       cudaMemcpy(s1, s, block * sizeof(int), cudaMemcpyDeviceToDevice);

   if (block>1) {
       threads=dim3(2 * 32, 1, 1);    
           blocks = dim3(block - 1, 1, 1);
       scandistr <<<blocks, threads >>> (res + (2 * 32), s1 + 1);
    }}

/*void oddeven(float* arr, int size) {
   for (int i = 0; i < size; i++) {
       for (int j = i & 1; j < size - 1; j += 2) {
           if (arr[j] > arr[j + 1]) {
               float t = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = t;
} } }}
*/
__global__ void oddeven(float* buckets, int n, int* posbucket, int* sizebuck) {
   int bsize=sizebuck[blockIdx.x];
   if (bsize==-1) 
      return;
         __shared__ float bucketsh[2048];

  bucketsh[2 * threadIdx.x] = FLT_MAX; 
   bucketsh[2 * threadIdx.x + 1] = FLT_MAX; 
   __syncthreads();
  if (2 * threadIdx.x <bsize) 
      bucketsh[2 * threadIdx.x] = buckets[2 * threadIdx.x + posbucket[blockIdx.x]];    
        if (2 * threadIdx.x+1 < bsize) 
       bucketsh[2 * threadIdx.x + 1]=buckets[2 * threadIdx.x + 1 + posbucket[blockIdx.x]];
    __syncthreads();
  for (int i=0;i<blockDim.x;i++) {

      if (2*threadIdx.x+ 1 <2047) {
       if (bucketsh[2 * threadIdx.x + 1] > bucketsh[2 * threadIdx.x + 2]) {
               float t = bucketsh[2 * threadIdx.x + 1];
               bucketsh[2 * threadIdx.x + 1] = bucketsh[2 * threadIdx.x + 2];
               bucketsh[2 * threadIdx.x + 2] = t;
            } }
          __syncthreads();
       if (threadIdx.x <2048) {
           if (bucketsh[2 * threadIdx.x ] > bucketsh[2 * threadIdx.x  + 1]) {
               float t = bucketsh[2 * threadIdx.x];
               bucketsh[2 * threadIdx.x] = bucketsh[2 * threadIdx.x + 1];
               bucketsh[2 * threadIdx.x + 1] = t;         }}
        __syncthreads();}
   if (2 * threadIdx.x< bsize) 
      buckets[2 * threadIdx.x+ posbucket[blockIdx.x]]=bucketsh[2 * threadIdx.x];
    if (2 * threadIdx.x+1 <bsize) 
       buckets[2 * threadIdx.x+1 + posbucket[blockIdx.x]]=bucketsh[2 * threadIdx.x+1];
}
__host__ void bucketsort(float* arrdev, int n) {
  thrust::device_ptr<const float> data_device_ptr = thrust::device_pointer_cast(arrdev);
    auto min_max = thrust::minmax_element(thrust::device, data_device_ptr,
                                          data_device_ptr + n);
   
    float min = *min_max.first;   
    float max = *min_max.second;
    // printf("%f %f",min,max);
   
    if (fabs(min - max)<1e-9) 
        return;
    
   int scnt=n/550+1;
   int* lensplit=NULL;
   cudaMalloc((void**)&lensplit, scnt * sizeof(int));
       cudaMemset(lensplit, 0, scnt * sizeof(int));
   histogramcalc <<<512, 512 >>> (arrdev, n, lensplit, min, max, scnt);
   int* firstsplit=NULL;
   cudaMalloc((void**)&firstsplit, scnt * sizeof(int));
   scanrec(lensplit, scnt, firstsplit);
       unsigned int* splitsize=NULL;
   cudaMalloc((void**)&splitsize, scnt * sizeof(unsigned int));
   cudaMemset(splitsize, 0, scnt * sizeof(unsigned int));
   float* sdev=NULL;
   cudaMalloc((void**)&sdev, n * sizeof(float));
       histogramsplit <<<512, 512 >>> (arrdev, n, sdev,
        firstsplit,
        splitsize,
        min, max, scnt);
   int bcnt=scnt;
   int* lenbuck=(int*)malloc(bcnt * sizeof(int));
   memset(lenbuck, 0, bcnt * sizeof(int));
   int* firstbuck=(int*)malloc(bcnt * sizeof(int));
       int bid=0;
   for (int sid=0;sid<scnt;sid++) {
       int firstsplit1=0;
       cudaMemcpy(&firstsplit1, &(firstsplit[sid]), sizeof(int), cudaMemcpyDeviceToHost);
       int lensplit1=0;
               cudaMemcpy(&lensplit1, &(lensplit[sid]), sizeof(int), cudaMemcpyDeviceToHost);
       if (lensplit1>2048) {
           bid++;
           float* split=&(sdev[firstsplit1]); 
           bucketsort(split, lensplit1);
           firstbuck[bid]=firstsplit1; 
           lenbuck[bid]=-1; 
           bid++;        }
        else {
           int buckcur =2048 - lenbuck[bid];
           if (lensplit1<=buckcur) {
               if (buckcur==2048) 
                   firstbuck[bid]=firstsplit1;
               lenbuck[bid]+=lensplit1;         }
           else {
               bid++;
               firstbuck[bid]=firstsplit1;
               lenbuck[bid]=lensplit1;   }}}
   if (lenbuck[bid]==0) 
       bcnt=bid;
   else 
       bcnt=bid + 1;
   /*        for (int i=0;i<bcnt;i++) {
       int sbuck=lenbuck[i];
       if (sbuck==-1) 
           continue;
       float* bucket=(float*)malloc(sbuck * sizeof(float));
       int posbuck=firstbuck[i];
       cudaMemcpy(bucket, &(sdev[posbuck]), sbuck * sizeof(float), cudaMemcpyDeviceToHost);
      oddeven(bucket, sbuck);        cudaMemcpy(&(sdev[posbuck]), bucket, sbuck * sizeof(float), cudaMemcpyHostToDevice);}
      */
      dim3 blocks(bcnt, 1, 1);
    dim3 threads(2048 / 2, 1, 1);    
 int *firstdev = NULL;
 cudaMalloc((void **)&firstdev, bcnt*sizeof(int));
 cudaMemcpy(firstdev, firstbuck, bcnt*sizeof(int),cudaMemcpyHostToDevice);
   int *lendev = NULL;
  cudaMalloc((void **)&lendev, bcnt*sizeof(int));
 cudaMemcpy(lendev, lenbuck, bcnt*sizeof(int),cudaMemcpyHostToDevice);

    oddeven <<<blocks, threads>>> (sdev,n,firstdev,lendev);

   cudaMemcpy(arrdev, sdev, n * sizeof(float), cudaMemcpyDeviceToDevice);
}



int main() {

    int n = 0;
//    float* data = read_data_as_plain_text(&n);
  //  float *data = read_data(&n);

    fread(&n, sizeof(int), 1, stdin);
    if (n == 0) {
        return 0;
    }
    float* data = new float[n];
	
    fread(data, sizeof(float), n, stdin);
    
    float* arrdev = NULL;
    cudaMalloc((void**)&arrdev, n * sizeof(float));
    cudaMemcpy(arrdev, data, n * sizeof(float), cudaMemcpyHostToDevice);


    bucketsort(arrdev, n);

    cudaMemcpy(data, arrdev, n * sizeof(float), cudaMemcpyDeviceToHost);
   
	fwrite(data, sizeof(float), n, stdout);

   // print_array(data, n);

   /* if (sorted(data, n)) {
        printf("--\nStatus: OK\n");
    }
    else {
        printf("--\nStatus: WA\n");
    }
*/
    return 0;
}


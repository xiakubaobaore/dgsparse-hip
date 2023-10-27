#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <fstream>
#include <string>
#include "../include/hip/spmm_hip.hpp"
#include "../include/gspmm.h"
#include <vector>

#define checkhiperror(condition)\
do{\
    if(hipSuccess != (condition)) {\
        fprintf(stderr, "Hip runTime error in line %d of file %s, \n",\
                __LINE__, __FILE__, hipGetErrorString(hipGetLastError()));\
        exit(EXIT_FAILURE);\
    }\
}while(0)

// A * B =C
// A: sparse, m*k
// B: dense, k*n
// C: dense, m*n

void init_random(float *A, int size){
    for(int i=0;i<size;i++){
        // A[i] = rand()/(float)RAND_MAX;
        A[i] = 1;
    }
}
 
 int main(int argc,char **argv)
{
 
    int size_m = 1000;
    int size_k = 1000;
    int size_n = 500;
    size_t size_A = size_m * size_k * sizeof(float);
    size_t size_B = size_n * size_k * sizeof(float);
    size_t size_C = size_m * size_n * sizeof(float);
    printf("[Spmm of %d*%d*%d elements]\n", size_m, size_k, size_n);
 
//1.申请Host内存并初始化
    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C = (float *)malloc(size_C);
    int *E = (int *)calloc(size_m * size_k, sizeof(int));

    float *h_ref = (float *)malloc(size_C);

    if (h_A == NULL || h_B == NULL || h_C == NULL 
        || E == NULL || h_ref == NULL){
        printf ("malloc failed\n");
        return -1;
    }
    init_random(h_A, size_m*size_k);
    init_random(h_B, size_k*size_n);
    init_random(h_C, size_m*size_n);

    std::vector<int> csrptr, csrind;
    std::vector<float> csrval;
    csrptr.push_back(0);
    for(int row=0;row < size_m;++row){
        for(int col=0;col < size_k;++col){
            int id = row*size_m+size_k;
            if(h_A[id] != 0){
                csrind.push_back(col);
                csrval.push_back(h_A[id]);
            }
        }
        csrptr.push_back(csrind.size());
    }
    size_t size_ptr = csrptr.size() * sizeof(int);
    size_t size_ind = csrind.size() * sizeof(int);
    size_t size_val = csrval.size() * sizeof(float);
    int *rowptr = (int *)malloc(size_ptr);
    int *rowind = (int *)malloc(size_ind);
    float *value = (float *)malloc(size_val);
    if (rowptr == NULL || rowind == NULL || value == NULL){
        printf("malloc failed\n");
        return -1;
    }
    memcpy(rowptr, &csrptr[0], size_ptr);
    memcpy(rowind, &csrind[0], size_ind);
    memcpy(value, &csrval[0], size_val);
    for(int i=0;i<csrptr.size();i++){
        printf("rowptr[%d] = %f", i, )
    }

//1.申请Device内存
    float *d_B = NULL;
    checkhiperror(hipMalloc((void **)&d_B, size_B));
    float *d_C = NULL;
    checkhiperror(hipMalloc((void **)&d_C, size_C));
    int *d_E = NULL;
    checkhiperror(hipMalloc((void **)&d_E, size_m * size_k * sizeof(int)));    
    int *d_ptr = NULL;
    checkhiperror(hipMalloc((void **)&d_ptr, size_ptr));
    int *d_ind = NULL;
    checkhiperror(hipMalloc((void **)&d_ind, size_ind));
    float *d_val = NULL;
    checkhiperror(hipMalloc((void **)&d_val, size_val));    
 
 //2.将两个向量从Host端提交到Device端
     checkhiperror(hipMemcpy(d_B,h_B,size_B,hipMemcpyHostToDevice));
     checkhiperror(hipMemcpy(d_E,E,size_m * size_k * sizeof(int),hipMemcpyHostToDevice));
     checkhiperror(hipMemcpy(d_ptr,rowptr,size_ptr,hipMemcpyHostToDevice));
     checkhiperror(hipMemcpy(d_ind,rowind,size_ind,hipMemcpyHostToDevice));
     checkhiperror(hipMemcpy(d_val,value,size_val,hipMemcpyHostToDevice));

 
//3.调用hip核函数 
     int Mdim_worker = csrptr.size() - 1;
     int Ndim_worker = size_n;

     int RefThreadPerBlock = 256;
     int Ndim_threadblock = CEIL(Ndim_worker, RefThreadPerBlock);
     int Ndim_thread_per_tb = min(Ndim_worker, RefThreadPerBlock);
     int Mdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
     int Mdim_threadblock = CEIL(Mdim_worker, Mdim_thread_per_tb);

     dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
     dim3 blockDim(Ndim_thread_per_tb, Mdim_thread_per_tb, 1);

     printf("launch kernel...\n");
     
    //  SWITCH_REDUCEOP(REDUCEOP::SUM, REDUCE, {
    //     SWITCH_COMPUTEOP(COMPUTEOP::ADD, COMPUTE, {
    //         hipLaunchKernelGGL(
    //             csrspmm_seqreduce_rowbalance_kernel,
    //             gridDim, blockDim, 0, 0, 
    //             Mdim_worker, Ndim_worker, &d_ptr, &d_ind, &d_val,
    //             &d_B, &d_C, &d_E);
    //     });
    //  });

    hipLaunchKernelGGL(
        csrspmm_seqreduce_rowbalance_kernel_without_template,
        gridDim, blockDim, 0, 0, 
        Mdim_worker, Ndim_worker, d_ptr, d_ind, d_val,
        d_B, d_C, d_E);


     printf("HIP kernel launch with %d blocks of %d threads\n",
        Mdim_thread_per_tb*Ndim_thread_per_tb, Mdim_threadblock*Ndim_threadblock);
  
//4.将两个向量相乘的结果从Device端传回Host端
    checkhiperror(hipMemcpy(h_C,d_C,size_C,hipMemcpyDeviceToHost));
    checkhiperror(hipDeviceSynchronize());
    //对比CPU和GPU计算结果误差
    for(int row = 0; row<size_m;row++){
        for(int col = 0; col<size_n;col++){
            int id = row*size_m+col;
            h_ref[id]=0;
            for(int k = 0; k<size_k;k++){
                h_ref[id] += h_A[row*size_m+k]*h_B[col*size_k+k];
            }
        }
    }

   for (int i = 0; i < size_C; ++i)
    {
        if (fabs(h_ref[i] - h_C[i]) > 1e-8)
        {
            fprintf(stderr, "Result verification failed at element %d!\nh_ref[%d] = %f, h_C[%d] = %f\n", i, i, h_ref[i], i, h_C[i]);
            exit(EXIT_FAILURE);
        }
    }
 
//5.释放内存
      hipFree(d_B);
      hipFree(d_C);
      hipFree(d_E);
      hipFree(d_ptr);
      hipFree(d_ind);
      hipFree(d_val);
      free(h_A);
      free(h_B);
      free(h_C);
      free(E);
      free(h_ref);
      free(rowptr);
      free(rowind);
      free(value);
 
     return 0;
 }
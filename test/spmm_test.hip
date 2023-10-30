#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <fstream>
#include <string>
#include "../include/hip/spmm_hip.hpp"
#include "../include/gspmm.h"
#include <vector>
#include <time.h>

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
void init_random_sparse(float *A, int size, float threshold){
    for(int i=0;i<size;i++){
        if(rand()/(float)RAND_MAX < threshold){
            A[i] = 1;
        }
        else{
            A[i] = 0;
        }
    }
}

void init_random(float *A, int size){
    for(int i=0;i<size;i++){
        // A[i] = rand()/(float)RAND_MAX;
        A[i] = 1;
    }
}

 int main(int argc,char **argv)
{
    srand((int)time(0));
    int size_m = 10000;
    int size_k = 10000;
    int size_n = 128;
    // int size_m = 100;
    // int size_k = 100;
    // int size_n = 50;
    size_t size_A = size_m * size_k * sizeof(float);
    size_t size_B = size_n * size_k * sizeof(float);
    size_t size_C = size_m * size_n * sizeof(float);
    printf("[Spmm of %d*%d*%d elements]\n", size_m, size_k, size_n);


//1.申请Host内存并初始化
    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C = (float *)malloc(size_C);
    int *E = (int *)calloc(size_m * size_n, sizeof(int));

    float *h_ref = (float *)malloc(size_C);

    if (h_A == NULL || h_B == NULL || h_C == NULL
        || E == NULL || h_ref == NULL){
        printf ("malloc failed\n");
        return -1;
    }
    init_random_sparse(h_A, size_m*size_k, 0.01);
    init_random(h_B, size_k*size_n);
    // for(int i=0;i<size_m*size_k;i++){
    //     printf("h_A[%d] = %f\n", i, h_A[i]);
    // }

    std::vector<int> csrptr, csrind;
    std::vector<float> csrval;
    csrptr.push_back(0);
    for(int row=0;row < size_m;++row){
        for(int col=0;col < size_k;++col){
            int id = row*size_k+col;
            if(h_A[id] != 0){
                csrind.push_back(col);
                csrval.push_back(h_A[id]);
            }
        }
        csrptr.push_back(csrind.size());
    }

    // 打印csr格式

    // for(int i=0;i<csrind.size();i++){
    //     printf("csrptr[%d] = %d, csrind[%d] = %d, csrval[%d] = %f\n"
    //     , i, csrptr[i], i, csrind[i], i, csrval[i]);
    // }
    // for(int row=0; row < size_m; row++){
    //     for(int col=0; col < size_k; col++){
    //         if(h_A[row * size_k + col] != 0){
    //             printf("row = %d, col = %d\n", row, col);
    //         }
    //     }
    // }

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

//1.申请Device内存
    float *d_B = NULL;
    checkhiperror(hipMalloc((void **)&d_B, size_B));
    float *d_C = NULL;
    checkhiperror(hipMalloc((void **)&d_C, size_C));
    int *d_E = NULL;
    checkhiperror(hipMalloc((void **)&d_E, size_m * size_n * sizeof(int)));
    int *d_ptr = NULL;
    checkhiperror(hipMalloc((void **)&d_ptr, size_ptr));
    int *d_ind = NULL;
    checkhiperror(hipMalloc((void **)&d_ind, size_ind));
    float *d_val = NULL;
    checkhiperror(hipMalloc((void **)&d_val, size_val));

 //2.将两个向量从Host端提交到Device端
     checkhiperror(hipMemcpy(d_B,h_B,size_B,hipMemcpyHostToDevice));
     checkhiperror(hipMemcpy(d_E,E,size_m * size_n * sizeof(int),hipMemcpyHostToDevice));
     checkhiperror(hipMemcpy(d_ptr,rowptr,size_ptr,hipMemcpyHostToDevice));
     checkhiperror(hipMemcpy(d_ind,rowind,size_ind,hipMemcpyHostToDevice));
     checkhiperror(hipMemcpy(d_val,value,size_val,hipMemcpyHostToDevice));


//3.调用hip核函数
     int Mdim_worker = size_m;
     int Ndim_worker = size_n;

     int RefThreadPerBlock = 512;
     int Ndim_threadblock = CEIL(Ndim_worker, RefThreadPerBlock);
     int Ndim_thread_per_tb = min(Ndim_worker, RefThreadPerBlock);
     int Mdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
     int Mdim_threadblock = CEIL(Mdim_worker, Mdim_thread_per_tb);

     dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
     dim3 blockDim(Ndim_thread_per_tb, Mdim_thread_per_tb, 1);

     printf("launch kernel...\n");

    SWITCH_REDUCEOP(REDUCEOP::SUM, REDUCE, {
        SWITCH_COMPUTEOP(COMPUTEOP::ADD, COMPUTE, {
            csrspmm_seqreduce_rowbalance_kernel<int, float, REDUCE, COMPUTE>
                <<<gridDim, blockDim>>>(
                    Mdim_worker, Ndim_worker, d_ptr, d_ind, d_val,
                    d_B, d_C, d_E);
        });
    });

    // hipLaunchKernelGGL(
    //     csrspmm_seqreduce_rowbalance_kernel_without_template,
    //     gridDim, blockDim, 0, 0,
    //     Mdim_worker, Ndim_worker, d_ptr, d_ind, d_val,
    //     d_B, d_C, d_E);

     printf("HIP kernel launch with %d*%d blocks of %d*%d threads\n",
        Mdim_threadblock, Ndim_threadblock, Ndim_thread_per_tb, Mdim_thread_per_tb);


//4.将两个向量相乘的结果从Device端传回Host端
    checkhiperror(hipMemcpy(h_C,d_C,size_C,hipMemcpyDeviceToHost));
    checkhiperror(hipDeviceSynchronize());
    //对比CPU和GPU计算结果误差
    for(int row = 0; row<size_m;row++){
        for(int col = 0; col<size_n;col++){
            int id = row*size_n+col;
            h_ref[id]=0;
            for(int k = 0; k<size_k;k++){
                h_ref[id] += h_A[row * size_k + k]*h_B[col * size_k + k];
            }
        }
    }

    //打印cpu和gpu上的计算结果

    // for(int row=0;row<size_m;row++){
    //     for(int col=0;col<size_n;col++){
    //         printf("h_C[%d][%d] = %f ", row, col, h_C[row*size_n+col]);
    //     }
    //     printf("\n");
    // }
    // for(int row=0;row<size_m;row++){
    //     for(int col=0;col<size_n;col++){
    //         printf("h_ref[%d][%d] = %f ", row, col, h_ref[row*size_n+col]);
    //     }
    //     printf("\n");
    // }

   for (int i = 0; i < size_m * size_n; ++i)
    {
        if (fabs(h_ref[i] - h_C[i]) > 1e-8)
        {
            printf("Result verification failed at element %d!\nh_ref[%d] = %f, h_C[%d] = %f\n", i, i, h_ref[i], i, h_C[i]);
            break;
        }
    }

    printf("###########test pass!#############\n");

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

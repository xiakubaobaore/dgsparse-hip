#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipsparse/hipsparse.h>
#include <rocsparse/rocsparse.h>
#include <vector>
#include <time.h>

#include "../include/hip/spmm_hip.hpp"
#include "../include/gspmm.h"

#define checkhiperror(condition)\
do{\
    if(hipSuccess != (condition)) {\
        fprintf(stderr, "Hip runTime error in line %d of file %s, \n",\
                __LINE__, __FILE__, hipGetErrorString(hipGetLastError()));\
        exit(EXIT_FAILURE);\
    }\
}while(0)

enum method{
    rocsparse = 0,
    dgsparse_0,
    dgsparse_1,
    dgsparse_2
};

__global__ void warm_up() {}

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
        A[i] = rand()/(float)RAND_MAX;
    }
}

void transpose(float *from, float *to, int a, int b){
    for(int i = 0; i < a; ++i){
        for(int j = 0; j < b; ++j){
            to[j * a + i] = from[i * b + j];
        }
    }
}

bool check_result(float *C_ref, float *C, int num){
    bool flag = true;
    for (int i = 0; i < num; ++i)
    {
        if (C_ref[i] - C[i] > 1e-4)
        {
            printf("Result verification failed at element %d!\nC_ref[%d] = %f, C[%d] = %f\n", i, i, C_ref[i], i, C[i]);
            flag = false;
        }
    }
    return flag;
}

float spmm_time(int size_m, int size_k, int size_n, int circle_num, float sparsity, method algorithm)
{
    bool is_transpose = (algorithm == method::rocsparse) ? true : false;
    size_t size_A = size_m * size_k * sizeof(float);
    size_t size_B = size_n * size_k * sizeof(float);
    size_t size_C = size_m * size_n * sizeof(float);
    printf("[Spmm of %d*%d*%d elements]\n", size_m, size_k, size_n);

    //申请Host内存并初始化
    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_B_trans = (float *)malloc(size_B);
    int *E = (int *)calloc(size_m * size_n, sizeof(int));

    if (h_A == NULL || h_B == NULL || E == NULL || h_B_trans == NULL){
        printf ("malloc failed\n");
        return false;
    }
    init_random_sparse(h_A, size_m*size_k, sparsity);
    init_random(h_B, size_k*size_n);
    transpose(h_B, h_B_trans, size_k, size_n);

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

    size_t size_ptr = csrptr.size() * sizeof(int);
    size_t size_ind = csrind.size() * sizeof(int);
    size_t size_val = csrval.size() * sizeof(float);
    int *rowptr = (int *)malloc(size_ptr);
    int *rowind = (int *)malloc(size_ind);
    float *value = (float *)malloc(size_val);
    if (rowptr == NULL || rowind == NULL || value == NULL){
        printf("malloc failed\n");
        return false;
    }
    memcpy(rowptr, &csrptr[0], size_ptr);
    memcpy(rowind, &csrind[0], size_ind);
    memcpy(value, &csrval[0], size_val);

    //申请Device内存
    float *d_B = NULL;
    checkhiperror(hipMalloc((void **)&d_B, size_B));
    float *d_B_trans = NULL;
    checkhiperror(hipMalloc((void **)&d_B_trans, size_B));
    float *d_C = NULL;
    checkhiperror(hipMalloc((void **)&d_C, size_C));
    hipMemset((void *)d_C, 0, size_C);
    int *d_E = NULL;
    checkhiperror(hipMalloc((void **)&d_E, size_m * size_n * sizeof(int)));
    int *d_ptr = NULL;
    checkhiperror(hipMalloc((void **)&d_ptr, size_ptr));
    int *d_ind = NULL;
    checkhiperror(hipMalloc((void **)&d_ind, size_ind));
    float *d_val = NULL;
    checkhiperror(hipMalloc((void **)&d_val, size_val));

    //从Host端提交到Device端
    checkhiperror(hipMemcpy(d_B,h_B,size_B,hipMemcpyHostToDevice));
    checkhiperror(hipMemcpy(d_B_trans,h_B_trans,size_B,hipMemcpyHostToDevice));
    checkhiperror(hipMemcpy(d_E,E,size_m * size_n * sizeof(int),hipMemcpyHostToDevice));
    checkhiperror(hipMemcpy(d_ptr,rowptr,size_ptr,hipMemcpyHostToDevice));
    checkhiperror(hipMemcpy(d_ind,rowind,size_ind,hipMemcpyHostToDevice));
    checkhiperror(hipMemcpy(d_val,value,size_val,hipMemcpyHostToDevice));

    clock_t start, end;
    float time_elapsed;
    for (int i = 0; i < 1000; i++)
        warm_up<<<1, 1>>>();
    //调用kernel

    if (algorithm == method::rocsparse){
        float alpha = 1;
        float beta = 0;
        rocsparse_handle rhandle;
        rocsparse_create_handle(&rhandle);
        rocsparse_mat_descr rdescr;
        rocsparse_create_mat_descr(&rdescr);
        rocsparse_set_mat_type(rdescr, rocsparse_matrix_type_general);
        rocsparse_set_mat_index_base(rdescr, rocsparse_index_base_zero);

        start = clock();
        for(int circle = 0; circle < circle_num; ++circle){
            rocsparse_scsrmm(rhandle, rocsparse_operation_none, rocsparse_operation_none,
                            size_m, size_n, size_k, csrval.size(), &alpha, rdescr,
                            d_val, d_ptr, d_ind, d_B_trans, size_k, &beta, d_C, size_m);
        }
        end = clock();
        time_elapsed = (end - start) * 1.0/circle_num;
        printf("rocsparse kernel time = %fms\n", time_elapsed);

        rocsparse_destroy_handle(rhandle);
        rocsparse_destroy_mat_descr(rdescr);

        // float alpha = 1;
        // float beta = 0;
        // hipsparseHandle_t handle;
        // hipsparseCreate(&handle);
        // hipsparseMatDescr_t descr;
        // hipsparseCreateMatDescr(&descr);
        // hipsparseSetMatType(descr, HIPSPARSE_MATRIX_TYPE_GENERAL);
        // hipsparseSetMatIndexBase(descr, HIPSPARSE_INDEX_BASE_ZERO);
        // start = clock();
        // for(int circle = 0; circle < circle_num; ++circle){
        //     hipsparseScsrmm(
        //         handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
        //         size_m, size_n, size_k, csrval.size(), &alpha, descr,
        //         d_val, d_ptr, d_ind,
        //         d_B_trans, size_k, &beta,
        //         d_C, size_m
        //     );
        // }
        // end = clock();
        // time_elapsed = (end - start) * 1.0/circle_num;
        // printf("hipsparse kernel time = %fms\n", time_elapsed);
        // hipsparseDestroy(handle);
        // hipsparseDestroyMatDescr(descr);
    }
    else if(algorithm == method::dgsparse_0){
        int Mdim_worker = size_m;
        int Ndim_worker = size_n;

        int RefThreadPerBlock = 256;
        int Ndim_threadblock = CEIL(Ndim_worker, RefThreadPerBlock);
        int Ndim_thread_per_tb = min(Ndim_worker, RefThreadPerBlock);
        int Mdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
        int Mdim_threadblock = CEIL(Mdim_worker, Mdim_thread_per_tb);

        dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
        dim3 blockDim(Ndim_thread_per_tb, Mdim_thread_per_tb, 1);

        start = clock();
        for(int circle = 0; circle < circle_num; ++circle){
            SWITCH_REDUCEOP(REDUCEOP::SUM, REDUCE, {
                SWITCH_COMPUTEOP(COMPUTEOP::ADD, COMPUTE, {
                    csrspmm_seqreduce_rowbalance_kernel<int, float, REDUCE, COMPUTE>
                        <<<gridDim, blockDim>>>(
                            Mdim_worker, Ndim_worker, d_ptr, d_ind, d_val,
                            d_B, d_C, d_E);
                });
            });
        }
        end = clock();
        time_elapsed = (end - start) * 1.0/circle_num;
        printf("dgsparse_0 kernel time = %fms\n", time_elapsed);
    }
    else if(algorithm == method::dgsparse_1){
        int Mdim_worker = size_m;
        int Ndim_worker = size_n;

        int RefThreadPerBlock = 256;
        int Ndim_threadblock = CEIL(Ndim_worker, RefThreadPerBlock);
        int Ndim_thread_per_tb = min(Ndim_worker, RefThreadPerBlock);
        int Mdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
        int Mdim_threadblock = CEIL(Mdim_worker, Mdim_thread_per_tb);

        dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
        dim3 blockDim(Ndim_thread_per_tb, Mdim_thread_per_tb, 1);

        start = clock();
        for(int circle = 0; circle < circle_num; ++circle){
            SWITCH_REDUCEOP(REDUCEOP::SUM, REDUCE, {
                SWITCH_COMPUTEOP(COMPUTEOP::ADD, COMPUTE, {
                    csrspmm_seqreduce_nnzbalance_kernel<int, float, REDUCE, COMPUTE>
                        <<<gridDim, blockDim>>>(
                            Mdim_worker, Ndim_worker, csrval.size(), d_ptr, d_ind, d_val,
                            d_B, d_C, d_E);
                });
            });
        }
        end = clock();
        time_elapsed = (end - start) * 1.0/circle_num;
        printf("dgsparse_1 kernel time = %fms\n", time_elapsed);
    }
    else if(algorithm == method::dgsparse_2){
        int Mdim_worker = size_m;
        int Ndim_worker = size_n;
        int coarsen_factor = (Ndim_worker % 4 == 0) ? 4 : (Ndim_worker % 2 == 0) ? 2 : 1;
        // partition large-N and map to blockdim.y to help cache performance
        int RefThreadPerBlock = 256;

        int Ndim_threadblock = CEIL(Ndim_worker, 32);
        int Ndim_warp_per_tb = min(Ndim_worker, 32) / coarsen_factor;
        int ref_warp_per_tb = RefThreadPerBlock / 32;
        int Mdim_warp_per_tb = CEIL(ref_warp_per_tb, Ndim_warp_per_tb);

        int gridDimX = CEIL(Mdim_worker, Mdim_warp_per_tb);
        int gridDimY = Ndim_threadblock;

        dim3 gridDim(gridDimX, gridDimY, 1);
        dim3 blockDim(Ndim_warp_per_tb * 32, Mdim_warp_per_tb, 1);

        if (coarsen_factor == 4){
            start = clock();
            for(int circle = 0; circle < circle_num; ++circle){
                SWITCH_REDUCEOP(REDUCEOP::SUM, REDUCE, {
                    SWITCH_COMPUTEOP(COMPUTEOP::ADD, COMPUTE, {
                        csrspmm_parreduce_rowbalance_kernel<int, float, float4, REDUCE,
                                                        COMPUTE><<<gridDim, blockDim>>>(
                                Mdim_worker, Ndim_worker, d_ptr, d_ind, d_val,
                                d_B, d_C, d_E);
                    });
                });
            }
            end = clock();
            time_elapsed = (end - start) * 1.0/circle_num;
            printf("dgsparse_2 kernel time = %fms\n", time_elapsed);
        }
        if (coarsen_factor == 2){
            start = clock();
            for(int circle = 0; circle < circle_num; ++circle){
                SWITCH_REDUCEOP(REDUCEOP::SUM, REDUCE, {
                    SWITCH_COMPUTEOP(COMPUTEOP::ADD, COMPUTE, {
                        csrspmm_parreduce_rowbalance_kernel<int, float, float2, REDUCE,
                                                        COMPUTE><<<gridDim, blockDim>>>(
                                Mdim_worker, Ndim_worker, d_ptr, d_ind, d_val,
                                d_B, d_C, d_E);
                    });
                });
            }
            end = clock();
            time_elapsed = (end - start) * 1.0/circle_num;
            printf("dgsparse_2 kernel time = %fms\n", time_elapsed);
        }
        else {
            start = clock();
            for(int circle = 0; circle < circle_num; ++circle){
                SWITCH_REDUCEOP(REDUCEOP::SUM, REDUCE, {
                    SWITCH_COMPUTEOP(COMPUTEOP::ADD, COMPUTE, {
                        csrspmm_parreduce_rowbalance_kernel<int, float, float, REDUCE,
                                                        COMPUTE><<<gridDim, blockDim>>>(
                                Mdim_worker, Ndim_worker, d_ptr, d_ind, d_val,
                                d_B, d_C, d_E);
                    });
                });
            }
            end = clock();
            time_elapsed = (end - start) * 1.0/circle_num;
            printf("dgsparse_2 kernel time = %fms\n", time_elapsed);
        }
    }
    checkhiperror(hipDeviceSynchronize());

    // 释放内存
    hipFree(d_B);
    hipFree(d_B_trans);
    hipFree(d_C);
    hipFree(d_E);
    hipFree(d_ptr);
    hipFree(d_ind);
    hipFree(d_val);
    free(h_A);
    free(h_B);
    free(h_B_trans);
    free(E);
    free(rowptr);
    free(rowind);
    free(value);

    csrptr.clear();csrptr.shrink_to_fit();
    csrind.clear();csrind.shrink_to_fit();
    csrval.clear();csrval.shrink_to_fit();
    return time_elapsed;
}

bool spmm_check(int size_m, int size_k, int size_n, method algorithm)
{
    bool is_transpose = (algorithm == method::rocsparse) ? true : false;
    size_t size_A = size_m * size_k * sizeof(float);
    size_t size_B = size_n * size_k * sizeof(float);
    size_t size_C = size_m * size_n * sizeof(float);
    printf("[Spmm of %d*%d*%d elements]\n", size_m, size_k, size_n);

    //申请Host内存并初始化
    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_B_trans = (float *)malloc(size_B);
    float *h_C = (float *)malloc(size_C);
    int *E = (int *)calloc(size_m * size_n, sizeof(int));

    float *h_ref = (float *)malloc(size_C); //参考结果

    if (h_A == NULL || h_B == NULL || h_C == NULL
        || E == NULL || h_ref == NULL || h_B_trans == NULL){
        printf ("malloc failed\n");
        return false;
    }
    init_random_sparse(h_A, size_m*size_k, 0.01);
    init_random(h_B, size_k*size_n);
    transpose(h_B, h_B_trans, size_k, size_n);

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

    size_t size_ptr = csrptr.size() * sizeof(int);
    size_t size_ind = csrind.size() * sizeof(int);
    size_t size_val = csrval.size() * sizeof(float);
    int *rowptr = (int *)malloc(size_ptr);
    int *rowind = (int *)malloc(size_ind);
    float *value = (float *)malloc(size_val);
    if (rowptr == NULL || rowind == NULL || value == NULL){
        printf("malloc failed\n");
        return false;
    }
    memcpy(rowptr, &csrptr[0], size_ptr);
    memcpy(rowind, &csrind[0], size_ind);
    memcpy(value, &csrval[0], size_val);

    //申请Device内存
    float *d_B = NULL;
    checkhiperror(hipMalloc((void **)&d_B, size_B));
    float *d_B_trans = NULL;
    checkhiperror(hipMalloc((void **)&d_B_trans, size_B));
    float *d_C = NULL;
    checkhiperror(hipMalloc((void **)&d_C, size_C));
    hipMemset((void *)d_C, 0, size_C);
    int *d_E = NULL;
    checkhiperror(hipMalloc((void **)&d_E, size_m * size_n * sizeof(int)));
    int *d_ptr = NULL;
    checkhiperror(hipMalloc((void **)&d_ptr, size_ptr));
    int *d_ind = NULL;
    checkhiperror(hipMalloc((void **)&d_ind, size_ind));
    float *d_val = NULL;
    checkhiperror(hipMalloc((void **)&d_val, size_val));

    //从Host端提交到Device端
    checkhiperror(hipMemcpy(d_B,h_B,size_B,hipMemcpyHostToDevice));
    checkhiperror(hipMemcpy(d_B_trans,h_B_trans,size_B,hipMemcpyHostToDevice));
    checkhiperror(hipMemcpy(d_E,E,size_m * size_n * sizeof(int),hipMemcpyHostToDevice));
    checkhiperror(hipMemcpy(d_ptr,rowptr,size_ptr,hipMemcpyHostToDevice));
    checkhiperror(hipMemcpy(d_ind,rowind,size_ind,hipMemcpyHostToDevice));
    checkhiperror(hipMemcpy(d_val,value,size_val,hipMemcpyHostToDevice));

    //调用kernel
    if (algorithm == method::rocsparse){
        float alpha = 1;
        float beta = 0;
        rocsparse_handle rhandle;
        rocsparse_create_handle(&rhandle);
        rocsparse_mat_descr rdescr;
        rocsparse_create_mat_descr(&rdescr);
        rocsparse_set_mat_type(rdescr, rocsparse_matrix_type_general);
        rocsparse_set_mat_index_base(rdescr, rocsparse_index_base_zero);

        rocsparse_scsrmm(rhandle, rocsparse_operation_none, rocsparse_operation_none,
                        size_m, size_n, size_k, csrval.size(), &alpha, rdescr,
                        d_val, d_ptr, d_ind, d_B_trans, size_k, &beta, d_C, size_m);

        rocsparse_destroy_handle(rhandle);
        rocsparse_destroy_mat_descr(rdescr);

        // float alpha = 1;
        // float beta = 0;
        // hipsparseHandle_t handle;
        // hipsparseCreate(&handle);
        // hipsparseMatDescr_t descr;
        // hipsparseCreateMatDescr(&descr);
        // hipsparseSetMatType(descr, HIPSPARSE_MATRIX_TYPE_GENERAL);
        // hipsparseSetMatIndexBase(descr, HIPSPARSE_INDEX_BASE_ZERO);

        // hipsparseScsrmm(
        //     handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
        //     size_m, size_n, size_k, csrval.size(), &alpha, descr,
        //     d_val, d_ptr, d_ind,
        //     d_B_trans, size_k, &beta,
        //     d_C, size_m
        // );

        // hipsparseDestroy(handle);
        // hipsparseDestroyMatDescr(descr);
    }else{
        if(algorithm == method::dgsparse_0){
            int Mdim_worker = size_m;
            int Ndim_worker = size_n;

            int RefThreadPerBlock = 256;
            int Ndim_threadblock = CEIL(Ndim_worker, RefThreadPerBlock);
            int Ndim_thread_per_tb = min(Ndim_worker, RefThreadPerBlock);
            int Mdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
            int Mdim_threadblock = CEIL(Mdim_worker, Mdim_thread_per_tb);

            dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
            dim3 blockDim(Ndim_thread_per_tb, Mdim_thread_per_tb, 1);

            SWITCH_REDUCEOP(REDUCEOP::SUM, REDUCE, {
                SWITCH_COMPUTEOP(COMPUTEOP::ADD, COMPUTE, {
                    csrspmm_seqreduce_rowbalance_kernel<int, float, REDUCE, COMPUTE>
                        <<<gridDim, blockDim>>>(
                            Mdim_worker, Ndim_worker, d_ptr, d_ind, d_val,
                            d_B, d_C, d_E);
                });
            });
        }
        else if(algorithm == method::dgsparse_1){
            int Mdim_worker = size_m;
            int Ndim_worker = size_n;

            int RefThreadPerBlock = 256;
            int Ndim_threadblock = CEIL(Ndim_worker, RefThreadPerBlock);
            int Ndim_thread_per_tb = min(Ndim_worker, RefThreadPerBlock);
            int Mdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
            int Mdim_threadblock = CEIL(Mdim_worker, Mdim_thread_per_tb);

            dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
            dim3 blockDim(Ndim_thread_per_tb, Mdim_thread_per_tb, 1);

            SWITCH_REDUCEOP(REDUCEOP::SUM, REDUCE, {
                SWITCH_COMPUTEOP(COMPUTEOP::ADD, COMPUTE, {
                    csrspmm_seqreduce_nnzbalance_kernel<int, float, REDUCE, COMPUTE>
                        <<<gridDim, blockDim>>>(
                            Mdim_worker, Ndim_worker, csrval.size(), d_ptr, d_ind, d_val,
                            d_B, d_C, d_E);
                });
            });
        }
        else if(algorithm == method::dgsparse_2){
            int Mdim_worker = size_m;
            int Ndim_worker = size_n;
            int coarsen_factor = (Ndim_worker % 4 == 0) ? 4 : (Ndim_worker % 2 == 0) ? 2 : 1;
            // partition large-N and map to blockdim.y to help cache performance
            int RefThreadPerBlock = 256;

            int Ndim_threadblock = CEIL(Ndim_worker, 32);
            int Ndim_warp_per_tb = min(Ndim_worker, 32) / coarsen_factor;
            int ref_warp_per_tb = RefThreadPerBlock / 32;
            int Mdim_warp_per_tb = CEIL(ref_warp_per_tb, Ndim_warp_per_tb);

            int gridDimX = CEIL(Mdim_worker, Mdim_warp_per_tb);
            int gridDimY = Ndim_threadblock;

            dim3 gridDim(gridDimX, gridDimY, 1);
            dim3 blockDim(Ndim_warp_per_tb * 32, Mdim_warp_per_tb, 1);

            if (coarsen_factor == 4){
                SWITCH_REDUCEOP(REDUCEOP::SUM, REDUCE, {
                    SWITCH_COMPUTEOP(COMPUTEOP::ADD, COMPUTE, {
                        csrspmm_parreduce_rowbalance_kernel<int, float, float4, REDUCE,
                                                        COMPUTE><<<gridDim, blockDim>>>(
                                Mdim_worker, Ndim_worker, d_ptr, d_ind, d_val,
                                d_B, d_C, d_E);
                    });
                });
            }
            if (coarsen_factor == 2){
                SWITCH_REDUCEOP(REDUCEOP::SUM, REDUCE, {
                    SWITCH_COMPUTEOP(COMPUTEOP::ADD, COMPUTE, {
                        csrspmm_parreduce_rowbalance_kernel<int, float, float2, REDUCE,
                                                        COMPUTE><<<gridDim, blockDim>>>(
                                Mdim_worker, Ndim_worker, d_ptr, d_ind, d_val,
                                d_B, d_C, d_E);
                    });
                });
            }
            else {
                SWITCH_REDUCEOP(REDUCEOP::SUM, REDUCE, {
                    SWITCH_COMPUTEOP(COMPUTEOP::ADD, COMPUTE, {
                        csrspmm_parreduce_rowbalance_kernel<int, float, float, REDUCE,
                                                        COMPUTE><<<gridDim, blockDim>>>(
                                Mdim_worker, Ndim_worker, d_ptr, d_ind, d_val,
                                d_B, d_C, d_E);
                    });
                });
            }
        }
    }

    //CPU计算
    if(is_transpose){
        for(int col = 0; col < size_n; col++){
            for(int row = 0; row < size_m; row++){
                int id = col * size_m + row;
                h_ref[id]=0;
                for(int k = 0; k<size_k;k++){
                    h_ref[id] += h_A[row * size_k + k] * h_B[k * size_n + col];
                }
            }
        }
    } else{
        for(int row = 0; row<size_m;row++){
            for(int col = 0; col<size_n;col++){
                int id = row*size_n+col;
                h_ref[id]=0;
                for(int k = 0; k<size_k;k++){
                    h_ref[id] += h_A[row * size_k + k] * h_B[k * size_n + col];
                }
            }
        }
    }
    //将结果从Device端传回Host端
    checkhiperror(hipMemcpy(h_C,d_C,size_C,hipMemcpyDeviceToHost));
    checkhiperror(hipDeviceSynchronize());

    bool flag = false;
    if(check_result(h_ref, h_C, size_m * size_n)){
        printf("###########spmm check pass!#############\n");
        flag = true;
    }


    // 释放内存
    hipFree(d_B);
    hipFree(d_B_trans);
    hipFree(d_C);
    hipFree(d_E);
    hipFree(d_ptr);
    hipFree(d_ind);
    hipFree(d_val);
    free(h_A);
    free(h_B);
    free(h_B_trans);
    free(h_C);
    free(E);
    free(h_ref);
    free(rowptr);
    free(rowind);
    free(value);

    csrptr.clear();csrptr.shrink_to_fit();
    csrind.clear();csrind.shrink_to_fit();
    csrval.clear();csrval.shrink_to_fit();
    return flag;
}

int main(int argc,char **argv)
{
    srand((int)time(0));
    spmm_check(1000, 1000, 128, method::dgsparse_0);
    spmm_check(1000, 1000, 128, method::rocsparse);
    spmm_check(1000, 1000, 128, method::dgsparse_1);
    spmm_check(1000, 1000, 128, method::dgsparse_2);
    int size_m[4]={1000, 5000, 10000};
    int size_k[4]={1000, 5000, 10000};
    int size_n[3]={32,64,128};

    float speedup_0 = 0, speedup_1 = 0, speedup_2 = 0;
    float time_ref, time_0, time_1, time_2;
    for(int i=0;i<3;++i){
        for(int j=0;j<3;++j){
            for(int k=0;k<3;++k){
                time_0 = spmm_time(size_m[i], size_k[j], size_n[k], 300, 0.01, method::dgsparse_0);
                time_1 = spmm_time(size_m[i], size_k[j], size_n[k], 300, 0.01, method::dgsparse_1);
                time_2 = spmm_time(size_m[i], size_k[j], size_n[k], 300, 0.01, method::dgsparse_2);
                time_ref = spmm_time(size_m[i], size_k[j], size_n[k], 300, 0.01, method::rocsparse);
                speedup_0 += time_ref/time_0;
                speedup_1 += time_ref/time_1;
                speedup_2 += time_ref/time_2;
            }
        }
    }
    speedup_0 /= 27;
    speedup_1 /= 27;
    speedup_2 /= 27;
    printf("average_speedup_0 = %f\n", speedup_0);
    printf("average_speedup_1 = %f\n", speedup_1);
    printf("average_speedup_2 = %f\n", speedup_2);
    return 0;
}

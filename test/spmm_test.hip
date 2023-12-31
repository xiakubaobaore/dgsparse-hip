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
#include "../include/hip/hip_util.hpp"

enum method{
    rocsparse = 0,
    dgsparse_0,
    dgsparse_1,
    dgsparse_2,
    dgsparse_3
};

__global__ void warm_up() {}

// A * B =C
// A: sparse, m*k
// B: dense, k*n
// C: dense, m*n
void init_random_sparse(float *A, int size, float threshold){
    for(int i=0;i<size;i++){
        if(rand()/(float)RAND_MAX < threshold){
            // A[i] = rand()/(float)RAND_MAX;
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
            // return false;
        }
    }
    return flag;
}

void generate_group(int nrow, int group_size, std::vector<int> &csrptr,
                    std::vector<int> &group_ptr,
                    std::vector<int> &group_row){
    group_ptr.push_back(0);
    for (int rid = 0; rid < nrow; rid++) {
        int A_lb = csrptr[rid];
        int A_hb = csrptr[rid + 1];
        int tmp_key = A_lb + group_size;
        while (tmp_key <= A_hb) {
            group_ptr.push_back(tmp_key);
            group_row.push_back(rid);
            tmp_key += group_size;
        }
        if (group_ptr.back() != A_hb) {
            group_row.push_back(rid);
            group_ptr.push_back(A_hb);
        }
    }
}

float spmm_time(float *h_A, int size_m, int size_k, int size_n, int iter, float sparsity, method algorithm)
{
    size_t size_A = size_m * size_k * sizeof(float);
    size_t size_B = size_n * size_k * sizeof(float);
    size_t size_C = size_m * size_n * sizeof(float);


    //申请Host内存并初始化
    float *h_B = (float *)malloc(size_B);
    float *h_B_trans = (float *)malloc(size_B);
    int *E = (int *)calloc(size_m * size_n, sizeof(int));

    if (h_B == NULL || E == NULL || h_B_trans == NULL){
        printf ("malloc failed\n");
        return false;
    }

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

    //申请Device内存
    float *d_B = NULL;
    checkHipError(hipMalloc((void **)&d_B, size_B));
    float *d_B_trans = NULL;
    checkHipError(hipMalloc((void **)&d_B_trans, size_B));
    float *d_C = NULL;
    checkHipError(hipMalloc((void **)&d_C, size_C));
    hipMemset((void *)d_C, 0, size_C);
    int *d_E = NULL;
    checkHipError(hipMalloc((void **)&d_E, size_m * size_n * sizeof(int)));
    int *d_ptr = NULL;
    checkHipError(hipMalloc((void **)&d_ptr, size_ptr));
    int *d_ind = NULL;
    checkHipError(hipMalloc((void **)&d_ind, size_ind));
    float *d_val = NULL;
    checkHipError(hipMalloc((void **)&d_val, size_val));

    //从Host端提交到Device端
    checkHipError(hipMemcpy(d_B,h_B,size_B,hipMemcpyHostToDevice));
    checkHipError(hipMemcpy(d_B_trans,h_B_trans,size_B,hipMemcpyHostToDevice));
    checkHipError(hipMemcpy(d_E,E,size_m * size_n * sizeof(int),hipMemcpyHostToDevice));
    checkHipError(hipMemcpy(d_ptr,&csrptr[0],size_ptr,hipMemcpyHostToDevice));
    checkHipError(hipMemcpy(d_ind,&csrind[0],size_ind,hipMemcpyHostToDevice));
    checkHipError(hipMemcpy(d_val,&csrval[0],size_val,hipMemcpyHostToDevice));

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
        for(int circle = 0; circle < iter; ++circle){
            rocsparse_scsrmm(rhandle, rocsparse_operation_none, rocsparse_operation_none,
                            size_m, size_n, size_k, csrval.size(), &alpha, rdescr,
                            d_val, d_ptr, d_ind, d_B_trans, size_k, &beta, d_C, size_m);
        }
        end = clock();
        time_elapsed = (end - start) * 1.0/iter;
        printf(" rocsparse  = %.3fms", time_elapsed);

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
        // for(int circle = 0; circle < iter; ++circle){
        //     hipsparseScsrmm(
        //         handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
        //         size_m, size_n, size_k, csrval.size(), &alpha, descr,
        //         d_val, d_ptr, d_ind,
        //         d_B_trans, size_k, &beta,
        //         d_C, size_m
        //     );
        // }
        // end = clock();
        // time_elapsed = (end - start) * 1.0/iter;
        // printf("hipsparse  = %.3fms\n", time_elapsed);
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
        for(int circle = 0; circle < iter; ++circle){
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
        time_elapsed = (end - start) * 1.0/iter;
        printf(" dgsparse_0 = %.3fms", time_elapsed);
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
        for(int circle = 0; circle < iter; ++circle){
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
        time_elapsed = (end - start) * 1.0/iter;
        printf(" dgsparse_1 = %.3fms", time_elapsed);
    }
    else if(algorithm == method::dgsparse_2){
        int Mdim_worker = size_m;
        int Ndim_worker = size_n;
        int coarsen_factor = (Ndim_worker % 4 == 0) ? 4 : (Ndim_worker % 2 == 0) ? 2 : 1;
        // partition large-N and map to blockdim.y to help cache performance
        int RefThreadPerBlock = 256;

        int Ndim_threadblock = CEIL(Ndim_worker, WARP_SIZE);
        int Ndim_warp_per_tb = min(Ndim_worker, WARP_SIZE) / coarsen_factor;
        int ref_warp_per_tb = RefThreadPerBlock / WARP_SIZE;
        int Mdim_warp_per_tb = CEIL(ref_warp_per_tb, Ndim_warp_per_tb);

        int gridDimX = CEIL(Mdim_worker, Mdim_warp_per_tb);
        int gridDimY = Ndim_threadblock;

        dim3 gridDim(gridDimX, gridDimY, 1);
        dim3 blockDim(Ndim_warp_per_tb * WARP_SIZE, Mdim_warp_per_tb, 1);

        if (coarsen_factor == 4){
            start = clock();
            for(int circle = 0; circle < iter; ++circle){
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
            time_elapsed = (end - start) * 1.0/iter;
        }
        else if (coarsen_factor == 2){
            start = clock();
            for(int circle = 0; circle < iter; ++circle){
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
            time_elapsed = (end - start) * 1.0/iter;
        }
        else {
            start = clock();
            for(int circle = 0; circle < iter; ++circle){
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
            time_elapsed = (end - start) * 1.0/iter;
        }
        printf(" dgsparse_2 = %.3fms", time_elapsed);
    }
    else if(algorithm == method::dgsparse_3){
        int Ndim_worker = size_n;
        // factor of thread coarsening
        int coarsen_factor = (Ndim_worker % 4 == 0) ? 4 : (Ndim_worker % 2 == 0) ? 2 : 1;
        // number of parallel warps along M-dimension
        const int segreduce_size_per_warp = WARP_SIZE;
        int Nnzdim_worker = csrval.size(); // CEIL(spmatA.nnz, segreduce_size_per_warp);
        // partition large-N and map to blockdim.y to help cache performance
        int Ndim_threadblock = CEIL(Ndim_worker, WARP_SIZE);
        int Ndim_warp_per_tb = min(Ndim_worker, WARP_SIZE) / coarsen_factor;
        // int Ndim_warp_per_tb = min(N, WARP_SIZE)

        int RefThreadPerBlock = 256;
        int ref_warp_per_tb = RefThreadPerBlock / WARP_SIZE;
        int Nnzdim_warp_per_tb = CEIL(ref_warp_per_tb, Ndim_warp_per_tb);

        // total number of warps
        int gridDimX = CEIL(Nnzdim_worker, Nnzdim_warp_per_tb);
        int gridDimY = Ndim_threadblock;
        dim3 gridDim(gridDimX, gridDimY, 1);
        dim3 blockDim(Ndim_warp_per_tb * WARP_SIZE, Nnzdim_warp_per_tb, 1);

        start = clock();
        for(int circle = 0; circle < iter; ++circle){
            if (coarsen_factor == 4) {
                csrspmm_parreduce_nnzbalance_kernel<int,float,float4><<<gridDim,
                blockDim>>>(
                    size_m, Ndim_worker, Nnzdim_worker, d_ptr, d_ind, d_val, d_B, d_C);
            } else if (coarsen_factor == 2) {
                csrspmm_parreduce_nnzbalance_kernel<int,float,float2><<<gridDim,
                blockDim>>>(
                    size_m, Ndim_worker, Nnzdim_worker, d_ptr, d_ind, d_val, d_B, d_C);
            } else {
                csrspmm_parreduce_nnzbalance_kernel<int,float,float><<<gridDim,
                blockDim>>>(
                    size_m, Ndim_worker, Nnzdim_worker, d_ptr, d_ind, d_val, d_B, d_C);
            }
        }
        end = clock();
        time_elapsed = (end - start) * 1.0/iter;
        printf(" dgsparse_3 = %.3fms", time_elapsed);
    }

    checkHipError(hipDeviceSynchronize());

    // 释放内存
    hipFree(d_B);
    hipFree(d_B_trans);
    hipFree(d_C);
    hipFree(d_E);
    hipFree(d_ptr);
    hipFree(d_ind);
    hipFree(d_val);
    free(h_B);
    free(h_B_trans);
    free(E);

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

    //申请Device内存
    float *d_B = NULL;
    checkHipError(hipMalloc((void **)&d_B, size_B));
    float *d_B_trans = NULL;
    checkHipError(hipMalloc((void **)&d_B_trans, size_B));
    float *d_C = NULL;
    checkHipError(hipMalloc((void **)&d_C, size_C));
    hipMemset((void *)d_C, 0, size_C);
    int *d_E = NULL;
    checkHipError(hipMalloc((void **)&d_E, size_m * size_n * sizeof(int)));
    int *d_ptr = NULL;
    checkHipError(hipMalloc((void **)&d_ptr, size_ptr));
    int *d_ind = NULL;
    checkHipError(hipMalloc((void **)&d_ind, size_ind));
    float *d_val = NULL;
    checkHipError(hipMalloc((void **)&d_val, size_val));

    //从Host端提交到Device端
    checkHipError(hipMemcpy(d_B,h_B,size_B,hipMemcpyHostToDevice));
    checkHipError(hipMemcpy(d_B_trans,h_B_trans,size_B,hipMemcpyHostToDevice));
    checkHipError(hipMemcpy(d_E,E,size_m * size_n * sizeof(int),hipMemcpyHostToDevice));
    checkHipError(hipMemcpy(d_ptr,&csrptr[0],size_ptr,hipMemcpyHostToDevice));
    checkHipError(hipMemcpy(d_ind,&csrind[0],size_ind,hipMemcpyHostToDevice));
    checkHipError(hipMemcpy(d_val,&csrval[0],size_val,hipMemcpyHostToDevice));

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
    }else if(algorithm == method::dgsparse_0){
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

        int Ndim_threadblock = CEIL(Ndim_worker, WARP_SIZE);
        int Ndim_warp_per_tb = min(Ndim_worker, WARP_SIZE) / coarsen_factor;
        int ref_warp_per_tb = RefThreadPerBlock / WARP_SIZE;
        int Mdim_warp_per_tb = CEIL(ref_warp_per_tb, Ndim_warp_per_tb);

        int gridDimX = CEIL(Mdim_worker, Mdim_warp_per_tb);
        int gridDimY = Ndim_threadblock;

        dim3 gridDim(gridDimX, gridDimY, 1);
        dim3 blockDim(Ndim_warp_per_tb * WARP_SIZE, Mdim_warp_per_tb, 1);

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
    else if(algorithm == method::dgsparse_3){
        int Ndim_worker = size_n;
        // factor of thread coarsening
        int coarsen_factor = (Ndim_worker % 4 == 0) ? 4 : (Ndim_worker % 2 == 0) ? 2 : 1;
        // number of parallel warps along M-dimension
        const int segreduce_size_per_warp = WARP_SIZE;
        int Nnzdim_worker = csrval.size(); // CEIL(spmatA.nnz, segreduce_size_per_warp);
        // partition large-N and map to blockdim.y to help cache performance
        int Ndim_threadblock = CEIL(Ndim_worker, WARP_SIZE);
        int Ndim_warp_per_tb = min(Ndim_worker, WARP_SIZE) / coarsen_factor;
        // int Ndim_warp_per_tb = min(N, WARP_SIZE)

        int RefThreadPerBlock = 256;
        int ref_warp_per_tb = RefThreadPerBlock / WARP_SIZE;
        int Nnzdim_warp_per_tb = CEIL(ref_warp_per_tb, Ndim_warp_per_tb);

        // total number of warps
        int gridDimX = CEIL(Nnzdim_worker, Nnzdim_warp_per_tb);
        int gridDimY = Ndim_threadblock;
        dim3 gridDim(gridDimX, gridDimY, 1);
        dim3 blockDim(Ndim_warp_per_tb * WARP_SIZE, Nnzdim_warp_per_tb, 1);

        if (coarsen_factor == 4) {
            csrspmm_parreduce_nnzbalance_kernel<int,float,float4><<<gridDim,
            blockDim>>>(
                size_m, Ndim_worker, Nnzdim_worker, d_ptr, d_ind, d_val, d_B, d_C);
        } else if (coarsen_factor == 2) {
            csrspmm_parreduce_nnzbalance_kernel<int,float,float2><<<gridDim,
            blockDim>>>(
                size_m, Ndim_worker, Nnzdim_worker, d_ptr, d_ind, d_val, d_B, d_C);
        } else {
            csrspmm_parreduce_nnzbalance_kernel<int,float,float><<<gridDim,
            blockDim>>>(
                size_m, Ndim_worker, Nnzdim_worker, d_ptr, d_ind, d_val, d_B, d_C);
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
    checkHipError(hipMemcpy(h_C,d_C,size_C,hipMemcpyDeviceToHost));
    checkHipError(hipDeviceSynchronize());

    bool flag = false;
    if(check_result(h_ref, h_C, size_m * size_n)){
        printf("\n###########spmm check pass!#############\n");
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

    csrptr.clear();csrptr.shrink_to_fit();
    csrind.clear();csrind.shrink_to_fit();
    csrval.clear();csrval.shrink_to_fit();
    return flag;
}

float spmm_group_time(float *h_A, int size_m, int size_k, int size_n, int iter, float sparsity)
{
    size_t size_A = size_m * size_k * sizeof(float);
    size_t size_B = size_n * size_k * sizeof(float);
    size_t size_C = size_m * size_n * sizeof(float);


    //申请Host内存并初始化
    float *h_B = (float *)malloc(size_B);

    if (h_B == NULL){
        printf ("malloc failed\n");
        return false;
    }
    init_random(h_B, size_k*size_n);

    std::vector<int> csrptr, csrind, group_ptr, group_row;
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

    generate_group(size_m, 4, csrptr, group_ptr, group_row);

    size_t size_ind = csrind.size() * sizeof(int);
    size_t size_val = csrval.size() * sizeof(float);
    size_t size_groupptr = group_ptr.size() * sizeof(int);
    size_t size_grouprow = group_row.size() * sizeof(int);

    //申请Device内存
    float *d_B = NULL;
    checkHipError(hipMalloc((void **)&d_B, size_B));
    float *d_C = NULL;
    checkHipError(hipMalloc((void **)&d_C, size_C));
    hipMemset((void *)d_C, 0, size_C);
    int *d_groupptr = NULL;
    checkHipError(hipMalloc((void **)&d_groupptr, size_groupptr));
    int *d_grouprow = NULL;
    checkHipError(hipMalloc((void **)&d_grouprow, size_grouprow));
    int *d_ind = NULL;
    checkHipError(hipMalloc((void **)&d_ind, size_ind));
    float *d_val = NULL;
    checkHipError(hipMalloc((void **)&d_val, size_val));

    //从Host端提交到Device端
    checkHipError(hipMemcpy(d_B,h_B,size_B,hipMemcpyHostToDevice));
    checkHipError(hipMemcpy(d_groupptr,&group_ptr[0],size_groupptr,hipMemcpyHostToDevice));
    checkHipError(hipMemcpy(d_grouprow,&group_row[0],size_grouprow,hipMemcpyHostToDevice));
    checkHipError(hipMemcpy(d_ind,&csrind[0],size_ind,hipMemcpyHostToDevice));
    checkHipError(hipMemcpy(d_val,&csrval[0],size_val,hipMemcpyHostToDevice));

    for (int i = 0; i < 1000; i++)
        warm_up<<<1, 1>>>();

    int Mdim_worker = group_row.size();
    int Ndim_worker = size_n;

    int RefThreadPerBlock = Ndim_worker > 256 ? Ndim_worker : 256;
    int Ndim_threadblock = CEIL(Ndim_worker, RefThreadPerBlock);
    int Ndim_thread_per_tb = min(Ndim_worker, RefThreadPerBlock);
    int Mdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
    int Mdim_threadblock = CEIL(Mdim_worker, Mdim_thread_per_tb);

    dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
    dim3 blockDim(Ndim_thread_per_tb, Mdim_thread_per_tb, 1);

    clock_t start, end;
    start = clock();
    for(int i = 0; i < iter; ++i){
        csrspmm_neighbor_group_kernel<int, float>
            <<<gridDim, blockDim>>>(
                Mdim_worker, Ndim_worker, d_groupptr, d_grouprow,
                d_ind, d_val, d_B, d_C);
    }
    end = clock();
    float time_elapsed = (end - start) * 1.0/iter;
    printf(" dgsparse_group = %.3fms", time_elapsed);

    checkHipError(hipDeviceSynchronize());

    // 释放内存
    hipFree(d_B);
    hipFree(d_C);
    hipFree(d_groupptr);
    hipFree(d_grouprow);
    hipFree(d_ind);
    hipFree(d_val);
    free(h_B);

    csrptr.clear();csrptr.shrink_to_fit();
    csrind.clear();csrind.shrink_to_fit();
    csrval.clear();csrval.shrink_to_fit();
    group_ptr.clear();group_ptr.shrink_to_fit();
    group_row.clear();group_row.shrink_to_fit();

    return time_elapsed;
}

bool spmm_group_check(int size_m, int size_k, int size_n)
{
    size_t size_A = size_m * size_k * sizeof(float);
    size_t size_B = size_n * size_k * sizeof(float);
    size_t size_C = size_m * size_n * sizeof(float);


    //申请Host内存并初始化
    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C = (float *)malloc(size_C);

    float *h_ref = (float *)malloc(size_C); //参考结果

    if (h_A == NULL || h_B == NULL || h_C == NULL || h_ref == NULL){
        printf ("malloc failed\n");
        return false;
    }
    init_random_sparse(h_A, size_m*size_k, 0.01);
    init_random(h_B, size_k*size_n);

    std::vector<int> csrptr, csrind, group_ptr, group_row;
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

    generate_group(size_m, 4, csrptr, group_ptr, group_row);

    size_t size_ind = csrind.size() * sizeof(int);
    size_t size_val = csrval.size() * sizeof(float);
    size_t size_groupptr = group_ptr.size() * sizeof(int);
    size_t size_grouprow = group_row.size() * sizeof(int);

    //申请Device内存
    float *d_B = NULL;
    checkHipError(hipMalloc((void **)&d_B, size_B));
    float *d_C = NULL;
    checkHipError(hipMalloc((void **)&d_C, size_C));
    hipMemset((void *)d_C, 0, size_C);
    int *d_groupptr = NULL;
    checkHipError(hipMalloc((void **)&d_groupptr, size_groupptr));
    int *d_grouprow = NULL;
    checkHipError(hipMalloc((void **)&d_grouprow, size_grouprow));
    int *d_ind = NULL;
    checkHipError(hipMalloc((void **)&d_ind, size_ind));
    float *d_val = NULL;
    checkHipError(hipMalloc((void **)&d_val, size_val));

    //从Host端提交到Device端
    checkHipError(hipMemcpy(d_B,h_B,size_B,hipMemcpyHostToDevice));
    checkHipError(hipMemcpy(d_groupptr,&group_ptr[0],size_groupptr,hipMemcpyHostToDevice));
    checkHipError(hipMemcpy(d_grouprow,&group_row[0],size_grouprow,hipMemcpyHostToDevice));
    checkHipError(hipMemcpy(d_ind,&csrind[0],size_ind,hipMemcpyHostToDevice));
    checkHipError(hipMemcpy(d_val,&csrval[0],size_val,hipMemcpyHostToDevice));

    int Mdim_worker = group_row.size();
    int Ndim_worker = size_n;

    int RefThreadPerBlock = Ndim_worker > 256 ? Ndim_worker : 256;
    int Ndim_threadblock = CEIL(Ndim_worker, RefThreadPerBlock);
    int Ndim_thread_per_tb = min(Ndim_worker, RefThreadPerBlock);
    int Mdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
    int Mdim_threadblock = CEIL(Mdim_worker, Mdim_thread_per_tb);

    dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
    dim3 blockDim(Ndim_thread_per_tb, Mdim_thread_per_tb, 1);

    csrspmm_neighbor_group_kernel<int, float>
        <<<gridDim, blockDim>>>(
            Mdim_worker, Ndim_worker, d_groupptr, d_grouprow,
            d_ind, d_val, d_B, d_C);

    //CPU计算
    for(int row = 0; row<size_m;row++){
        for(int col = 0; col<size_n;col++){
            int id = row*size_n+col;
            h_ref[id]=0;
            for(int k = 0; k<size_k;k++){
                h_ref[id] += h_A[row * size_k + k] * h_B[k * size_n + col];
            }
        }
    }
    //将结果从Device端传回Host端
    checkHipError(hipMemcpy(h_C,d_C,size_C,hipMemcpyDeviceToHost));
    checkHipError(hipDeviceSynchronize());

    bool flag = false;
    if(check_result(h_ref, h_C, size_m * size_n)){
        printf("\n###########spmm group check pass!#############\n");
        flag = true;
    }


    // 释放内存
    hipFree(d_B);
    hipFree(d_C);
    hipFree(d_groupptr);
    hipFree(d_grouprow);
    hipFree(d_ind);
    hipFree(d_val);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_ref);

    csrptr.clear();csrptr.shrink_to_fit();
    csrind.clear();csrind.shrink_to_fit();
    csrval.clear();csrval.shrink_to_fit();
    group_ptr.clear();group_ptr.shrink_to_fit();
    group_row.clear();group_row.shrink_to_fit();

    return flag;
}

int main(int argc,char **argv)
{
    srand((int)time(0));

    //check correctness
    spmm_check(1000, 1000, 128, method::dgsparse_0);
    spmm_check(1000, 1000, 128, method::rocsparse);
    spmm_check(1000, 1000, 128, method::dgsparse_1);
    spmm_check(1000, 1000, 128, method::dgsparse_2);
    spmm_check(1000, 1000, 128, method::dgsparse_3);
    spmm_group_check(1000, 1000, 128);

    int size_m[4]={1000, 5000, 10000};
    int size_k[4]={1000, 5000, 10000};
    int size_n[3]={32,64,128};

    float speedup_0 = 0, speedup_1 =0, speedup_2 = 0, speedup_3 = 0, speedup_group = 0;
    float time_ref, time_0, time_1, time_2, time_3, time_group;
    for(int i=0;i<3;++i){
        for(int j=0;j<3;++j){
            for(int k=0;k<3;++k){
                size_t size_sparse = size_m[i] * size_k[j] * sizeof(float);
                float *sparse = (float *)malloc(size_sparse);
                init_random_sparse(sparse, size_m[i] * size_k[j], 0.01);

                printf("[Spmm of %d*%d*%d", size_m[i], size_k[j], size_n[k]);
                time_0 = spmm_time(sparse, size_m[i], size_k[j], size_n[k], 300, 0.01, method::dgsparse_0);
                time_1 = spmm_time(sparse, size_m[i], size_k[j], size_n[k], 300, 0.01, method::dgsparse_1);
                time_2 = spmm_time(sparse, size_m[i], size_k[j], size_n[k], 300, 0.01, method::dgsparse_2);
                time_3 = spmm_time(sparse, size_m[i], size_k[j], size_n[k], 300, 0.01, method::dgsparse_3);
                time_group = spmm_group_time(sparse, size_m[i], size_k[j], size_n[k], 300, 0.01);
                time_ref = spmm_time(sparse, size_m[i], size_k[j], size_n[k], 300, 0.01, method::rocsparse);
                printf("\n");
                speedup_0 += time_ref/time_0;
                speedup_1 += time_ref/time_1;
                speedup_2 += time_ref/time_2;
                speedup_3 += time_ref/time_3;
                speedup_group += time_ref/time_group;

                free(sparse);
            }
        }
    }
    speedup_0 /= 27;
    speedup_1 /= 27;
    speedup_2 /= 27;
    speedup_3 /= 27;
    speedup_group /= 27;
    printf("average_speedup_0 = %.3f\n", speedup_0);
    printf("average_speedup_1 = %.3f\n", speedup_1);
    printf("average_speedup_2 = %.3f\n", speedup_2);
    printf("average_speedup_3 = %.3f\n", speedup_3);
    printf("average_speedup_group = %.3f\n", speedup_group);
    return 0;
}

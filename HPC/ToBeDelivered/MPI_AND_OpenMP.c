//mpicc -O3 -fopenmp -o financial_parallel financial_parallel.c -lm
//mpirun -np 4 ./financial_parallel
/*
#!/bin/bash
#SBATCH --account=tra24_IngInfB2
#SBATCH --partition=g100_usr_prod
#SBATCH --nodes=2                     
#SBATCH --ntasks-per-node=4           
#SBATCH --cpus-per-task=4             
#SBATCH -o TEST2_Strong.out
#SBATCH -e TEST2_Strong.err

# Carica i moduli necessari
module load autoload intelmpi

# Configurazione OpenMP
export OMP_NUM_THREADS=4
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Configurazione Intel MPI
export I_MPI_PIN_DOMAIN=auto
export I_MPI_PIN_ORDER=compact
export I_MPI_DEBUG=5  # Per debugging delle assegnazioni

# Esegui con statistiche dettagliate
srun --cpu-bind=cores ./hybrid_program

*/
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define DAYS 365*10
#define TITLES 2000
#define INDEX 2000

typedef struct {
    float* prices;
    float* returns;
    float* averages;
    float* variances;
    float* stdDevs;
    float* covariances;
    float** betas;
    float** correlations;
} FinancialData;

void cleanupFinancialData(FinancialData* data) {
    if (data) {
        if (data->prices) free(data->prices);
        if (data->returns) free(data->returns);
        if (data->averages) free(data->averages);
        if (data->variances) free(data->variances);
        if (data->stdDevs) free(data->stdDevs);
        if (data->covariances) free(data->covariances);
        if (data->betas) {
            for (int i = 0; i < TITLES; i++) {
                if (data->betas[i]) free(data->betas[i]);
            }
            free(data->betas);
        }
        if (data->correlations) {
            for (int i = 0; i < TITLES; i++) {
                if (data->correlations[i]) free(data->correlations[i]);
            }
            free(data->correlations);
        }
    }
}
void initializeData(float* prices) {
    unsigned int seed = time(NULL);
    #pragma omp parallel
    {
        unsigned int local_seed = seed + omp_get_thread_num();
        #pragma omp for collapse(2) schedule(static)
        for (int i = 0; i < TITLES + INDEX; i++) {
            for (int j = 0; j < DAYS; j++) {
                float base_price = 50 + (rand_r(&local_seed) % 151);
                prices[i * DAYS + j] = base_price + ((float)rand_r(&local_seed) / RAND_MAX);
            }
        }
    }
}

void calculateReturns(float* prices, float* returns, int local_size, int rank) {
    #pragma omp parallel for collapse(2) schedule(guided)
    for (int row = 0; row < local_size; row++) {
        for (int day = 0; day < (DAYS - 1); day++) {
            returns[row * (DAYS - 1) + day] = (prices[row * DAYS + day + 1] - prices[row * DAYS + day]) / prices[row * DAYS + day];
        }
    }
}

void calculateAverages(float* returns, float* averages, int local_size, int rank) {
    #pragma omp parallel for schedule(guided)
    for (int row = 0; row < local_size; row++) {
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for (int day = 0; day < (DAYS - 1); day++) {
            sum += returns[row * (DAYS - 1) + day];
        }
        averages[row] = sum / (DAYS - 1);
    }
}

void calculateVariances(float* returns, float* averages, float* variances, float* stdDevs, int local_size, int rank) {
    #pragma omp parallel for schedule(guided)
    for (int row = 0; row < local_size; row++) {
        float variance = 0.0f;
        float avg = averages[row];
        
        #pragma omp simd reduction(+:variance)
        for (int day = 0; day < (DAYS - 1); day++) {
            float diff = returns[row * (DAYS - 1) + day] - avg;
            variance += diff * diff;
        }
        
        variances[row] = variance / (DAYS - 1);
        stdDevs[row] = sqrtf(variances[row]);
    }
}

void calculateCovariances(float* returns, float* averages, float* covariances, int size, int rank, int* recvcounts, int* displs, float* local_cov) {
    int index_block_size = INDEX / size;
    int index_remainder = INDEX % size;
    int index_start_idx = rank * index_block_size + (rank < index_remainder ? rank : index_remainder);
    int index_end_idx = index_start_idx + index_block_size + (rank < index_remainder ? 1 : 0);
    int local_index_size = index_end_idx - index_start_idx;

    #pragma omp parallel
    {
        #pragma omp for collapse(2) schedule(guided)
        for (int i = 0; i < local_index_size; i++) {
            for (int t = 0; t < TITLES; t++) {
                float sum = 0.0f;
                float avg_t = averages[t];
                float avg_i = averages[TITLES + index_start_idx + i];
                
                #pragma omp simd reduction(+:sum)
                for (int d = 0; d < DAYS - 1; d++) {
                    sum += (returns[t * (DAYS - 1) + d] - avg_t) * 
                          (returns[(TITLES + index_start_idx + i) * (DAYS - 1) + d] - avg_i);
                }
                local_cov[i * TITLES + t] = sum / (DAYS - 1);
            }
        }
    }
    
    MPI_Allgatherv(local_cov, local_index_size * TITLES, MPI_FLOAT, 
                   covariances, recvcounts, displs, MPI_FLOAT, MPI_COMM_WORLD);
}

void calculateBetasAndCorrelations(float* covariances, float* variances, float** betas, float** correlations, int size, int rank) {
    int total_combinations = TITLES * INDEX;
    int block_size = total_combinations / size;
    int remainder = total_combinations % size;
    int start_idx = rank * block_size + (rank < remainder ? rank : remainder);
    int end_idx = start_idx + block_size + (rank < remainder ? 1 : 0);
    
    const float epsilon = 1e-7f;
    
    #pragma omp parallel for collapse(2) schedule(guided)
    for (int title = 0; title < TITLES; title++) {
        for (int idx = 0; idx < INDEX; idx++) {
            int i = title * INDEX + idx;
            if (i >= start_idx && i < end_idx) {
                float var_idx = variances[TITLES + idx];
                float var_title = variances[title];
                float covar = covariances[title + idx * TITLES];
                
                betas[title][idx] = (fabsf(var_idx) > epsilon) ? 
                    covar / var_idx : 0.0f;
                
                correlations[title][idx] = (fabsf(var_idx) > epsilon && fabsf(var_title) > epsilon) ?
                    covar / (sqrtf(var_idx) * sqrtf(var_title)) : 0.0f;
            }
        }
    }
}
int main(int argc, char* argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        printf("Warning: The MPI implementation does not support MPI_THREAD_FUNNELED\n");
        MPI_Finalize();
        return 1;
    }

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Set number of OpenMP threads
    int num_threads = 4; // Adjust based on your system
    omp_set_num_threads(num_threads);

    double begin, end, global_elaps, local_elaps;
    FinancialData data = {0};

    // Allocate memory
    data.prices = (float*)malloc((TITLES + INDEX) * DAYS * sizeof(float));
    data.returns = (float*)malloc((TITLES + INDEX) * (DAYS - 1) * sizeof(float));
    data.averages = (float*)malloc((TITLES + INDEX) * sizeof(float));
    data.variances = (float*)malloc((TITLES + INDEX) * sizeof(float));
    data.stdDevs = (float*)malloc((TITLES + INDEX) * sizeof(float));
    data.covariances = (float*)malloc(TITLES * INDEX * sizeof(float));
    data.betas = (float**)malloc(TITLES * sizeof(float*));
    data.correlations = (float**)malloc(TITLES * sizeof(float*));

    // Check allocations
    if (!data.prices || !data.returns || !data.averages || !data.variances || 
        !data.stdDevs || !data.covariances || !data.betas || !data.correlations) {
        fprintf(stderr, "Memory allocation failed\n");
        cleanupFinancialData(&data);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Initialize 2D arrays
    for (int i = 0; i < TITLES; i++) {
        data.betas[i] = (float*)malloc(INDEX * sizeof(float));
        data.correlations[i] = (float*)malloc(INDEX * sizeof(float));
        if (!data.betas[i] || !data.correlations[i]) {
            fprintf(stderr, "Memory allocation for 2D arrays failed\n");
            cleanupFinancialData(&data);
            MPI_Finalize();
            return EXIT_FAILURE;
        }
        for (int j = 0; j < INDEX; j++) {
            data.betas[i][j] = 0.0f;
            data.correlations[i][j] = 0.0f;
        }
    }

    // Initialize arrays for MPI communications
    int* recvcounts = (int*)malloc(size * sizeof(int));
    int* displs = (int*)malloc(size * sizeof(int));
    
    // Calculate receive counts and displacements for covariance gathering
    for (int i = 0; i < size; i++) {
        int i_block_size = INDEX / size;
        int i_remainder = INDEX % size;
        int i_start = i * i_block_size + (i < i_remainder ? i : i_remainder);
        int i_end = i_start + i_block_size + (i < i_remainder ? 1 : 0);
        int l_i_size = i_end - i_start;

        recvcounts[i] = l_i_size * TITLES;
        displs[i] = (i > 0) ? displs[i - 1] + recvcounts[i - 1] : 0;
    }

    // Work distribution
    int block_size = (int)ceil((double)(TITLES + INDEX) / size);
    int start_idx = rank * block_size;
    int end_idx = (rank + 1) * block_size;
    if (end_idx > TITLES + INDEX) end_idx = TITLES + INDEX;
    int local_size = end_idx - start_idx;

    // Allocate local arrays
    float* localReturns = (float*)malloc(local_size * (DAYS - 1) * sizeof(float));
    float* localAverage = (float*)malloc(local_size * sizeof(float));
    float* localVariance = (float*)malloc(local_size * sizeof(float));
    float* localDev = (float*)malloc(local_size * sizeof(float));
    float* local_cov = (float*)malloc((TITLES * local_size) * sizeof(float));

    if (!localReturns || !localAverage || !localVariance || !localDev || !local_cov) {
        fprintf(stderr, "Memory allocation for local arrays failed\n");
        cleanupFinancialData(&data);
        free(localReturns); free(localAverage); free(localVariance); free(localDev);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Initialize local covariance array
    for (int i = 0; i < (TITLES * local_size); i++) {
        local_cov[i] = 0.0f;
    }

    // Initialize global covariance array
    for (int i = 0; i < TITLES * INDEX; i++) {
        data.covariances[i] = 0.0f;
    }

    // Initialize data on root process
    if (rank == 0) {
        initializeData(data.prices);
    }

    // Setup for MPI communications
    int* sendcounts_returns = (int*)malloc(size * sizeof(int));
    int* sendcounts_other = (int*)malloc(size * sizeof(int));
    int* displs_returns = (int*)malloc(size * sizeof(int));
    int* displs_other = (int*)malloc(size * sizeof(int));

    for (int i = 0; i < size; i++) {
        int b_size = (int)ceil((double)(TITLES + INDEX) / size);
        int start = i * b_size;
        int end = (i + 1) * b_size;
        if (end > TITLES + INDEX) end = TITLES + INDEX;
        int l_size = end - start;
        
        sendcounts_returns[i] = l_size * (DAYS - 1);
        sendcounts_other[i] = l_size;
        displs_returns[i] = (i > 0) ? displs_returns[i - 1] + sendcounts_returns[i - 1] : 0;
        displs_other[i] = (i > 0) ? displs_other[i - 1] + sendcounts_other[i - 1] : 0;
    }

    // Broadcast initial prices to all processes
    MPI_Bcast(data.prices, (TITLES + INDEX) * DAYS, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Start timing
    begin = MPI_Wtime();

    // Calculate returns, averages, and variances using OpenMP within each MPI process
    calculateReturns(data.prices + start_idx * DAYS, localReturns, local_size, rank);
    calculateAverages(localReturns, localAverage, local_size, rank);
    calculateVariances(localReturns, localAverage, localVariance, localDev, local_size, rank);

    // Gather results from all processes
    MPI_Allgatherv(localReturns, local_size * (DAYS - 1), MPI_FLOAT,
                   data.returns, sendcounts_returns, displs_returns, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgatherv(localAverage, local_size, MPI_FLOAT,
                   data.averages, sendcounts_other, displs_other, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgatherv(localVariance, local_size, MPI_FLOAT,
                   data.variances, sendcounts_other, displs_other, MPI_FLOAT, MPI_COMM_WORLD);

    // Calculate covariances using hybrid parallelization
    calculateCovariances(data.returns, data.averages, data.covariances, size, rank, recvcounts, displs, local_cov);

    // Calculate betas and correlations
    calculateBetasAndCorrelations(data.covariances, data.variances, data.betas, data.correlations, size, rank);

    // End timing
    end = MPI_Wtime();
    local_elaps = end - begin;
    MPI_Reduce(&local_elaps, &global_elaps, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Print final statistics
    if (rank == 0) {
        printf("Number of MPI processes: %d\n", size);
        printf("Number of OpenMP threads per process: %d\n", num_threads);
        printf("Total parallel units: %d\n", size * num_threads);
        printf("Execution time: %f seconds\n", global_elaps);
        printf("Input size: %d elements\n", (TITLES + INDEX) * DAYS);
        printf("Memory used: %lu MB\n", ((TITLES + INDEX) * DAYS * sizeof(float)) / (1024 * 1024));
    }

   // Cleanup
    free(local_cov);
    free(sendcounts_returns);
    free(sendcounts_other);
    free(displs_returns);
    free(displs_other);
    free(localReturns);
    free(localAverage);
    free(localVariance);
    free(localDev);
    free(recvcounts);
    free(displs);
    cleanupFinancialData(&data);

    MPI_Finalize();
    return 0;
}

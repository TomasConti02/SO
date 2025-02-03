#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>

#define DAYS 10
#define TITLES 10
#define INDEX 10

// Structure to hold financial data
typedef struct {
    float* prices;
    float* returns;
    float* averages;
    float* variances;
    float* stdDevs;
    float** covariances;
    float** betas;
    float** correlations;
} FinancialData;

// Function prototypes
void initializeData(float* prices);
void calculateReturns(float* prices, float* returns, int size, int rank);
void calculateAverages(float* returns, float* averages, int size, int rank, float* local_sum, float* global_sum);
void calculateVariances(float* returns, float* averages, float* variances, float* stdDevs, int size, int rank, float* local_var, float* global_var);
void calculateCovariances(float* returns, float* averages, float** covariances, int size, int rank, float (*local_cov)[INDEX], float (*global_cov)[INDEX]);
void calculateBetasAndCorrelations(float** covariances, float* variances, float* stdDevs, float** betas, float** correlations, int size, int rank);
void cleanupData(FinancialData* data);
int main(int argc, char* argv[]) {
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double begin, end, global_elaps, local_elaps;
     FinancialData data = {
        .prices = (float*)malloc((TITLES + INDEX) * DAYS * sizeof(float)), //prezzi di partenza 
        .returns = (float*)malloc((TITLES + INDEX) * (DAYS - 1) * sizeof(float)), //rendimenti risultatinti 
        .averages = (float*)malloc((TITLES + INDEX) * sizeof(float)),
        .variances = (float*)malloc((TITLES + INDEX) * sizeof(float)),
        .stdDevs = (float*)malloc((TITLES + INDEX) * sizeof(float)),
        .covariances = (float**)malloc(TITLES * sizeof(float*)),
        .betas = (float**)malloc(TITLES * sizeof(float*)),
        .correlations = (float**)malloc(TITLES * sizeof(float*))
    };

    // Allocate 2D arrays
    for (int i = 0; i < TITLES; i++) {
        data.covariances[i] = (float*)malloc(INDEX * sizeof(float));
        data.betas[i] = (float*)malloc(INDEX * sizeof(float));
        data.correlations[i] = (float*)malloc(INDEX * sizeof(float));
    }

    // Initialize price data (only in rank 0)
    if (rank == 0) {
        initializeData(data.prices);
    }

    float *local_var = (float*)malloc((TITLES + INDEX) * sizeof(float));
    float *global_var = (float*)malloc((TITLES + INDEX) * sizeof(float));
    float (*local_cov)[INDEX] = (float(*)[INDEX])malloc(TITLES * sizeof(float[INDEX]));
    float (*global_cov)[INDEX] = (float(*)[INDEX])malloc(TITLES * sizeof(float[INDEX]));
    float *local_sum = (float*)malloc((TITLES + INDEX) * sizeof(float));
    float *global_sum = (float*)malloc((TITLES + INDEX) * sizeof(float));
    for (int i = 0; i < TITLES + INDEX; i++) {
        local_var[i] = 0.0f;
        global_var[i] = 0.0f;
        local_sum[i] = 0.0f;
        global_sum[i] = 0.0f;
    }
    for (int i = 0; i < TITLES; i++) {
        for (int j = 0; j < INDEX; j++) {
            local_cov[i][j] = 0.0f;
            global_cov[i][j] = 0.0f;
        }
    }
    MPI_Bcast(data.prices, (TITLES + INDEX) * DAYS, MPI_FLOAT, 0, MPI_COMM_WORLD);
    begin = MPI_Wtime();
    calculateReturns(data.prices, data.returns, size, rank);
    calculateAverages(data.returns, data.averages, size, rank, local_sum, global_sum);
    calculateVariances(data.returns, data.averages, data.variances, data.stdDevs, size, rank, local_var, global_var);
    calculateCovariances(data.returns, data.averages, data.covariances, size, rank, local_cov, global_cov);
    calculateBetasAndCorrelations(data.covariances, data.variances, data.stdDevs, data.betas, data.correlations, size, rank);

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    local_elaps = end - begin;
    MPI_Reduce(&local_elaps, &global_elaps, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Number of processes: %d\n", size);
        printf("Execution time: %f seconds\n", global_elaps);
        printf("Input size: %d elements\n", (TITLES + INDEX) * DAYS);
        printf("Memory used: %lu MB\n", ((TITLES + INDEX) * DAYS * sizeof(float)) / (1024 * 1024));
    }

    free(local_var);
    free(global_var);
    free(local_cov);
    free(global_cov);
    free(local_sum);
    free(global_sum);

    cleanupData(&data);
    MPI_Finalize();
    return 0;
}

void initializeData(float* prices) {
    srand(time(NULL));
    for (int j = 0; j < DAYS; j++) {
        prices[0 * DAYS + j] = 100 + (rand() % 51) + ((float)rand() / RAND_MAX);
        prices[1 * DAYS + j] = 80 + (rand() % 41) + ((float)rand() / RAND_MAX);
        prices[2 * DAYS + j] = 100 + (rand() % 201) + ((float)rand() / RAND_MAX);
        prices[3 * DAYS + j] = 50 + (rand() % 51) + ((float)rand() / RAND_MAX);
    }
}

void calculateReturns(float* prices, float* returns, int size, int rank) {
    int total_elements = (TITLES + INDEX) * (DAYS - 1);
    int block_size = total_elements / size;
    int remainder = total_elements % size;

    int start_idx = rank * block_size;
    int end_idx = (rank < size - 1) ? start_idx + block_size : start_idx + block_size + remainder;
    int local_count = end_idx - start_idx;

    // Create a separate receive buffer for gathering
    float* gathered_returns = (float*)malloc(total_elements * sizeof(float));

    // Calculate local returns
    for (int i = start_idx; i < end_idx; i++) {
        int row = i / (DAYS - 1);
        int day = i % (DAYS - 1);
        returns[i] = (prices[row * DAYS + day + 1] - prices[row * DAYS + day]) / prices[row * DAYS + day];
    }

    // Gather returns from all processes
    MPI_Allgatherv(
        returns + start_idx,  // Send buffer (local portion)
        local_count,          // Send count
        MPI_FLOAT,            
        gathered_returns,     // Receive buffer
        (int[]){block_size, block_size, block_size, block_size},  // Receive counts for each process
        (int[]){0, block_size, block_size * 2, block_size * 3},   // Displacements
        MPI_FLOAT, 
        MPI_COMM_WORLD
    );

    // Copy gathered results back to returns
    memcpy(returns, gathered_returns, total_elements * sizeof(float));

    // Free temporary buffer
    free(gathered_returns);
}
void calculateAverages(float* returns, float* averages, int size, int rank, float* local_sum, float* global_sum){
    int block_size = ((TITLES + INDEX) * (DAYS - 1)) / size;
    int remainder = ((TITLES + INDEX) * (DAYS - 1)) % size;
    int start_idx = rank * block_size;
    int end_idx = (rank < size - 1) ? start_idx + block_size : start_idx + block_size + remainder;
    for (int i = start_idx; i < end_idx; i++) {
        int row = i / (DAYS - 1);
        local_sum[row] += returns[i];
    }
    MPI_Allreduce(local_sum, global_sum, TITLES + INDEX, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    for (int i = 0; i < (TITLES + INDEX); i++) {
        averages[i] = global_sum[i] / (float)(DAYS - 1);
    }
}
void calculateVariances(float* returns, float* averages, float* variances, float* stdDevs, int size, int rank, float* local_var, float* global_var)  {
    int block_size = ((TITLES + INDEX) * (DAYS - 1)) / size;
    int remainder = ((TITLES + INDEX) * (DAYS - 1)) % size;
    int start_idx = rank * block_size;
    int end_idx = (rank < size - 1) ? start_idx + block_size : start_idx + block_size + remainder;

    for (int i = start_idx; i < end_idx; i++) {
        int row = i / (DAYS - 1);
        local_var[row] += ((returns[i] - averages[row])*(returns[i] - averages[row]));
    }

    MPI_Allreduce(local_var, global_var, TITLES + INDEX, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    for (int row = 0; row < TITLES + INDEX; row++) {
        variances[row] = global_var[row] / (float)(DAYS - 1);
        stdDevs[row] = sqrtf(variances[row]);
    }
}
void calculateCovariances(float* returns, float* averages, float** covariances, int size, int rank, float (*local_cov)[INDEX], float (*global_cov)[INDEX]){
    int block_size = ((INDEX) * (DAYS - 1)) / size;
    int remainder = ((INDEX) * (DAYS - 1)) % size;
    int start_idx = rank * block_size;
     int end_idx = (rank < size - 1) ?  start_idx + block_size :  start_idx + block_size + remainder;
      for (int i = start_idx; i < end_idx; i++) {
        int index_col = i / (DAYS - 1);    // Determina quale indice
        int day = i % (DAYS - 1);          // Determina il giorno
        for (int title = 0; title < TITLES; title++) {
            local_cov[title][index_col] += 
                (returns[(title * (DAYS - 1)) + day] - averages[title]) *
                (returns[((TITLES + index_col) * (DAYS - 1)) + day] - averages[TITLES + index_col]);
        }
    }

    // Combina i risultati da tutti i processi
    MPI_Allreduce(local_cov, global_cov, TITLES * INDEX, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    for (int i = 0; i < TITLES; i++) {
        for (int j = 0; j < INDEX; j++) {
            covariances[i][j] = global_cov[i][j] / (float)(DAYS - 1);
        }
    }
}
void calculateBetasAndCorrelations(float** covariances, float* variances, float* stdDevs, float** betas, float** correlations, int size, int rank) {
    int block_size = (TITLES * INDEX) / size;
    int remainder = (TITLES * INDEX) % size;
    int start_idx = rank * block_size;
    int end_idx = (rank < size - 1) ? start_idx + block_size : start_idx + block_size + remainder;

    for (int i = start_idx; i < end_idx; i++) {
        int title = i / INDEX;
        int idx = i % INDEX;

        if (variances[TITLES + idx] != 0) {
            betas[title][idx] = covariances[title][idx] / variances[TITLES + idx];
            correlations[title][idx] = covariances[title][idx] / (stdDevs[TITLES + idx] * stdDevs[title]);
        }
    }
}

void cleanupData(FinancialData* data) {
    free(data->prices);
    free(data->returns);
    free(data->averages);
    free(data->variances);
    free(data->stdDevs);
    
    for (int i = 0; i < TITLES; i++) {
        free(data->covariances[i]);
        free(data->betas[i]);
        free(data->correlations[i]);
    }
    
    free(data->covariances);
    free(data->betas);
    free(data->correlations);
}

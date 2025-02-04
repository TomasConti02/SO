#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>

#define DAYS 10
#define TITLES 2
#define INDEX 2

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

void initializeData(float* prices) {
    srand(time(NULL));
    for (int i = 0; i < TITLES + INDEX; i++) {
        for (int j = 0; j < DAYS; j++) {
            // Base price tra 50 e 200 con variazione casuale
            float base_price = 50 + (rand() % 151);
            prices[i * DAYS + j] = base_price + ((float)rand() / RAND_MAX);
        }
    }
}
/*
void initializeData(float* prices) {
    float valori_titolo_0[] = {100, 110, 120, 110, 120, 130, 120, 130, 140, 130};
    float valori_titolo_1[] = {100, 90, 80, 90, 100, 90, 80, 90, 100, 90};
    float valori_indice_0[] = {100, 120, 140, 160, 180, 200, 220, 240, 260, 280};
    float valori_indice_1[] = {100, 95, 90, 85, 80, 75, 70, 65, 60, 55};

    for (int j = 0; j < DAYS; j++) {
        prices[0 * DAYS + j] = valori_titolo_0[j];
        prices[1 * DAYS + j] = valori_titolo_1[j];
        prices[2 * DAYS + j] = valori_indice_0[j];
        prices[3 * DAYS + j] = valori_indice_1[j];
    }
}*/

void calculateReturns(float* prices, float* returns, int size, int rank) {
    int total_elements = (TITLES + INDEX) * (DAYS - 1);
    int block_size = total_elements / size;
    int remainder = total_elements % size;
    int start_idx = rank * block_size;
    int end_idx = (rank < size - 1) ? start_idx + block_size : start_idx + block_size + remainder;

    for (int i = start_idx; i < end_idx; i++) {
        int row = i / (DAYS - 1);
        int day = i % (DAYS - 1);
        returns[i] = (prices[row * DAYS + day + 1] - prices[row * DAYS + day]) / prices[row * DAYS + day];
    }
    MPI_Allgather(MPI_IN_PLACE, end_idx - start_idx, MPI_FLOAT, returns, block_size, MPI_FLOAT, MPI_COMM_WORLD);
}

void calculateAverages(float* returns, float* averages, int size, int rank, float* local_sum, float* global_sum) {
    int total_elements = (TITLES + INDEX) * (DAYS - 1);
    int block_size = total_elements / size;
    int remainder = total_elements % size;
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

void calculateVariances(float* returns, float* averages, float* variances, float* stdDevs, int size, int rank, float* local_var, float* global_var) {
    int total_elements = (TITLES + INDEX) * (DAYS - 1);
    int block_size = total_elements / size;
    int remainder = total_elements % size;
    int start_idx = rank * block_size;
    int end_idx = (rank < size - 1) ? start_idx + block_size : start_idx + block_size + remainder;

    for (int i = start_idx; i < end_idx; i++) {
        int row = i / (DAYS - 1);
        local_var[row] += ((returns[i] - averages[row]) * (returns[i] - averages[row]));
    }

    MPI_Allreduce(local_var, global_var, TITLES + INDEX, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    for (int row = 0; row < TITLES + INDEX; row++) {
        variances[row] = global_var[row] / (float)(DAYS - 1);
        stdDevs[row] = sqrtf(variances[row]);
    }
}

void calculateCovariances(float* returns, float* averages, float** covariances, int size, int rank, float* local_cov, float* global_cov) {
    int total_elements = INDEX * (DAYS - 1);
    int block_size = total_elements / size;
    int remainder = total_elements % size;
    int start_idx = rank * block_size;
    int end_idx = (rank < size - 1) ? start_idx + block_size : start_idx + block_size + remainder;

    for (int i = start_idx; i < end_idx; i++) {
        int index_col = i / (DAYS - 1);
        int day = i % (DAYS - 1);

        for (int title = 0; title < TITLES; title++) {
            local_cov[title * INDEX + index_col] +=
                (returns[(title * (DAYS - 1)) + day] - averages[title]) *
                (returns[((TITLES + index_col) * (DAYS - 1)) + day] - averages[TITLES + index_col]);
        }
    }

    MPI_Allreduce(local_cov, global_cov, TITLES * INDEX, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    for (int i = 0; i < TITLES; i++) {
        for (int j = 0; j < INDEX; j++) {
            covariances[i][j] = global_cov[i * INDEX + j] / (float)(DAYS - 1);
        }
    }
}

void calculateBetasAndCorrelations(float** covariances, float* variances, float* stdDevs, float** betas, float** correlations, int size, int rank, float* beta_local, float* correlation_local, float* beta_global, float* correlation_global) {
    int total_elements = TITLES * INDEX;
    int block_size = total_elements / size;
    int remainder = total_elements % size;
    int start_idx = rank * block_size;
    int end_idx = (rank < size - 1) ? start_idx + block_size : start_idx + block_size + remainder;

    for (int i = start_idx; i < end_idx; i++) {
        int title = i / INDEX;
        int idx = i % INDEX;

        if (variances[TITLES + idx] != 0) {
            beta_local[title * INDEX + idx] = covariances[title][idx] / variances[TITLES + idx];
            correlation_local[title * INDEX + idx] = covariances[title][idx] / (stdDevs[TITLES + idx] * stdDevs[title]);
        }
    }

    MPI_Allreduce(beta_local, beta_global, TITLES * INDEX, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(correlation_local, correlation_global, TITLES * INDEX, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    for (int i = 0; i < TITLES; i++) {
        for (int j = 0; j < INDEX; j++) {
            betas[i][j] = beta_global[i * INDEX + j];
            correlations[i][j] = correlation_global[i * INDEX + j];
        }
    }
}
void printResults(FinancialData* data, int rank) {
    if (rank == 0) {
        printf("\n=== ANALISI DEI RENDIMENTI FINANZIARI ===\n");

        for (int i = 0; i < TITLES + INDEX; i++) {
            printf("\n%s %d:\n", (i < TITLES) ? "TITOLO" : "INDICE", i);
            printf("Media: %f\n", data->averages[i]);
            printf("Deviazione Standard: %.4f%%\n", data->stdDevs[i] * 100);
            printf("Varianza: %f\n", data->variances[i]);
        }

        printf("\n=== COVARIANZE ===\n");
        for (int i = 0; i < TITLES; i++) {
            for (int j = 0; j < INDEX; j++) {
                printf("Covarianza Titolo %d con Indice %d: %.4f\n", i, j, data->covariances[i][j]);
            }
        }

        printf("\n=== BETA E CORRELAZIONI ===\n");
        for (int i = 0; i < TITLES; i++) {
            printf("\nTitolo %d:\n", i);
            for (int j = 0; j < INDEX; j++) {
                printf("Con Indice %d - Beta: %.4f, Correlazione: %.4f\n", j, data->betas[i][j], data->correlations[i][j]);
            }
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

int main(int argc, char* argv[]) {
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double begin, end, global_elaps, local_elaps;

    FinancialData data = {
        .prices = (float*)malloc((TITLES + INDEX) * DAYS * sizeof(float)),
        .returns = (float*)malloc((TITLES + INDEX) * (DAYS - 1) * sizeof(float)),
        .averages = (float*)malloc((TITLES + INDEX) * sizeof(float)),
        .variances = (float*)malloc((TITLES + INDEX) * sizeof(float)),
        .stdDevs = (float*)malloc((TITLES + INDEX) * sizeof(float)),
        .covariances = (float**)malloc(TITLES * sizeof(float*)),
        .betas = (float**)malloc(TITLES * sizeof(float*)),
        .correlations = (float**)malloc(TITLES * sizeof(float*))
    };

    for (int i = 0; i < TITLES; i++) {
        data.covariances[i] = (float*)malloc(INDEX * sizeof(float));
        data.betas[i] = (float*)malloc(INDEX * sizeof(float));
        data.correlations[i] = (float*)malloc(INDEX * sizeof(float));
    }

    float* local_var = (float*)malloc((TITLES + INDEX) * sizeof(float));
    float* global_var = (float*)malloc((TITLES + INDEX) * sizeof(float));
    float* local_cov = (float*)malloc(TITLES * INDEX * sizeof(float)); // Changed to 1D
    float* global_cov = (float*)malloc(TITLES * INDEX * sizeof(float)); // Changed to 1D
    float* beta_local = (float*)malloc(TITLES * INDEX * sizeof(float));   // Changed to 1D
    float* correlation_local = (float*)malloc(TITLES * INDEX * sizeof(float)); // Changed to 1D
    float* beta_global = (float*)malloc(TITLES * INDEX * sizeof(float));  // Changed to 1D
    float* correlation_global = (float*)malloc(TITLES * INDEX * sizeof(float)); // Changed to 1D
    float* local_sum = (float*)malloc((TITLES + INDEX) * sizeof(float));
    float* global_sum = (float*)malloc((TITLES + INDEX) * sizeof(float));

    if (!local_var || !global_var || !local_cov || !global_cov ||
        !beta_local || !correlation_local || !beta_global || !correlation_global ||
        !local_sum || !global_sum || !data.prices || !data.returns || !data.averages ||
        !data.variances || !data.stdDevs) {
        fprintf(stderr, "Memory allocation failed!\n");
        MPI_Finalize();
        return 1;
    }

    for (int i = 0; i < TITLES; i++) {
        if (!data.covariances[i] || !data.betas[i] || !data.correlations[i]) {
            fprintf(stderr, "Memory allocation failed for matrix!\n");
            MPI_Finalize();
            return 1;
        }
    }

    for (int i = 0; i < TITLES + INDEX; i++) {
        local_var[i] = 0;
        global_var[i] = 0;
        local_sum[i] = 0;
        global_sum[i] = 0;
    }

    for (int i = 0; i < TITLES * INDEX; i++) { // Initialize 1D arrays
        local_cov[i] = 0;
        global_cov[i] = 0;
        beta_local[i] = 0;
        correlation_local[i] = 0;
        beta_global[i] = 0;
        correlation_global[i] = 0;
    }

    if (rank == 0) {
        initializeData(data.prices);
    }

    MPI_Bcast(data.prices, (TITLES + INDEX) * DAYS, MPI_FLOAT, 0, MPI_COMM_WORLD);

    begin = MPI_Wtime();

    calculateReturns(data.prices, data.returns, size, rank);
    calculateAverages(data.returns, data.averages, size, rank, local_sum, global_sum);
    calculateVariances(data.returns, data.averages, data.variances, data.stdDevs, size, rank, local_var, global_var);
    calculateCovariances(data.returns, data.averages, data.covariances, size, rank, local_cov, global_cov);
    calculateBetasAndCorrelations(data.covariances, data.variances, data.stdDevs, data.betas, data.correlations, size, rank, beta_local, correlation_local, beta_global, correlation_global);

    printResults(&data, rank);

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

    cleanupData(&data);

    free(local_cov);
    free(global_cov);
    free(beta_local);
    free(correlation_local);
    free(beta_global);
    free(correlation_global);
    free(local_var);
    free(global_var);
    free(local_sum);
    free(global_sum);

    MPI_Finalize();
    return 0;
}

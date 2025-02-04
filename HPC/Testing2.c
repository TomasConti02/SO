#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>

#define DAYS 365*15
#define TITLES 500
#define INDEX 500

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

void cleanupFinancialData(FinancialData* data) {
    if (data->prices) free(data->prices);
    if (data->returns) free(data->returns);
    if (data->averages) free(data->averages);
    if (data->variances) free(data->variances);
    if (data->stdDevs) free(data->stdDevs);

    if (data->covariances) {
        for (int i = 0; i < TITLES; i++) {
            if (data->covariances[i]) free(data->covariances[i]);
        }
        free(data->covariances);
    }

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

void calculateReturns(float* prices, float* returns, int size, int rank) {
    int block_size = (TITLES + INDEX) / size;
    int remainder = (TITLES + INDEX) % size;

    int start_idx = rank * block_size + (rank < remainder ? rank : remainder);
    int end_idx = start_idx + block_size + (rank < remainder ? 1 : 0);

    for (int row = start_idx; row < end_idx; row++) {
        for (int day = 0; day < (DAYS - 1); day++) {
            returns[row * (DAYS - 1) + day] = (prices[row * DAYS + day + 1] - prices[row * DAYS + day]) / prices[row * DAYS + day];
        }
    }
}

void calculateAverages(float* returns, float* averages, int size, int rank) {
    int block_size = (TITLES + INDEX) / size;
    int remainder = (TITLES + INDEX) % size;
    int start_idx = rank * block_size + (rank < remainder ? rank : remainder);
    int end_idx = start_idx + block_size + (rank < remainder ? 1 : 0);

    for (int row = start_idx; row < end_idx; row++) {
        averages[row] = 0.0f;
        for (int day = 0; day < (DAYS - 1); day++) {
            averages[row] += returns[row * (DAYS - 1) + day];
        }
        averages[row] /= (DAYS - 1);
    }
}

void calculateVariances(float* returns, float* averages, float* variances, float* stdDevs, int size, int rank) {
    int block_size = (TITLES + INDEX) / size;
    int remainder = (TITLES + INDEX) % size;

    int start_idx = rank * block_size + (rank < remainder ? rank : remainder);
    int end_idx = start_idx + block_size + (rank < remainder ? 1 : 0);

    for (int row = start_idx; row < end_idx; row++) {
        variances[row] = 0;
        for (int day = 0; day < (DAYS - 1); day++) {
            variances[row] += ((returns[row * (DAYS - 1) + day] - averages[row]) * (returns[row * (DAYS - 1) + day] - averages[row]));
        }
        variances[row] = variances[row] / (float)(DAYS - 1);
        stdDevs[row] = sqrtf(variances[row]);
    }
}

void calculateCovariances(float* returns, float* averages, float** covariances, float* local_cov, float* global_cov, int size, int rank) {
    int block_size = (DAYS - 1) / size;
    int remainder = (DAYS - 1) % size;

    int start_idx = rank * block_size + (rank < remainder ? rank : remainder);
    int end_idx = start_idx + block_size + (rank < remainder ? 1 : 0);

    for (int i = 0; i < TITLES * INDEX; i++) {
        local_cov[i] = 0;
    }

    for (int day = start_idx; day < end_idx; day++) {
        for (int title = 0; title < TITLES; title++) {
            for (int idx = 0; idx < INDEX; idx++) {
                local_cov[title * INDEX + idx] += 
                    (returns[title * (DAYS - 1) + day] - averages[title]) *
                    (returns[(TITLES + idx) * (DAYS - 1) + day] - averages[TITLES + idx]);
            }
        }
    }

    MPI_Allreduce(local_cov, global_cov, TITLES * INDEX, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    for (int i = 0; i < TITLES; i++) {
        for (int j = 0; j < INDEX; j++) {
            covariances[i][j] = global_cov[i * INDEX + j] / (float)(DAYS - 1);
        }
    }
}

void calculateBetasAndCorrelations(float** covariances, float* variances, float* stdDevs, float** betas, float** correlations, float* beta_local, float* correlation_local, int size, int rank) {
    int block_size = (TITLES * INDEX) / size;
    int remainder = (TITLES * INDEX) % size;
    int start_idx = rank * block_size;
    int end_idx = (rank < size - 1) ? start_idx + block_size : start_idx + block_size + remainder;

    for (int i = start_idx; i < end_idx; i++) {
        int title = i / INDEX;
        int idx = i % INDEX;
        
        if (variances[TITLES + idx] != 0) {
            beta_local[i] = covariances[title][idx] / variances[TITLES + idx];
            correlation_local[i] = covariances[title][idx] / (stdDevs[TITLES + idx] * stdDevs[title]);
        }
    }

    for (int i = start_idx; i < end_idx; i++) {
        int title = i / INDEX;
        int idx = i % INDEX;
        
        betas[title][idx] = beta_local[i];
        correlations[title][idx] = correlation_local[i];
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
    }
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

    float* local_cov = (float*)malloc(TITLES * INDEX * sizeof(float));
    float* global_cov = (float*)malloc(TITLES * INDEX * sizeof(float));
    float* beta_local = (float*)malloc(TITLES * INDEX * sizeof(float));
    float* correlation_local = (float*)malloc(TITLES * INDEX * sizeof(float));

    if (!data.prices || !data.returns || !data.averages || !data.variances || !data.stdDevs || 
        !data.covariances || !data.betas || !data.correlations || !local_cov || !global_cov || 
        !beta_local || !correlation_local) {
        fprintf(stderr, "Memory allocation failed\n");
        cleanupFinancialData(&data);
        free(local_cov);
        free(global_cov);
        free(beta_local);
        free(correlation_local);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    for (int i = 0; i < TITLES; i++) {
        data.covariances[i] = (float*)malloc(INDEX * sizeof(float));
        data.betas[i] = (float*)malloc(INDEX * sizeof(float));
        data.correlations[i] = (float*)malloc(INDEX * sizeof(float));

        if (!data.covariances[i] || !data.betas[i] || !data.correlations[i]) {
            fprintf(stderr, "Memory allocation for 2D arrays failed\n");
            cleanupFinancialData(&data);
            free(local_cov);
            free(global_cov);
            free(beta_local);
            free(correlation_local);
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }

    if (rank == 0) {
        initializeData(data.prices);
    }

    MPI_Bcast(data.prices, (TITLES + INDEX) * DAYS, MPI_FLOAT, 0, MPI_COMM_WORLD);
    begin = MPI_Wtime();

    calculateReturns(data.prices, data.returns, size, rank);
    calculateAverages(data.returns, data.averages, size, rank);
    calculateVariances(data.returns, data.averages, data.variances, data.stdDevs, size, rank);

    int total_rows = TITLES + INDEX;
    int block_size = total_rows / size;
    int remainder = total_rows % size;
    int my_count = block_size + (rank < remainder ? 1 : 0);

    int recvcounts[size];
    int displs[size];

    MPI_Allgather(&my_count, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);

    displs[0] = 0;
    for (int i = 1; i < size; i++) {
        displs[i] = displs[i - 1] + recvcounts[i - 1];
    }

    int send_offset = (rank * block_size + (rank < remainder ? rank : remainder));

    int* recvcounts_returns = (int*)malloc(size * sizeof(int));
    int* displs_returns = (int*)malloc(size * sizeof(int));

    for (int i = 0; i < size; i++) {
        recvcounts_returns[i] = recvcounts[i] * (DAYS - 1);
        displs_returns[i] = displs[i] * (DAYS - 1);
    }

    MPI_Allgatherv(data.returns + send_offset * (DAYS - 1), my_count * (DAYS - 1), MPI_FLOAT, data.returns, recvcounts_returns, displs_returns, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgatherv(data.averages + send_offset, my_count, MPI_FLOAT, data.averages, recvcounts, displs, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgatherv(data.variances + send_offset, my_count, MPI_FLOAT, data.variances, recvcounts, displs, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgatherv(data.stdDevs + send_offset, my_count, MPI_FLOAT, data.stdDevs, recvcounts, displs, MPI_FLOAT, MPI_COMM_WORLD);

    calculateCovariances(data.returns, data.averages, data.covariances, local_cov, global_cov, size, rank);
    calculateBetasAndCorrelations(data.covariances, data.variances, data.stdDevs, data.betas, data.correlations, beta_local, correlation_local, size, rank);

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

    //printResults(&data, rank);

    cleanupFinancialData(&data);
    free(local_cov);
    free(global_cov);
    free(beta_local);
    free(correlation_local);
    free(recvcounts_returns);
    free(displs_returns);

    MPI_Finalize();
    return 0;
}

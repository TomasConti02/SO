#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#define DAYS 5
#define TITLES 2
#define INDEX 2

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

void initializeData(float* prices) {
    float valori_titolo_1[] = {200, 150, 100, 100, 90};
    float valori_titolo_0[] = {100, 110, 130, 125, 140};
    float valori_indice_0[] = {100, 105, 115, 113, 120};
    float valori_indice_1[] = {50, 90, 100, 120, 150};

    int num_valori = sizeof(valori_titolo_0) / sizeof(valori_titolo_0[0]);

    for (int j = 0; j < DAYS; j++) {
        prices[0 * DAYS + j] = valori_titolo_0[j % num_valori];
        prices[1 * DAYS + j] = valori_titolo_1[j % num_valori];
        prices[2 * DAYS + j] = valori_indice_0[j % num_valori];
        prices[3 * DAYS + j] = valori_indice_1[j % num_valori];
    }
}

void calculateReturns(float* prices, float* returns, int local_size, int rank) {
    for (int row = 0; row < local_size; row++) {
        for (int day = 0; day < (DAYS - 1); day++) {
            returns[row * (DAYS - 1) + day] = (prices[row * DAYS + day + 1] - prices[row * DAYS + day]) / prices[row * DAYS + day];
        }
    }
}

void calculateAverages(float* returns, float* averages, int local_size, int rank) {
    for (int row = 0; row < local_size; row++) {
        averages[row] = 0.0f;
        for (int day = 0; day < (DAYS - 1); day++) {
            averages[row] += returns[row * (DAYS - 1) + day];
        }
        averages[row] /= (DAYS - 1);
    }
}

void calculateVariances(float* returns, float* averages, float* variances, float* stdDevs, int local_size, int rank) {
    for (int row = 0; row < local_size; row++) {
        variances[row] = 0;
        for (int day = 0; day < (DAYS - 1); day++) {
            float diff = returns[row * (DAYS - 1) + day] - averages[row];
            variances[row] += diff * diff;
        }
        variances[row] /= (float)(DAYS - 1);
    }
}

void calculateCovariances(float* returns, float* averages, float* covariances, int size, int rank, int* recvcounts, int* displs, float* local_cov) {
    int index_block_size = INDEX / size;
    int index_remainder = INDEX % size;
    int index_start_idx = rank * index_block_size + (rank < index_remainder ? rank : index_remainder);
    int index_end_idx = index_start_idx + index_block_size + (rank < index_remainder ? 1 : 0);
    int local_index_size = index_end_idx - index_start_idx;

    for (int i = 0; i < local_index_size; i++) {
        for (int t = 0; t < TITLES; t++) {
            float sum = 0.0f;
            for (int d = 0; d < DAYS - 1; d++) {
                sum += (returns[t * (DAYS - 1) + d] - averages[t]) * (returns[(TITLES + index_start_idx + i) * (DAYS - 1) + d] - averages[TITLES + index_start_idx + i]);
            }
            local_cov[i * TITLES + t] = sum / (DAYS - 1);
        }
    }
    MPI_Allgatherv(local_cov, local_index_size * TITLES, MPI_FLOAT, covariances, recvcounts, displs, MPI_FLOAT, MPI_COMM_WORLD);
}

void calculateBetasAndCorrelations(float* covariances, float* variances, float** betas, float** correlations, int size, int rank) {
    int total_combinations = TITLES * INDEX;
    int block_size = total_combinations / size;
    int remainder = total_combinations % size;
    int start_idx = rank * block_size + (rank < remainder ? rank : remainder);
    int end_idx = start_idx + block_size + (rank < remainder ? 1 : 0);
    const float epsilon = 1e-7;

    for (int i = start_idx; i < end_idx; i++) {
        int title = i / INDEX;
        int idx = i % INDEX;

        if (fabsf(variances[TITLES + idx]) > epsilon) {
            betas[title][idx] = covariances[title + idx * TITLES] / variances[TITLES + idx];
        } else {
            betas[title][idx] = 0.0f;
        }

        if (fabsf(variances[TITLES + idx]) > epsilon && fabsf(variances[title]) > epsilon) {
            correlations[title][idx] = covariances[title + idx * TITLES] / (sqrtf(variances[TITLES + idx]) * sqrtf(variances[title]));
        } else {
            correlations[title][idx] = 0.0f;
        }
    }
}

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

int main(int argc, char* argv[]) {
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double begin, end, global_elaps, local_elaps;

    FinancialData data = {0};

    data.prices = (float*)malloc((TITLES + INDEX) * DAYS * sizeof(float));
    data.returns = (float*)malloc((TITLES + INDEX) * (DAYS - 1) * sizeof(float));
    data.averages = (float*)malloc((TITLES + INDEX) * sizeof(float));
    data.variances = (float*)malloc((TITLES + INDEX) * sizeof(float));
    data.stdDevs = (float*)malloc((TITLES + INDEX) * sizeof(float));
    data.covariances = (float*)malloc(TITLES * INDEX * sizeof(float));
    data.betas = (float**)malloc(TITLES * sizeof(float*));
    data.correlations = (float**)malloc(TITLES * sizeof(float*));

    if (!data.prices || !data.returns || !data.averages || !data.variances || !data.stdDevs ||
        !data.covariances || !data.betas || !data.correlations) {
        fprintf(stderr, "Memory allocation failed\n");
        cleanupFinancialData(&data);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

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

    int* recvcounts = (int*)malloc(size * sizeof(int));
    int* displs = (int*)malloc(size * sizeof(int));

    for (int i = 0; i < size; i++) {
        int i_block_size = INDEX / size;
        int i_remainder = INDEX % size;
        int i_start = i * i_block_size + (i < i_remainder ? i : i_remainder);
        int i_end = i_start + i_block_size + (i < i_remainder ? 1 : 0);
        int l_i_size = i_end - i_start;

        recvcounts[i] = l_i_size * TITLES;
        displs[i] = (i > 0) ? displs[i - 1] + recvcounts[i - 1] : 0;
    }

    int block_size = (int)ceil((double)(TITLES + INDEX) / size);
    int start_idx = rank * block_size;
    int end_idx = (rank + 1) * block_size;
    if (end_idx > TITLES + INDEX) end_idx = TITLES + INDEX;
    int local_size = end_idx - start_idx;

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

    for (int i = 0; i < TITLES * INDEX; i++) {
        data.covariances[i] = 0.0f;
    }
    for (int i = 0; i < (TITLES * local_size); i++) {
        local_cov[i] = 0.0f;
    }

    if (rank == 0) {
        initializeData(data.prices);
    }

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

    MPI_Bcast(data.prices, (TITLES + INDEX) * DAYS, MPI_FLOAT, 0, MPI_COMM_WORLD);
    begin = MPI_Wtime();

    calculateReturns(data.prices + start_idx * DAYS, localReturns, local_size, rank);
    calculateAverages(localReturns, localAverage, local_size, rank);
    calculateVariances(localReturns, localAverage, localVariance, localDev, local_size, rank);

    MPI_Allgatherv(localReturns, local_size * (DAYS - 1), MPI_FLOAT, data.returns, sendcounts_returns, displs_returns, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgatherv(localAverage, local_size, MPI_FLOAT, data.averages, sendcounts_other, displs_other, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgatherv(localVariance, local_size, MPI_FLOAT, data.variances, sendcounts_other, displs_other, MPI_FLOAT, MPI_COMM_WORLD);

    calculateCovariances(data.returns, data.averages, data.covariances, size, rank, recvcounts, displs, local_cov);
    calculateBetasAndCorrelations(data.covariances, data.variances, data.betas, data.correlations, size, rank);

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

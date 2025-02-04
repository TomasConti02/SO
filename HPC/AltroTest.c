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
void initializeData(float* prices) {
    float valori_titolo_0[] = {100, 110, 120, 110, 120, 130, 120, 130, 140, 130};
    float valori_titolo_1[] = {100, 90, 80, 90, 100, 90, 80, 90, 100, 90};
    float valori_indice_0[] = {100, 120, 140, 160, 180, 200, 220, 240, 260, 280};
    float valori_indice_1[] = {100, 95, 90, 85, 80, 75, 70, 65, 60, 55};

    int num_valori = sizeof(valori_titolo_0) / sizeof(valori_titolo_0[0]); // Get the number of elements

    for (int j = 0; j < DAYS; j++) {
        prices[0 * DAYS + j] = valori_titolo_0[j % num_valori];  // Use modulo operator
        prices[1 * DAYS + j] = valori_titolo_1[j % num_valori];  // Use modulo operator
        prices[2 * DAYS + j] = valori_indice_0[j % num_valori];  // Use modulo operator
        prices[3 * DAYS + j] = valori_indice_1[j % num_valori];  // Use modulo operator
    }
}
void calculateReturns(float* prices, float* returns, int local_size) {
    for (int row = 0; row < local_size; row++) {
        for (int day = 0; day < (DAYS - 1); day++) {
            returns[row * (DAYS - 1) + day] = (prices[row * DAYS + day + 1] - prices[row * DAYS + day]) / prices[row * DAYS + day];
        }
    }
}

void calculateAverages(float* returns, float* averages, int local_size) {
    for (int row = 0; row < local_size; row++) {
        averages[row] = 0.0f;
        for (int day = 0; day < (DAYS - 1); day++) {
            averages[row] += returns[row * (DAYS - 1) + day];
        }
        averages[row] /= (DAYS - 1);
    }
}

void calculateVariances(float* returns, float* averages, float* variances, float* stdDevs, int local_size) {
    for (int row = 0; row < local_size; row++) {
        variances[row] = 0;
        for (int day = 0; day < (DAYS - 1); day++) {
            variances[row] += ((returns[row * (DAYS - 1) + day] - averages[row]) * (returns[row * (DAYS - 1) + day] - averages[row]));
        }
        variances[row] = variances[row] / (float)(DAYS - 1);
        stdDevs[row] = sqrtf(variances[row]);
    }
}

void calculateCovariances(float* returns, float* averages, float** covariances, int size, int rank, float* local_cov, float* global_cov) {
    int block_size = (DAYS - 1) / size;
    int remainder = (DAYS - 1) % size;

    int start_idx = rank * block_size + (rank < remainder ? rank : remainder);
    int end_idx = start_idx + block_size + (rank < remainder ? 1 : 0);

    // Inizializza le covarianze locali a zero
    for (int i = 0; i < TITLES * INDEX; i++) {
        local_cov[i] = 0.0f;
    }

    // Calcola le covarianze locali
    for (int day = start_idx; day < end_idx; day++) {
        for (int title = 0; title < TITLES; title++) {
            for (int idx = 0; idx < INDEX; idx++) {
                int cov_index = title * INDEX + idx;
                local_cov[cov_index] += 
                    (returns[title * (DAYS - 1) + day] - averages[title]) *
                    (returns[(TITLES + idx) * (DAYS - 1) + day] - averages[TITLES + idx]);
            }
        }
    }

    // Riduci le covarianze locali in global_cov
    MPI_Allreduce(local_cov, global_cov, TITLES * INDEX, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    // Calcola le covarianze finali
    for (int i = 0; i < TITLES; i++) {
        for (int j = 0; j < INDEX; j++) {
            covariances[i][j] = global_cov[i * INDEX + j] / (float)(DAYS - 1);
        }
    }
}

void calculateBetasAndCorrelations(float** covariances, float* variances, float* stdDevs, float** betas, float** correlations, int size, int rank) {
    int block_size = (TITLES * INDEX) / size;
    int remainder = (TITLES * INDEX) % size;
    int start_idx = rank * block_size;
    int end_idx = (rank < size - 1) ? start_idx + block_size : start_idx + block_size + remainder;

    // Define a small margin of error (epsilon)
    const float epsilon = 1e-7;

    // Calculate betas and correlations for this process's block
    for (int i = start_idx; i < end_idx; i++) {
        int title = i / INDEX;
        int idx = i % INDEX;

        // Check for near-zero variance before calculating beta
        if (fabsf(variances[TITLES + idx]) > epsilon) {
            betas[title][idx] = covariances[title][idx] / variances[TITLES + idx];
        } else {
            betas[title][idx] = 0.0f; // Default value if variance is near-zero
        }

        // Check for near-zero standard deviations before calculating correlation
        if (fabsf(stdDevs[TITLES + idx]) > epsilon && fabsf(stdDevs[title]) > epsilon) {
            correlations[title][idx] = covariances[title][idx] / (stdDevs[TITLES + idx] * stdDevs[title]);
        } else {
            correlations[title][idx] = 0.0f; // Default value if standard deviation is near-zero
        }

        // Print local results for this process
        //printf("Rank %d: Beta[%d][%d] = %f, Correlation[%d][%d] = %f\n", rank, title, idx, betas[title][idx], title, idx, correlations[title][idx]);
    }
}
void cleanupFinancialData(FinancialData* data) {
    if (data) {
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
    data.covariances = (float**)malloc(TITLES * sizeof(float*));
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
        data.covariances[i] = (float*)malloc(INDEX * sizeof(float));
        data.betas[i] = (float*)malloc(INDEX * sizeof(float));
        data.correlations[i] = (float*)malloc(INDEX * sizeof(float));

        if (!data.covariances[i] || !data.betas[i] || !data.correlations[i]) {
            fprintf(stderr, "Memory allocation for 2D arrays failed\n");
            cleanupFinancialData(&data);
            MPI_Finalize();
            return EXIT_FAILURE;
        }

        // Inizializza beta e correlazioni a zero
        for (int j = 0; j < INDEX; j++) {
            data.betas[i][j] = 0.0f;
            data.correlations[i][j] = 0.0f;
        }
    }

    int block_size = (TITLES + INDEX) / size;
    int remainder = (TITLES + INDEX) % size;

    int start_idx = rank * block_size + (rank < remainder ? rank : remainder);
    int end_idx = start_idx + block_size + (rank < remainder ? 1 : 0);
    int local_size = end_idx - start_idx;

    float* localReturns = (float*)malloc(local_size * (DAYS - 1) * sizeof(float));
    float* localAverage = (float*)malloc(local_size * sizeof(float));
    float* localVariance = (float*)malloc(local_size * sizeof(float));
    float* localDev = (float*)malloc(local_size * sizeof(float));
    float* local_cov = (float*)malloc(TITLES * INDEX * sizeof(float));
    float* global_cov = (float*)malloc(TITLES * INDEX * sizeof(float));

    if (!localReturns || !localAverage || !localVariance || !localDev || !local_cov || !global_cov) {
        fprintf(stderr, "Memory allocation for local arrays failed\n");
        cleanupFinancialData(&data);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (rank == 0) {
        initializeData(data.prices);
    }

    MPI_Bcast(data.prices, (TITLES + INDEX) * DAYS, MPI_FLOAT, 0, MPI_COMM_WORLD);

    begin = MPI_Wtime();

    calculateReturns(data.prices + start_idx * DAYS, localReturns, local_size);
    calculateAverages(localReturns, localAverage, local_size);
    calculateVariances(localReturns, localAverage, localVariance, localDev, local_size);

    // Assemblare e distribuire i returns a tutti i processi
    MPI_Allgather(localReturns, local_size * (DAYS - 1), MPI_FLOAT, data.returns, local_size * (DAYS - 1), MPI_FLOAT, MPI_COMM_WORLD);

    // Assemblare e distribuire le medie a tutti i processi
    MPI_Allgather(localAverage, local_size, MPI_FLOAT, data.averages, local_size, MPI_FLOAT, MPI_COMM_WORLD);

    // Assemblare e distribuire le varianze a tutti i processi
    MPI_Allgather(localVariance, local_size, MPI_FLOAT, data.variances, local_size, MPI_FLOAT, MPI_COMM_WORLD);

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
        /*
        // Stampa le varianze calcolate
        printf("Variances:\n");
        for (int i = 0; i < (TITLES + INDEX); i++) {
            printf("  Variance of asset/index %d: %f\n", i, data.variances[i]);
        }

        // Stampa le covarianze calcolate
        printf("Covariances:\n");
        for (int i = 0; i < TITLES; i++) {
            printf("  Covariance of title %d:\n", i);
            for (int j = 0; j < INDEX; j++) {
                printf("    With index %d: %f\n", j, data.covariances[i][j]);
            }
        }*/
    }

    free(localReturns);
    free(localAverage);
    free(localVariance);
    free(localDev);
    free(local_cov);
    free(global_cov);
    cleanupFinancialData(&data);

    MPI_Finalize();
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#define DAYS 10
#define TITLES 2
#define INDEX 2

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
void calculateAverages(float* returns, float* averages, int size, int rank);
void calculateVariances(float* returns, float* averages, float* variances, float* stdDevs, int size, int rank);
void calculateCovariances(float* returns, float* averages, float** covariances, int size, int rank);
void calculateBetasAndCorrelations(float** covariances, float* variances, float* stdDevs, float** betas, float** correlations, int size, int rank);
void printResults(FinancialData* data, int rank);
void cleanupData(FinancialData* data);

int main(int argc, char* argv[]) {
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Initialize financial data structure
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

    // Broadcast initial price data to all processes
    MPI_Bcast(data.prices, (TITLES + INDEX) * DAYS, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Calculate returns
    calculateReturns(data.prices, data.returns, size, rank);

    // Calculate averages
    calculateAverages(data.returns, data.averages, size, rank);

    // Calculate variances and standard deviations
    calculateVariances(data.returns, data.averages, data.variances, data.stdDevs, size, rank);

    // Calculate covariances
    calculateCovariances(data.returns, data.averages, data.covariances, size, rank);

    // Calculate betas and correlations
    calculateBetasAndCorrelations(data.covariances, data.variances, data.stdDevs, data.betas, data.correlations, size, rank);

    // Print results
    printResults(&data, rank);

    // Cleanup
    cleanupData(&data);
    MPI_Finalize();
    return 0;
}
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
} 
//forti correlazioniù
/*
void initializeData(float* prices) {
    // Indice 0: trend crescente regolare
    float valori_indice_0[] = {100, 110, 120, 130, 140, 150, 160, 170, 180, 190};
    
    // Titolo 0: forte correlazione positiva con Indice 0
    float valori_titolo_0[] = {100, 108, 118, 128, 138, 148, 158, 168, 178, 188};
    
    // Indice 1: trend con forti aumenti
    float valori_indice_1[] = {100, 120, 140, 160, 180, 200, 220, 240, 260, 280};
    
    // Titolo 1: trend esattamente opposto all'Indice 1
    float valori_titolo_1[] = {280, 260, 240, 220, 200, 180, 160, 140, 120, 100};

    for (int j = 0; j < DAYS; j++) {
        prices[0 * DAYS + j] = valori_titolo_0[j];
        prices[1 * DAYS + j] = valori_titolo_1[j];
        prices[2 * DAYS + j] = valori_indice_0[j];
        prices[3 * DAYS + j] = valori_indice_1[j];
    }
}
*/

void calculateReturns(float* prices, float* returns, int size, int rank) {
    int block_size = ((TITLES + INDEX) * (DAYS - 1)) / size;
    int remainder = ((TITLES + INDEX) * (DAYS - 1)) % size;
    int start_idx = rank * block_size;
    int end_idx = (rank < size - 1) ? start_idx + block_size : start_idx + block_size + remainder;

    for (int i = start_idx; i < end_idx; i++) {
        int row = i / (DAYS - 1);
        int day = i % (DAYS - 1);
        returns[i] = (prices[row * DAYS + day + 1] - prices[row * DAYS + day]) / prices[row * DAYS + day];
    }
    //block_size	Numero di elementi che ciascun processo riceve
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,  returns, block_size, MPI_FLOAT, MPI_COMM_WORLD);
}

void calculateAverages(float* returns, float* averages, int size, int rank) {
    float local_sum[TITLES + INDEX] = {0};
    float global_sum[TITLES + INDEX] = {0};
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

void calculateVariances(float* returns, float* averages, float* variances, float* stdDevs, int size, int rank) {
    float local_var[TITLES + INDEX] = {0};
    float global_var[TITLES + INDEX] = {0};
    int block_size = ((TITLES + INDEX) * (DAYS - 1)) / size;
    int remainder = ((TITLES + INDEX) * (DAYS - 1)) % size;
    int start_idx = rank * block_size;
    int end_idx = (rank < size - 1) ? start_idx + block_size : start_idx + block_size + remainder;

    for (int i = start_idx; i < end_idx; i++) {
        int row = i / (DAYS - 1);
        float diff = returns[i] - averages[row];
        local_var[row] += diff * diff;
    }

    MPI_Allreduce(local_var, global_var, TITLES + INDEX, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    for (int row = 0; row < TITLES + INDEX; row++) {
        variances[row] = global_var[row] / (float)(DAYS - 1);
        stdDevs[row] = sqrtf(variances[row]);
    }
}

void calculateCovariances(float* returns, float* averages, float** covariances, int size, int rank) {
    float local_cov[TITLES][INDEX] = {0};
    float global_cov[TITLES][INDEX] = {0};
    int block_size = (DAYS - 1) / size;
    int remainder = (DAYS - 1) % size;
    int start_idx = rank * block_size;
    int end_idx = (rank < size - 1) ? start_idx + block_size : start_idx + block_size + remainder;

    for (int day = start_idx; day < end_idx; day++) {
        for (int title = 0; title < TITLES; title++) {
            for (int idx = 0; idx < INDEX; idx++) {
                local_cov[title][idx] += 
                    (returns[title * (DAYS - 1) + day] - averages[title]) *
                    (returns[(TITLES + idx) * (DAYS - 1) + day] - averages[TITLES + idx]);
            }
        }
    }

    MPI_Allreduce(&local_cov, &global_cov, TITLES * INDEX, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    for (int i = 0; i < TITLES; i++) {
        for (int j = 0; j < INDEX; j++) {
            covariances[i][j] = global_cov[i][j] / (float)(DAYS - 1);
        }
    }
}

void calculateBetasAndCorrelations(float** covariances, float* variances, float* stdDevs, 
                                 float** betas, float** correlations, int size, int rank) {
    for (int title = 0; title < TITLES; title++) {
        for (int idx = 0; idx < INDEX; idx++) {
            if (variances[TITLES + idx] != 0) {
                betas[title][idx] = covariances[title][idx] / variances[TITLES + idx];
                correlations[title][idx] = covariances[title][idx] / 
                    (stdDevs[TITLES + idx] * stdDevs[title]);
            }
        }
    }
}

void printResults(FinancialData* data, int rank) {
    if (rank == 0) {
        printf("\n=== ANALISI DEI RENDIMENTI FINANZIARI ===\n");
        
        // Print returns
        for (int i = 0; i < TITLES + INDEX; i++) {
            printf("\n%s %d:\n", (i < TITLES) ? "TITOLO" : "INDICE", i);
            printf("Media: %.4f%%\n", data->averages[i] * 100);
            printf("Deviazione Standard: %.4f%%\n", data->stdDevs[i] * 100);
        }

        // Print betas and correlations
        printf("\n=== BETA E CORRELAZIONI ===\n");
        for (int i = 0; i < TITLES; i++) {
            printf("\nTitolo %d:\n", i);
            for (int j = 0; j < INDEX; j++) {
                printf("Con Indice %d - Beta: %.4f, Correlazione: %.4f\n",
                       j, data->betas[i][j], data->correlations[i][j]);
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

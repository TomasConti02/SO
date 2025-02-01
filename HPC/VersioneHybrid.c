//mpicc -fopenmp MPI_AND_OpenMP.c -o hybrid_finance -lm
//export OMP_NUM_THREADS=4
//mpirun -np 2 ./hybrid_finance
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>
#include <omp.h>  // Added OpenMP header

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
    int provided;
    // Initialize MPI with thread support
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        printf("Warning: The MPI implementation does not have full thread support\n");
    }

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Set number of OpenMP threads
    #pragma omp parallel
    {
        if (rank == 0 && omp_get_thread_num() == 0) {
            printf("Number of OpenMP threads per MPI process: %d\n", omp_get_num_threads());
        }
    }

    // Rest of initialization remains the same...
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

    if (rank == 0) {
        initializeData(data.prices);
    }

    MPI_Bcast(data.prices, (TITLES + INDEX) * DAYS, MPI_FLOAT, 0, MPI_COMM_WORLD);

    calculateReturns(data.prices, data.returns, size, rank);
    calculateAverages(data.returns, data.averages, size, rank);
    calculateVariances(data.returns, data.averages, data.variances, data.stdDevs, size, rank);
    calculateCovariances(data.returns, data.averages, data.covariances, size, rank);
    calculateBetasAndCorrelations(data.covariances, data.variances, data.stdDevs, data.betas, data.correlations, size, rank);

    printResults(&data, rank);
    cleanupData(&data);
    MPI_Finalize();
    return 0;
}

// InitializeData remains the same...

void calculateReturns(float* prices, float* returns, int size, int rank) {
    int block_size = ((TITLES + INDEX) * (DAYS - 1)) / size;
    int remainder = ((TITLES + INDEX) * (DAYS - 1)) % size;
    int start_idx = rank * block_size;
    int end_idx = (rank < size - 1) ? start_idx + block_size : start_idx + block_size + remainder;

    #pragma omp parallel for schedule(static)
    for (int i = start_idx; i < end_idx; i++) {
        int row = i / (DAYS - 1);
        int day = i % (DAYS - 1);
        returns[i] = (prices[row * DAYS + day + 1] - prices[row * DAYS + day]) / prices[row * DAYS + day];
    }

    MPI_Allgather(returns + start_idx, end_idx - start_idx, MPI_FLOAT, returns, block_size, MPI_FLOAT, MPI_COMM_WORLD);
}

void calculateAverages(float* returns, float* averages, int size, int rank) {
    float local_sum[TITLES + INDEX] = {0};
    float global_sum[TITLES + INDEX] = {0};
    int block_size = ((TITLES + INDEX) * (DAYS - 1)) / size;
    int remainder = ((TITLES + INDEX) * (DAYS - 1)) % size;
    int start_idx = rank * block_size;
    int end_idx = (rank < size - 1) ? start_idx + block_size : start_idx + block_size + remainder;

    #pragma omp parallel
    {
        float thread_local_sum[TITLES + INDEX] = {0};
        
        #pragma omp for schedule(static)
        for (int i = start_idx; i < end_idx; i++) {
            int row = i / (DAYS - 1);
            thread_local_sum[row] += returns[i];
        }

        #pragma omp critical
        for (int i = 0; i < TITLES + INDEX; i++) {
            local_sum[i] += thread_local_sum[i];
        }
    }

    MPI_Allreduce(local_sum, global_sum, TITLES + INDEX, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    #pragma omp parallel for schedule(static)
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

    #pragma omp parallel
    {
        float thread_local_var[TITLES + INDEX] = {0};
        
        #pragma omp for schedule(static)
        for (int i = start_idx; i < end_idx; i++) {
            int row = i / (DAYS - 1);
            float diff = returns[i] - averages[row];
            thread_local_var[row] += diff * diff;
        }

        #pragma omp critical
        for (int i = 0; i < TITLES + INDEX; i++) {
            local_var[i] += thread_local_var[i];
        }
    }

    MPI_Allreduce(local_var, global_var, TITLES + INDEX, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    #pragma omp parallel for schedule(static)
    for (int row = 0; row < TITLES + INDEX; row++) {
        variances[row] = global_var[row] / (float)(DAYS - 1);
        stdDevs[row] = sqrtf(variances[row]);
    }
}

void calculateCovariances(float* returns, float* averages, float** covariances, int size, int rank) {
    int block_size = ((INDEX) * (DAYS - 1)) / size;
    int remainder = ((INDEX) * (DAYS - 1)) % size;
    int start_idx = rank * block_size;
    int end_idx = (rank < size - 1) ? start_idx + block_size : start_idx + block_size + remainder;
    
    float local_cov[TITLES][INDEX] = {0};
    float global_cov[TITLES][INDEX] = {0};

    #pragma omp parallel
    {
        float thread_local_cov[TITLES][INDEX] = {0};
        
        #pragma omp for schedule(static)
        for (int i = start_idx; i < end_idx; i++) {
            int index_col = i / (DAYS - 1);
            int day = i % (DAYS - 1);
            
            for (int title = 0; title < TITLES; title++) {
                thread_local_cov[title][index_col] += 
                    (returns[(title * (DAYS - 1)) + day] - averages[title]) *
                    (returns[((TITLES + index_col) * (DAYS - 1)) + day] - averages[TITLES + index_col]);
            }
        }

        #pragma omp critical
        for (int title = 0; title < TITLES; title++) {
            for (int idx = 0; idx < INDEX; idx++) {
                local_cov[title][idx] += thread_local_cov[title][idx];
            }
        }
    }

    MPI_Allreduce(&local_cov, &global_cov, TITLES * INDEX, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < TITLES; i++) {
        for (int j = 0; j < INDEX; j++) {
            covariances[i][j] = global_cov[i][j] / (float)(DAYS - 1);
        }
    }
}

void calculateBetasAndCorrelations(float** covariances, float* variances, float* stdDevs, 
                                 float** betas, float** correlations, int size, int rank) {
    int block_size = (TITLES * INDEX) / size;
    int remainder = (TITLES * INDEX) % size;
    int start_idx = rank * block_size;
    int end_idx = (rank < size - 1) ? start_idx + block_size : start_idx + block_size + remainder;

    float beta_local[TITLES][INDEX] = {0};
    float correlation_local[TITLES][INDEX] = {0};
    float beta_global[TITLES][INDEX] = {0};
    float correlation_global[TITLES][INDEX] = {0};

    // Calcola solo per gli indici assegnati a questo processo
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < TITLES; i++) {
        for (int j = 0; j < INDEX; j++) {
            // Calcola solo se questo elemento appartiene al blocco del processo
            int flat_idx = i * INDEX + j;
            if (flat_idx >= start_idx && flat_idx < end_idx) {
                if (variances[TITLES + j] != 0) {
                    beta_local[i][j] = covariances[i][j] / variances[TITLES + j];
                    correlation_local[i][j] = covariances[i][j] / 
                        (stdDevs[TITLES + j] * stdDevs[i]);
                }
            }
        }
    }

    // Raccoglie i risultati da tutti i processi
    MPI_Allreduce(&beta_local, &beta_global, TITLES * INDEX, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&correlation_local, &correlation_global, TITLES * INDEX, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    // Copia i risultati nelle matrici finali
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < TITLES; i++) {
        for (int j = 0; j < INDEX; j++) {
            betas[i][j] = beta_global[i][j];
            correlations[i][j] = correlation_global[i][j];
        }
    }
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

// PrintResults and cleanupData remain the same...
void printResults(FinancialData* data, int rank) {
    if (rank == 0) {
         printf("\n=== ANALISI DEI RENDIMENTI FINANZIARI ===\n");

    // Stampa medie, varianze e deviazioni standard
    for (int i = 0; i < TITLES + INDEX; i++) {
        printf("\n%s %d:\n", (i < TITLES) ? "TITOLO" : "INDICE", i);
        printf("Media: %f\n", data->averages[i]);
        printf("Deviazione Standard: %.4f%%\n", data->stdDevs[i] * 100);
        printf("Varianza: %f\n", data->variances[i]);
    }

    // Stampa covarianze
    printf("\n=== COVARIANZE ===\n");
    for (int i = 0; i < TITLES; i++) {
        for (int j = 0; j < INDEX; j++) {
            printf("Covarianza Titolo %d con Indice %d: %.4f\n", i, j, data->covariances[i][j]);
        }
    }

    // Stampa beta e correlazioni
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

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>    // Per time()

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
    //MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,  returns, block_size, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgather(returns + start_idx, end_idx - start_idx, MPI_FLOAT, returns, block_size, MPI_FLOAT, MPI_COMM_WORLD);
   // MPI_Gatherv(returns + start_idx, end_idx - start_idx, MPI_FLOAT, returns, recv_counts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
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
        //float diff = returns[i] - averages[row];
        local_var[row] += ((returns[i] - averages[row])*(returns[i] - averages[row]));
    }

    MPI_Allreduce(local_var, global_var, TITLES + INDEX, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    for (int row = 0; row < TITLES + INDEX; row++) {
        variances[row] = global_var[row] / (float)(DAYS - 1);
        stdDevs[row] = sqrtf(variances[row]);
    }
}
void calculateCovariances(float* returns, float* averages, float** covariances, int size, int rank) {
    int block_size = ((INDEX) * (DAYS - 1)) / size;
    int remainder = ((INDEX) * (DAYS - 1)) % size;

    // Array per memorizzare i risultati locali e globali
    float local_cov[TITLES][INDEX] = {0};
    float global_cov[TITLES][INDEX] = {0};
    
    // Calcolo degli indici di inizio e fine per questo processo
     int start_idx = rank * block_size;
     int end_idx = (rank < size - 1) ?  start_idx + block_size :  start_idx + block_size + remainder;
      for (int i = start_idx; i < end_idx; i++) {
        // Determina l'indice e il giorno dalla posizione i
        int index_col = i / (DAYS - 1);    // Determina quale indice
        int day = i % (DAYS - 1);          // Determina il giorno

        // Calcola la covarianza per tutti i titoli con questo indice
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
/*
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
}*/
/*OKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK
void calculateCovariances(float* returns, float* averages, float** covariances, int size, int rank) {
    float local_cov[TITLES][INDEX] = {0};
    float global_cov[TITLES][INDEX] = {0};
    
    // Calcolare il numero di colonne da trattare per ciascun processo
    int block_size = (DAYS - 1) / size;
    int remainder = (DAYS - 1) % size;
    int start_idx = rank * block_size;
    int end_idx = (rank < size - 1) ? start_idx + block_size : start_idx + block_size + remainder;

    // Distribuire per colonne (giorni)
    for (int day = start_idx; day < end_idx; day++) {
        for (int title = 0; title < TITLES; title++) {
            for (int idx = 0; idx < INDEX; idx++) {
                // Calcolo della covarianza per titolo e indice
                local_cov[title][idx] += 
                    (returns[title * (DAYS - 1) + day] - averages[title]) *
                    (returns[(TITLES + idx) * (DAYS - 1) + day] - averages[TITLES + idx]);
            }
        }
    }

    // Sommare i risultati usando MPI_Allreduce
    MPI_Allreduce(&local_cov, &global_cov, TITLES * INDEX, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    // Calcolare la covarianza media per ogni coppia
    for (int i = 0; i < TITLES; i++) {
        for (int j = 0; j < INDEX; j++) {
            covariances[i][j] = global_cov[i][j] / (float)(DAYS - 1);
        }
    }
}*/
/*
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
}*/
void calculateBetasAndCorrelations(float** covariances, float* variances, float* stdDevs, float** betas, float** correlations, int size, int rank) {

    int block_size = (TITLES * INDEX) / size; 
    int remainder = (TITLES * INDEX) % size;

    int start_idx = rank * block_size;
    int end_idx = (rank < size - 1) ? start_idx + block_size : start_idx + block_size + remainder;

    // Inizializza array locali
    float beta_local[TITLES][INDEX] = {0};
    float correlation_local[TITLES][INDEX] = {0};
    float beta_global[TITLES][INDEX] = {0};
    float correlation_global[TITLES][INDEX] = {0};

    // Calcolo parallelo di Beta e Correlazione solo per il blocco assegnato
    for (int i = start_idx; i < end_idx; i++) {
        int title = i / INDEX;
        int idx = i % INDEX;

        if (variances[TITLES + idx] != 0) {
            beta_local[title][idx] = covariances[title][idx] / variances[TITLES + idx];
            correlation_local[title][idx] = covariances[title][idx] / 
                                            (stdDevs[TITLES + idx] * stdDevs[title]);
        }
    }

    // Riduzione dei risultati per ottenere i valori globali
    MPI_Allreduce(&beta_local, &beta_global, TITLES * INDEX, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&correlation_local, &correlation_global, TITLES * INDEX, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    // Assegnare i valori finali alle matrici di output
    for (int i = 0; i < TITLES; i++) {
        for (int j = 0; j < INDEX; j++) {
            betas[i][j] = beta_global[i][j];
            correlations[i][j] = correlation_global[i][j];
        }
    }
}
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

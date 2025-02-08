#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>

#define DAYS 5
#define TITLES 2
#define INDEX 2

typedef struct {
    float* prices;
    float* returns;
    float* averages;
    float* variances;
    float* stdDevs;
    float*covariances;
    float** betas;
    float** correlations;
} FinancialData;
/*
void initializeData(float* prices) {
    srand(time(NULL));
    for (int i = 0; i < TITLES + INDEX; i++) {
        for (int j = 0; j < DAYS; j++) {
            // Base price tra 50 e 200 con variazione casuale
            float base_price = 50 + (rand() % 151);
            prices[i * DAYS + j] = base_price + ((float)rand() / RAND_MAX);
        }
    }
}*/

void initializeData(float* prices) {
    float valori_titolo_1[] = {200, 150, 100, 100, 90};
    float valori_titolo_0[] = {100, 110, 130, 125, 140};
    float valori_indice_0[] = {100, 105, 115, 113, 120};
    float valori_indice_1[] = {50, 90, 100, 120, 150};

    int num_valori = sizeof(valori_titolo_0) / sizeof(valori_titolo_0[0]); // Get the number of elements

    for (int j = 0; j < DAYS; j++) {
        prices[0 * DAYS + j] = valori_titolo_0[j % num_valori];  // Use modulo operator
        prices[1 * DAYS + j] = valori_titolo_1[j % num_valori];  // Use mlsodulo operator
        prices[2 * DAYS + j] = valori_indice_0[j % num_valori];  // Use modulo operator
        prices[3 * DAYS + j] = valori_indice_1[j % num_valori];  // Use modulo operator
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
            float diff = returns[row * (DAYS - 1) + day] - averages[row]; // Calculate difference once
            variances[row] += diff * diff; // Use the difference
        }
        variances[row] /= (float)(DAYS - 1);
        stdDevs[row] = sqrtf(variances[row]);
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
//
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
            printf("calcoalta dal rank =%d ->Cov(%f)\n",rank, local_cov[i * TITLES + t]);
        }
    }
    MPI_Allgatherv(local_cov, local_index_size * TITLES, MPI_FLOAT, covariances, recvcounts, displs, MPI_FLOAT, MPI_COMM_WORLD);
}
/*
void calculateBetasAndCorrelations(float* covariances, float* variances, float* stdDevs, float** betas, float** correlations, int size, int rank) {
    int block_size = (TITLES * INDEX) / size;
    int remainder = (TITLES * INDEX) % size;
    int start_idx = rank * block_size;
    int end_idx = (rank < size - 1) ? start_idx + block_size : start_idx + block_size + remainder;
    const float epsilon = 1e-7;
    for (int i = start_idx; i < end_idx; i++) {
        int title = i / INDEX;
        int idx = i % INDEX;
        if (fabsf(variances[TITLES + idx]) > epsilon) {
            betas[title][idx] = covariances[title * INDEX + idx] / variances[TITLES + idx];
        } else {
            betas[title][idx] = 0.0f; // Default value if variance is near-zero
        }
        if (fabsf(stdDevs[TITLES + idx]) > epsilon && fabsf(stdDevs[title]) > epsilon) {
            correlations[title][idx] = covariances[title * INDEX + idx] / (stdDevs[TITLES + idx] * stdDevs[title]);
        } else {
            correlations[title][idx] = 0.0f; // Default value if standard deviation is near-zero
        }

        // Print local results for this process
        
    }
}*/
/*
void calculateBetasAndCorrelations(float* covariances, float* variances, float* stdDevs, float** betas, float** correlations, int size, int rank) {
    // Calcolo del numero totale di combinazioni
    int total_combinations = TITLES * INDEX;
    
    // Distribuzione del lavoro
    int block_size = total_combinations / size;
    int remainder = total_combinations % size;
    int start_idx = rank * block_size + (rank < remainder ? rank : remainder);
    int end_idx = start_idx + block_size + (rank < remainder ? 1 : 0);
    
    const float epsilon = 1e-7;
    
    // Debug: stampa la distribuzione del lavoro
    printf("Rank %d: calcolando combinazioni da %d a %d\n", rank, start_idx, end_idx - 1);
    
    // Calcolo locale per ogni combinazione assegnata al processo
    for (int i = start_idx; i < end_idx; i++) {
        int title = i / INDEX;
        int idx = i % INDEX;
        
        if (fabsf(variances[TITLES + idx]) > epsilon) {
            betas[title][idx] = covariances[title + idx * TITLES] / variances[TITLES + idx];
            printf("Rank %d: Beta[%d][%d] = %f\n", rank, title, idx, betas[title][idx]);
            correlations[title][idx] = covariances[title + idx * TITLES] /  (sqrtf(variances[TITLES + idx]) * sqrtf(stdDevs[title]));
            printf("Rank %d: Correlation[%d][%d] = %f\n", rank, title, idx, correlations[title][idx]);
        } else {
            betas[title][idx] = 0.0f;
            printf("Rank %d: Beta[%d][%d] = 0 (varianza troppo piccola)\n", rank, title, idx);
        }
        
        if (fabsf(stdDevs[TITLES + idx]) > epsilon && fabsf(stdDevs[title]) > epsilon) {
            
            printf("Rank %d: Correlation[%d][%d] = %f\n", rank, title, idx, correlations[title][idx]);
        } else {
            correlations[title][idx] = 0.0f;
            printf("Rank %d: Correlation[%d][%d] = 0 (deviazione standard troppo piccola)\n", rank, title, idx);
        }
    }
}*/
void calculateBetasAndCorrelations(float* covariances, float* variances, float** betas, float** correlations, int size, int rank) {
    // Calcolo del numero totale di combinazioni
    int total_combinations = TITLES * INDEX;
    
    // Distribuzione del lavoro
    int block_size = total_combinations / size;
    int remainder = total_combinations % size;
    int start_idx = rank * block_size + (rank < remainder ? rank : remainder);
    int end_idx = start_idx + block_size + (rank < remainder ? 1 : 0);
    
    const float epsilon = 1e-7;
    
    // Debug: stampa la distribuzione del lavoro
    printf("Rank %d: calcolando combinazioni da %d a %d\n", rank, start_idx, end_idx - 1);
    
    // Calcolo locale per ogni combinazione assegnata al processo
    for (int i = start_idx; i < end_idx; i++) {
        int title = i / INDEX;
        int idx = i % INDEX;
        
        // Calcolo del beta
        if (fabsf(variances[TITLES + idx]) > epsilon) {
            betas[title][idx] = covariances[title + idx * TITLES] / variances[TITLES + idx];
            printf("Rank %d: Beta[%d][%d] = %f\n", rank, title, idx, betas[title][idx]);
        } else {
            betas[title][idx] = 0.0f;
            printf("Rank %d: Beta[%d][%d] = 0 (varianza troppo piccola)\n", rank, title, idx);
        }
        
        // Calcolo della correlazione
        if (fabsf(variances[TITLES + idx]) > epsilon && fabsf(variances[title]) > epsilon) {
            correlations[title][idx] = covariances[title + idx * TITLES] / (sqrtf(variances[TITLES + idx]) * variances[title]);
            printf("Rank %d: Correlation[%d][%d] = %f\n", rank, title, idx, correlations[title][idx]);
        } else {
            correlations[title][idx] = 0.0f;
            printf("Rank %d: Correlation[%d][%d] = 0 (deviazione standard troppo piccola)\n", rank, title, idx);
        }
    }
}
void printPrices(float* prices, int rank) {
    if (rank == 0) {
        printf("Prices:\n");
        for (int i = 0; i < TITLES + INDEX; i++) {
            printf("Asset %d: ", i);
            for (int j = 0; j < DAYS; j++) {
                printf("%.2f ", prices[i * DAYS + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

void printReturns(float* returns, int rank) {
    if (rank == 0) {
        printf("Returns:\n");
        for (int i = 0; i < TITLES + INDEX; i++) {
            printf("Asset %d: ", i);
            for (int j = 0; j < DAYS - 1; j++) {
                printf("%.6f ", returns[i * (DAYS - 1) + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

void printAverages(float* averages, int rank) {
    if (rank == 0) {
        printf("Averages:\n");
        for (int i = 0; i < TITLES + INDEX; i++) {
            printf("Asset %d: %.6f\n", i, averages[i]);
        }
        printf("\n");
    }
}

void printVariancesAndStdDevs(float* variances, float* stdDevs, int rank) {
    if (rank == 0) {
        printf("Variances and Standard Deviations:\n");
        for (int i = 0; i < TITLES + INDEX; i++) {
            printf("Asset %d: Variance = %.6f, StdDev = %.6f\n", i, variances[i], stdDevs[i]);
        }
        printf("\n");
    }
}

void printCovariancesMatrix(float* covariances, int rank) {
    if (rank == 0) {
        printf("\nCovariance Matrix:\n");
        printf("           ");
        // Print header with Index numbers
        for (int j = 0; j < INDEX; j++) {
            printf("Index%-5d ", j);
        }
        printf("\n");
        
        // Print each row
        for (int i = 0; i < TITLES; i++) {
            printf("Title%-5d ", i);
            for (int j = 0; j < INDEX; j++) {
                printf("%-10.6f ", covariances[i + j * TITLES]);
            }
            printf("\n");
        }
        printf("\n");
    }
}
void printBetasAndCorrelations(float** betas, float** correlations, int rank) {
    if (rank == 0) {
        printf("Betas and Correlations:\n");
        for (int i = 0; i < TITLES; i++) {
            for (int j = 0; j < INDEX; j++) {
                printf("Beta(Title %d, Index %d) = %.6f, Correlation(Title %d, Index %d) = %.6f\n",
                       i, j, betas[i][j], i, j, correlations[i][j]);
            }
        }
        printf("\n");
    }
}
int main(int argc, char* argv[]) {
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double begin, end, global_elaps, local_elaps;

    FinancialData data = {0};

    // Allocate memory for FinancialData structure
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
    /////////////////////////////////////////
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
    //////////////////////////////////////// WORK DISTRIBUTION FOR EACH CORE
    int block_size = (int)ceil((double)(TITLES + INDEX) / size);
    int start_idx = rank * block_size;
    int end_idx = (rank + 1) * block_size;
    if (end_idx > TITLES + INDEX) end_idx = TITLES + INDEX;
    int local_size = end_idx - start_idx;
    ////////////////////////////////////////

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
    printPrices(data.prices, rank);
    calculateReturns(data.prices + start_idx * DAYS, localReturns, local_size, rank);
    calculateAverages(localReturns, localAverage, local_size, rank);
    
    calculateVariances(localReturns, localAverage, localVariance, localDev, local_size, rank);
    
    MPI_Allgatherv(localReturns, local_size * (DAYS - 1), MPI_FLOAT,data.returns, sendcounts_returns, displs_returns, MPI_FLOAT,MPI_COMM_WORLD);
    MPI_Allgatherv(localAverage, local_size, MPI_FLOAT,data.averages, sendcounts_other, displs_other, MPI_FLOAT,MPI_COMM_WORLD);
    MPI_Allgatherv(localVariance, local_size, MPI_FLOAT,data.variances, sendcounts_other, displs_other, MPI_FLOAT,MPI_COMM_WORLD);
    printReturns(data.returns, rank);
    printAverages(data.averages, rank);
    printVariancesAndStdDevs(data.variances, data.stdDevs, rank);
    // ... (Your covariance, beta, and correlation calculations using the gathered data)
    calculateCovariances(data.returns, data.averages, data.covariances, size, rank, recvcounts, displs, local_cov);
    calculateBetasAndCorrelations(data.covariances, data.variances, data.betas, data.correlations, size, rank);
    MPI_Barrier(MPI_COMM_WORLD);
    printCovariancesMatrix(data.covariances, rank);
    printBetasAndCorrelations(data.betas, data.correlations, rank);
    end = MPI_Wtime();
    local_elaps = end - begin;
    MPI_Reduce(&local_elaps, &global_elaps, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Number of processes: %d\n", size);
        printf("Execution time: %f seconds\n", global_elaps);
        printf("Input size: %d elements\n", (TITLES + INDEX) * DAYS );
        printf("Memory used: %lu MB\n", ((TITLES + INDEX) * DAYS * sizeof(float)) / (1024 * 1024) );
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















/*

void calculateCovariances(float* returns, float* averages, float* covariances, int size, int rank, float* local_cov) {
    int block_size = INDEX / size;
    int remainder = INDEX % size;
    int start_idx = rank * block_size + (rank < remainder ? rank : remainder);
    int end_idx = start_idx + block_size + (rank < remainder ? 1 : 0);

    // Calcola le covarianze locali
    for (int i = 0, idx = start_idx; idx < end_idx; i++, idx++) {
        for (int t = 0; t < TITLES; t++) {
            float sum = 0.0f;
            for (int d = 0; d < DAYS - 1; d++) {
                sum += (returns[t * (DAYS - 1) + d] - averages[t]) * (returns[idx * (DAYS - 1) + d] - averages[idx]);
            }
            local_cov[i * TITLES + t] = sum / (DAYS - 1);
        }
    }

    // Raccogli i risultati da tutti i processi
    MPI_Allgather(local_cov, TITLES * (end_idx - start_idx), MPI_FLOAT, covariances, TITLES * (end_idx - start_idx), MPI_FLOAT, MPI_COMM_WORLD);
}
*/
/*
void calculateCovariances(float* returns, float* averages, float* covariances, int size, int rank, float* local_cov) {
    int block_size = INDEX / size;
    int remainder = INDEX % size;
    int start_idx = rank * block_size + (rank < remainder ? rank : remainder);
    int end_idx = start_idx + block_size + (rank < remainder ? 1 : 0);
    int local_size = end_idx - start_idx;

    // Calcola le covarianze locali
    for (int i = 0, idx = start_idx; idx < end_idx; i++, idx++) {
        for (int t = 0; t < TITLES; t++) {
            float sum = 0.0f;
            for (int d = 0; d < DAYS - 1; d++) {
                sum += (returns[t * (DAYS - 1) + d] - averages[t]) * 
                       (returns[idx * (DAYS - 1) + d] - averages[idx]);
            }
            local_cov[i * TITLES + t] = sum / (DAYS - 1);
        }
    }

    // Prepara i parametri per MPI_Allgatherv
    int recvcounts[size];
    int displs[size];
    for(int i = 0; i < size; i++) {
        int proc_block = block_size + (i < remainder ? 1 : 0);
        recvcounts[i] = proc_block * TITLES;
        displs[i] = (i * block_size + (i < remainder ? i : remainder)) * TITLES;
    }

    // Buffer temporaneo per la raccolta
    float* temp_buffer = (float*)malloc(INDEX * TITLES * sizeof(float));
    
    // Raccogli i risultati da tutti i processi
    MPI_Allgatherv(local_cov, local_size * TITLES, MPI_FLOAT, temp_buffer, recvcounts, displs, MPI_FLOAT, MPI_COMM_WORLD);

    // Copia nel formato finale
    for (int i = 0; i < TITLES; i++) {
        for (int j = 0; j < INDEX; j++) {
             covariances[i * INDEX + j]  = temp_buffer[j * TITLES + i];
        }
    }

    free(temp_buffer);
}*/

/*
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
}*/
/*
void calculateBetasAndCorrelations(float* covariances, float* variances, float* stdDevs, float** betas, float** correlations, int size, int rank) {
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
            betas[title][idx] = covariances[title * INDEX + idx] / variances[TITLES + idx];
        } else {
            betas[title][idx] = 0.0f; // Default value if variance is near-zero
        }

        // Check for near-zero standard deviations before calculating correlation
        if (fabsf(stdDevs[TITLES + idx]) > epsilon && fabsf(stdDevs[title]) > epsilon) {
            correlations[title][idx] = covariances[title * INDEX + idx] / (stdDevs[TITLES + idx] * stdDevs[title]);
        } else {
            correlations[title][idx] = 0.0f; // Default value if standard deviation is near-zero
        }

        // Print local results for this process
        //printf("Rank %d: Beta[%d][%d] = %f, Correlation[%d][%d] = %f\n", rank, title, idx, betas[title][idx], title, idx, correlations[title][idx]);
    }
}*/

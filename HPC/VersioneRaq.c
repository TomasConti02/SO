
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

    int start_idx = rank * block_size;
    int end_idx = (rank < size - 1) ? start_idx + block_size : start_idx + block_size + remainder;

    for (int row = start_idx; row < end_idx; row++) {
        for (int day = 0; day < (DAYS - 1); day++) {
            returns[row * (DAYS - 1) + day] = (prices[row * DAYS + day + 1] - prices[row * DAYS + day]) / prices[row * DAYS + day];
        }
    }
}

void calculateAverages(float* returns, float* averages, int size, int rank) {
    int block_size = ((TITLES + INDEX)) / size;
    int remainder = ((TITLES + INDEX)) % size;
    int start_idx = rank * block_size;
    int end_idx = (rank < size - 1) ? start_idx + block_size : start_idx + block_size + remainder;

    for (int row = start_idx; row < end_idx; row++) {
        averages[row] = 0.0f;
        for (int day = 0; day < (DAYS - 1); day++) {
            averages[row] += returns[row * (DAYS - 1) + day];
        }
        averages[row] /= (DAYS - 1);
    }
}

void calculateVariances(float* returns, float* averages, float* variances, float* stdDevs, int size, int rank) {
    int block_size = ((TITLES + INDEX)) / size;
    int remainder = ((TITLES + INDEX)) % size;
    int start_idx = rank * block_size;
    int end_idx = (rank < size - 1) ? start_idx + block_size : start_idx + block_size + remainder;

    for (int row = start_idx; row < end_idx; row++) {
        variances[row] = 0;
        for (int day = 0; day < (DAYS - 1); day++) {
            variances[row] += ((returns[row * (DAYS - 1) + day] - averages[row]) * (returns[row * (DAYS - 1) + day] - averages[row]));
        }
        variances[row] = variances[row] / (float)(DAYS - 1);
        stdDevs[row] = sqrtf(variances[row]);
    }
}
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
}
int main(int argc, char* argv[]) {
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double begin, end;

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

    // Check for allocation failures
    if (!data.prices || !data.returns || !data.averages || !data.variances || !data.stdDevs || 
        !data.covariances || !data.betas || !data.correlations) {
        fprintf(stderr, "Memory allocation failed\n");
        cleanupFinancialData(&data);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Allocate 2D arrays
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
    }

    if (rank == 0) {
        initializeData(data.prices);
    }
    MPI_Bcast(data.prices, (TITLES + INDEX) * DAYS, MPI_FLOAT, 0, MPI_COMM_WORLD);

    begin = MPI_Wtime();

    // Calculate returns
    calculateReturns(data.prices, data.returns, size, rank);
    calculateAverages(data.returns, data.averages, size, rank);
    calculateVariances(data.returns, data.averages, data.variances, data.stdDevs, size, rank);

    // Gather results to rank 0
    MPI_Allgather(data.returns + rank * (DAYS - 1), (DAYS - 1), MPI_FLOAT,  data.returns, (DAYS - 1), MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgather(data.averages + rank, 1, MPI_FLOAT,  data.averages, 1, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgather(data.variances + rank, 1, MPI_FLOAT,   data.variances, 1, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgather(data.stdDevs + rank, 1, MPI_FLOAT, data.stdDevs, 1, MPI_FLOAT, MPI_COMM_WORLD);

    //MPI_Bcast(data.returns, (TITLES + INDEX) * (DAYS - 1), MPI_FLOAT, 0, MPI_COMM_WORLD);
    //MPI_Bcast(data.averages, (TITLES + INDEX), MPI_FLOAT, 0, MPI_COMM_WORLD);

    calculateCovariances(data.returns, data.averages, data.covariances, size, rank);
    if (rank == 0) {
    // Stampa i ritorni
    printf("Returns after Gather:\n");
    for (int i = 0; i < (TITLES + INDEX) * (DAYS - 1); i++) {
        printf("Return[%d] = %f\n", i, data.returns[i]);
    }

    // Stampa le medie
    printf("Averages after Gather:\n");
    for (int i = 0; i < (TITLES + INDEX); i++) {
        printf("Average[%d] = %f\n", i, data.averages[i]);
    }
    
    // Stampa le varianze
    printf("Variances after Gather:\n");
    for (int i = 0; i < (TITLES + INDEX); i++) {
        printf("Variance[%d] = %f\n", i, data.variances[i]);
    }

    // Stampa le deviazioni standard
    printf("Standard Deviations after Gather:\n");
    for (int i = 0; i < (TITLES + INDEX); i++) {
        printf("StdDev[%d] = %f\n", i, data.stdDevs[i]);
    }

    // Stampa le covarianze
    printf("\nCovariances:\n");
    for (int i = 0; i < TITLES; i++) {
        for (int j = 0; j < INDEX; j++) {
            printf("Covariance[%d][%d] = %f\n", i, j, data.covariances[i][j]);
        }
    }
    }
    // Cleanup
    cleanupFinancialData(&data);
    MPI_Finalize();
    return 0;
}
/*#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>    // Per time()

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
    // Simula l'inizializzazione dei dati
    for (int i = 0; i < (TITLES + INDEX) * DAYS; i++) {
        prices[i] = (float)rand() / RAND_MAX;
    }
}*/
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

    int start_idx = rank * block_size;
    int end_idx = (rank < size - 1) ? start_idx + block_size : start_idx + block_size + remainder;

    for (int row = start_idx; row < end_idx; row++) {
        for (int day = 0; day < (DAYS - 1); day++) {
            returns[row * (DAYS - 1) + day] = (prices[row * DAYS + day + 1] - prices[row * DAYS + day]) / prices[row * DAYS + day];
            //printf("Return[%d] = %f\n", (row * (DAYS - 1) + day), returns[row * (DAYS - 1) + day]);
        }
    }
}
void calculateAverages(float* returns, float* averages, int size, int rank) {
    int block_size = ((TITLES + INDEX) ) / size;
    int remainder = ((TITLES + INDEX) ) % size;
    int start_idx = rank * block_size;
    int end_idx = (rank < size - 1) ? start_idx + block_size : start_idx + block_size + remainder;
     for (int row = start_idx; row < end_idx; row++) {
        averages[row] = 0.0f;
        for (int day = 0; day < (DAYS - 1); day++) {
            averages[row] += returns[row * (DAYS - 1) + day];
        }
        averages[row] /= (DAYS - 1);
        //printf("Average[%d] = %f\n", row, averages[row]);
    }
}
void calculateVariances(float* returns, float* averages, float* variances, float* stdDevs, int size, int rank) {
    int block_size = ((TITLES + INDEX) ) / size;
    int remainder = ((TITLES + INDEX) ) % size;
    int start_idx = rank * block_size;
    int end_idx = (rank < size - 1) ? start_idx + block_size : start_idx + block_size + remainder;

    for (int row = start_idx; row < end_idx; row++) {
        variances[row]=0;
        for (int day = 0; day < (DAYS - 1); day++) {
        variances[row]+= ((returns[row * (DAYS - 1) + day] - averages[row]) * (returns[row * (DAYS - 1) + day] - averages[row]));
        }
        variances[row] = variances[row] / (float)(DAYS - 1);
        stdDevs[row] = sqrtf(variances[row]);
        //printf("Variance[%d] = %f\n", row, variances[row]);
    }
}
int main(int argc, char* argv[]) {
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double begin, end, global_elaps, local_elaps;

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
    // Check for allocation failures
    if (!data.prices || !data.returns || !data.averages || !data.variances || !data.stdDevs || 
        !data.covariances || !data.betas || !data.correlations) {
        fprintf(stderr, "Memory allocation failed\n");
        cleanupFinancialData(&data);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Allocate 2D arrays
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
    }

    if (rank == 0) {
        initializeData(data.prices);
    }
    MPI_Bcast(data.prices, (TITLES + INDEX) * DAYS, MPI_FLOAT, 0, MPI_COMM_WORLD);
    begin = MPI_Wtime();
    // Calculate returns
    calculateReturns(data.prices, data.returns, size, rank);
    calculateAverages(data.returns, data.averages, size, rank);
    calculateVariances(data.returns, data.averages, data.variances, data.stdDevs, size, rank);

    MPI_Allreduce(MPI_IN_PLACE, data.returns, (TITLES + INDEX) * (DAYS - 1), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, data.averages, TITLES + INDEX, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    //MPI_Allreduce(MPI_IN_PLACE, data.variances, TITLES + INDEX, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    //MPI_Allreduce(MPI_IN_PLACE, data.stdDevs, TITLES + INDEX, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
    // Stampa i ritorni
    printf("Returns after Allreduce:\n");
    for (int i = 0; i < (TITLES + INDEX) * (DAYS - 1); i++) {
        printf("Return[%d] = %f\n", i, data.returns[i]);
    }

    // Stampa le medie
    printf("Averages after Allreduce:\n");
    for (int i = 0; i < (TITLES + INDEX); i++) {
        printf("Average[%d] = %f\n", i, data.averages[i]);
    }
    }
    // Pulizia
    cleanupFinancialData(&data);
    MPI_Finalize();
    return 0;
}*/

//gcc -o financial_analysis financial_analysis.c -lm
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DAYS 10
#define TITLES 2
#define INDEX 2

// Struttura per memorizzare i dati finanziari
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

// Prototipi delle funzioni
void initializeData(float* prices);
void calculateReturns(float* prices, float* returns);
void calculateAverages(float* returns, float* averages);
void calculateVariances(float* returns, float* averages, float* variances, float* stdDevs);
void calculateCovariances(float* returns, float* averages, float** covariances);
void calculateBetasAndCorrelations(float** covariances, float* variances, float* stdDevs, float** betas, float** correlations);
void printResults(FinancialData* data);
void cleanupData(FinancialData* data);

int main() {
    // Inizializza la struttura dei dati finanziari
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

    // Alloca memoria per gli array 2D
    for (int i = 0; i < TITLES; i++) {
        data.covariances[i] = (float*)malloc(INDEX * sizeof(float));
        data.betas[i] = (float*)malloc(INDEX * sizeof(float));
        data.correlations[i] = (float*)malloc(INDEX * sizeof(float));
    }

    // Inizializza i dati dei prezzi
    initializeData(data.prices);

    // Calcola i rendimenti
    calculateReturns(data.prices, data.returns);

    // Calcola le medie dei rendimenti
    calculateAverages(data.returns, data.averages);

    // Calcola le varianze e le deviazioni standard
    calculateVariances(data.returns, data.averages, data.variances, data.stdDevs);

    // Calcola le covarianze
    calculateCovariances(data.returns, data.averages, data.covariances);

    // Calcola i beta e le correlazioni
    calculateBetasAndCorrelations(data.covariances, data.variances, data.stdDevs, data.betas, data.correlations);

    // Stampa i risultati
    printResults(&data);

    // Libera la memoria
    cleanupData(&data);

    return 0;
}

// Inizializza i dati dei prezzi
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

// Calcola i rendimenti giornalieri
void calculateReturns(float* prices, float* returns) {
    for (int row = 0; row < TITLES + INDEX; row++) {
        for (int day = 0; day < DAYS - 1; day++) {
            returns[row * (DAYS - 1) + day] = (prices[row * DAYS + day + 1] - prices[row * DAYS + day]) / prices[row * DAYS + day];
        }
    }
}

// Calcola le medie dei rendimenti
void calculateAverages(float* returns, float* averages) {
    for (int row = 0; row < TITLES + INDEX; row++) {
        averages[row] = 0.0f;
        for (int day = 0; day < DAYS - 1; day++) {
            averages[row] += returns[row * (DAYS - 1) + day];
        }
        averages[row] /= (DAYS - 1);
    }
}

// Calcola le varianze e le deviazioni standard
void calculateVariances(float* returns, float* averages, float* variances, float* stdDevs) {
    for (int row = 0; row < TITLES + INDEX; row++) {
        variances[row] = 0.0f;
        for (int day = 0; day < DAYS - 1; day++) {
            float diff = returns[row * (DAYS - 1) + day] - averages[row];
            variances[row] += diff * diff;
        }
        variances[row] /= (DAYS - 1);
        stdDevs[row] = sqrtf(variances[row]);
    }
}

// Calcola le covarianze
void calculateCovariances(float* returns, float* averages, float** covariances) {
    for (int title = 0; title < TITLES; title++) {
        for (int idx = 0; idx < INDEX; idx++) {
            covariances[title][idx] = 0.0f;
            for (int day = 0; day < DAYS - 1; day++) {
                covariances[title][idx] += (returns[title * (DAYS - 1) + day] - averages[title]) *
                                           (returns[(TITLES + idx) * (DAYS - 1) + day] - averages[TITLES + idx]);
            }
            covariances[title][idx] /= (DAYS - 1);
        }
    }
}

// Calcola i beta e le correlazioni
void calculateBetasAndCorrelations(float** covariances, float* variances, float* stdDevs, float** betas, float** correlations) {
    for (int title = 0; title < TITLES; title++) {
        for (int idx = 0; idx < INDEX; idx++) {
            if (variances[TITLES + idx] != 0) {
                betas[title][idx] = covariances[title][idx] / variances[TITLES + idx];
                correlations[title][idx] = covariances[title][idx] / (stdDevs[TITLES + idx] * stdDevs[title]);
            }
        }
    }
}

// Stampa i risultati
void printResults(FinancialData* data) {
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

// Libera la memoria allocata
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

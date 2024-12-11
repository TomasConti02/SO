#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <sched.h>

#define ARRAY_SIZE 1000
#define CHUNK_SIZE (ARRAY_SIZE / 2)

int main(int argc, char *argv[]) {
    int rank, size;
    int cpu_id;
    MPI_Status status;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size != 3) {
        if (rank == 0) {
            printf("Questo programma richiede esattamente 3 processi. Terminazione.\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    cpu_id = sched_getcpu();
    printf("Processo %d in esecuzione su CPU %d\n", rank, cpu_id);

    if (rank == 0) {  // Processo master
        int numbers[ARRAY_SIZE];
        for (int i = 0; i < ARRAY_SIZE; i++) {
            numbers[i] = i + 1;
        }
        
        // Invia la prima metà al processo 1 (calcolerà la somma)
        printf("Master: Invio primi %d numeri al processo 1 per somma\n", CHUNK_SIZE);
        MPI_Send(numbers, CHUNK_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD);
        
        // Invia la seconda metà al processo 2 (calcolerà la media)
        printf("Master: Invio secondi %d numeri al processo 2 per media\n", CHUNK_SIZE);
        MPI_Send(numbers + CHUNK_SIZE, CHUNK_SIZE, MPI_INT, 2, 0, MPI_COMM_WORLD);
        
        // Riceve i risultati
        int sum;
        float average;
        MPI_Recv(&sum, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&average, 1, MPI_FLOAT, 2, 0, MPI_COMM_WORLD, &status);
        
        printf("Master: Risultati ricevuti:\n");
        printf("- Somma dalla prima metà (processo 1): %d\n", sum);
        printf("- Media dalla seconda metà (processo 2): %.2f\n", average);
    }
    else if (rank == 1) {  // Worker 1: calcola la somma
        int chunk[CHUNK_SIZE];
        MPI_Recv(chunk, CHUNK_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        
        int sum = 0;
        for (int i = 0; i < CHUNK_SIZE; i++) {
            sum += chunk[i];
        }
        
        printf("Worker 1: Somma calcolata = %d\n", sum);
        MPI_Send(&sum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    else if (rank == 2) {  // Worker 2: calcola la media
        int chunk[CHUNK_SIZE];
        MPI_Recv(chunk, CHUNK_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        
        int sum = 0;
        for (int i = 0; i < CHUNK_SIZE; i++) {
            sum += chunk[i];
        }
        float average = (float)sum / CHUNK_SIZE;
        
        printf("Worker 2: Media calcolata = %.2f\n", average);
        MPI_Send(&average, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }
    
    MPI_Finalize();
    return 0;
}

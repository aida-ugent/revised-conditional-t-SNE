#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include "tsne.h"

// Function that runs the Barnes-Hut implementation of t-SNE
int main() {

    // Define some variables
	int origN, N, D, no_dims, max_iter, *landmarks, K;
	double perc_landmarks;
	double perplexity, theta, *data;
  int rand_seed = -1;
  TSNE* tsne = new TSNE();

  int num_label_vals, *labels;
  double alpha, beta;

    // Read the parameters and the dataset
	if(tsne->load_data(&data, &labels, &alpha, &beta, &theta, &perplexity, &num_label_vals, &origN, &D, &no_dims, &rand_seed, &max_iter, &K)) {

		// Make dummy landmarks
        N = origN;
        int* landmarks = (int*) malloc(N * sizeof(int));
        if(landmarks == NULL) { printf("Memory allocation failed!\n"); exit(1); }
        for(int n = 0; n < N; n++) landmarks[n] = n;

		// Now fire up the SNE implementation
		double* Y = (double*) malloc(N * no_dims * sizeof(double));
		double* costs = (double*) calloc(N, sizeof(double));
    if(Y == NULL || costs == NULL) { printf("Memory allocation failed!\n"); exit(1); }
		// tsne->run(data, N, D, Y, no_dims, perplexity, theta, rand_seed, false, max_iter);

    int stop_lying_iter = 250;
    int mom_switch_iter = 250;
    tsne->run(data, Y, labels, alpha, beta, theta, perplexity, num_label_vals, N, D, no_dims, rand_seed, 
            false, max_iter, stop_lying_iter, mom_switch_iter, K);

		// Save the results
		tsne->save_data(Y, landmarks, costs, N, no_dims);

        // Clean up the memory
		free(data); data = NULL;
		free(Y); Y = NULL;
		free(costs); costs = NULL;
		free(landmarks); landmarks = NULL;
    }
    delete(tsne);
}

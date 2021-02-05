/*
** EPITECH PROJECT, 2019
** ptrace
** File description:
** HEADER
*/

#ifndef MACHINE_LEARNING
#define MACHINE_LEARNING
#define BIAS_RANGE 6   // from [-3 ; 3]
#define WEIGHT_RANGE 6 // from [-3 ; 3]

#include <math.h>
#include <time.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

typedef struct neural_network_s {
    float **activations;
    float ***weights;
    float **bias;
    float *expectation;
    float **errors;
    float **errors_temp;
    float *costs;
    float **cost_average;
    float cost;
    int *layer_size;
    int layers_nbr;
    int cycles, cycle;
} neural_network_t;

void ia_init(neural_network_t *, int, int, ...);
int ia_read_file(neural_network_t *);
void ia_write_file(neural_network_t *);
void free_all(neural_network_t *);
void print_activations(neural_network_t *);
void print_bias(neural_network_t *);
void print_weights(neural_network_t *);
void print_costs(neural_network_t *);
void randomize(neural_network_t *);
float sigmoid(float);
float sigmoid_derivative(float);
float ai_z(neural_network_t *, int, int);

void ia_compute_cost(neural_network_t *);
void final_cost(neural_network_t *);
void ia_forward_propagation(neural_network_t *);
void ia_backward_propagation(neural_network_t *);
void ia_adjustment(neural_network_t *);

void set_inputs(neural_network_t *);
void set_expectation(neural_network_t *);

#endif

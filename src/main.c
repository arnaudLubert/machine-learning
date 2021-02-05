/*
** EPITECH PROJECT, 2019
** ptrace
** File description:
** HEADER
*/
#include "machine_learning.h"

int main(void) {
    neural_network_t network;

    ia_init(&network, 10, 3, 2, 2, 1);
    network.activations[0][0] = 0.2;
    network.activations[0][1] = 0.3;
    network.bias[1][0] = 0.2;
    network.bias[1][1] = 0.2;
    network.bias[2][0] = 0.2;
    network.weights[1][0][0] = 0.3;
    network.weights[1][0][1] = 0.3;
    network.weights[1][1][0] = 0.5;
    network.weights[1][1][1] = 0.6;
    network.weights[2][0][0] = 0.3;
    network.expectation[0] = 0.0;

    for (int loop = 0; loop != 1; loop++) {
        for (int j = 0; j != network.cycles; j++) {
            printf(" ");
        //    randomize(&network);
            ia_forward_propagation(&network);
            print_activations(&network);
            //ia_compute_cost(&network);
            ia_backward_propagation(&network);
            network.cycle++;
            ia_adjustment(&network);
        }
        //final_cost(&network);
    }
    printf(" ");
    ia_forward_propagation(&network);
    print_activations(&network);
    ia_write_file(&network);

    free_all(&network);

    return 0;
}

// to override
void set_inputs(neural_network_t *network) {

// homme si grand, poilu, sensible
    for (int i = 0; i != network->layer_size[0]; i++)
        network->activations[0][i] = (float)rand() / (float)(RAND_MAX) * 1.0;
}

// to override
void set_expectation(neural_network_t *network) {

    if ((network->activations[0][0] + network->activations[0][1] + 1 - network->activations[0][2]) < 1.5) { // woman
        network->expectation[0] = 0.0;
        network->expectation[1] = 1.0;
    } else { // man
        network->expectation[0] = 1.0;
        network->expectation[1] = 0.0;
    }
}

void ia_init(neural_network_t *network, int cycles, int layers, ...) {
    va_list args;
    int length;

    //if (ia_read_file(&network)) {

//    } else {
        va_start(args, layers);
        network->bias = malloc(sizeof(float *) * layers);
        network->errors = malloc(sizeof(float *) * layers);
        network->weights = malloc(sizeof(float **) * layers);
        network->layer_size = malloc(sizeof(int *) * layers);
        network->errors_temp = malloc(sizeof(float *) * layers);
        network->activations = malloc(sizeof(float *) * layers);
        network->layers_nbr = layers;

        for (int l = 0; l != layers; l++) {
            length = va_arg(args, int);
            network->layer_size[l] = length;
            network->activations[l] = malloc(sizeof(float) * length);

            for (int j = 0; j != length; j++)
                network->activations[l][j] = 0.0;

            if (l != 0) {
                network->bias[l] = malloc(sizeof(float) * length);
                network->errors[l] = malloc(sizeof(float) * length);
                network->weights[l] = malloc(sizeof(float *) * length);
                network->errors_temp[l] = malloc(sizeof(float) * length);

                for (int j = 0; j != length; j++) {
                    network->bias[l][j] = 0.0;
                    network->errors[l][j] = 0.0;
                    network->errors_temp[l][j] = 0.0;
                    network->weights[l][j] = malloc(sizeof(float) * network->layer_size[l - 1]);

                    for (int k = network->layer_size[l - 1] - 1; k != -1; k--)
                        network->weights[l][j][k] = 0.0;
                }
            }
        }
        va_end(args);
//    }

    network->cycles = cycles;
    network->cycle = 0;
    network->expectation = malloc(sizeof(float) * network->layer_size[network->layers_nbr - 1]);
    network->costs = malloc(sizeof(float) * network->layer_size[network->layers_nbr - 1]);
    network->cost_average = malloc(sizeof(float *) * cycles);

    for (int i = cycles - 1; i != -1; i--) {
        network->cost_average[i] = malloc(sizeof(float) * network->layer_size[network->layers_nbr - 1]);
        for (int j = network->layer_size[network->layers_nbr - 1] - 1; j != -1; j--)
            network->cost_average[i][j] = 0.0;
    }

    for (int i = network->layer_size[network->layers_nbr - 1] - 1; i != -1; i--) {
        network->expectation[i] = 0.0;
            network->costs[i] = 0.0;
    }

    ia_read_file(network);
    print_bias(network);
    print_weights(network);
}


/*
layers_nbr
bias_line_size float,float,float,float,float
float,float,float,float,float,                  (weights)
float,float,float,float,float,                  (weights)
float,float,float,float,float,                  (weights)
bias_line_size float,float,float,float,float
float,float,float,float,float,
float,float,float,float,float,
float,float,float,float,float,
*/
int ia_read_file(neural_network_t *network) {
    int n;
    size_t buff_len = 2048;
    char *buff = malloc(buff_len);
    int double_arr_size, arr_size, line_size, integer, i;
    int l = 0;
    int j = 0;
    int k = 0;
    float floating;
    FILE *file = fopen("neurons.ia", "r");

    if ( !file)
        return 0;
    n = getline(&buff, &buff_len, file);

    if (n == 0)
        return 0;
    i = 0;
    memcpy(&integer, &buff[i], sizeof(int));
    //network->activations = malloc(sizeof(float *) * layers);
    printf("%d\n", integer);
    n = getline(&buff, &buff_len, file);

    while (n > 0) {
        i = 0;
        j = 0;
        memcpy(&integer, &buff[i], sizeof(int));
        printf("d:%d\n", integer); // bias length
        i += sizeof(int) + 1;

        while (i < n && j != network->layer_size[l]) {
            memcpy(&network->bias[l][j], &buff[i], sizeof(float));
            i += sizeof(float) + 1;
            j++;
        }
        j = 0;
        n = getline(&buff, &buff_len, file);

        while (l != 0 && j != network->layer_size[l]) {
            i = 0;
            k = 0;

            while (k != network->layer_size[l - 1] && i < n) {
                memcpy(&network->weights[l][j][k], &buff[i], sizeof(float));
                i += sizeof(float) + 1;
                k++;
            }
            n = getline(&buff, &buff_len, file);
            j++;
        }
    //    n = getline(&buff, &buff_len, file);
        l++;
    }

    fclose(file);
    free(buff);
    return 1;
}

void ia_write_file(neural_network_t *network) {
    print_bias(network);
    print_weights(network);
    char buff[sizeof(int)];

    int fd = open("neurons.ia", O_WRONLY | O_CREAT | O_TRUNC, 0644);

    if (fd == -1) {
        fprintf(stderr, "Cannot open file for write operation.\r\n");
        return;
    }

    write(fd, &network->layers_nbr, sizeof(int));
    write(fd, "\n", 1);

    for (int i = sizeof(int) - 1; i != -1; i--)
        write(fd, "\0", 1);
    write(fd, "\n", 1);

    for (int l = 1; l != network->layers_nbr; l++) {
        write(fd, &network->layer_size[l], sizeof(int));

        for (int j = 0; j != network->layer_size[l]; j++) {
            write(fd, ",", 1);
            write(fd, &network->bias[l][j], sizeof(float));
        }
        write(fd, "\n", 1);

        for (int j = 0; j != network->layer_size[l]; j++) {
//            write(fd, &network->weights[l][j][], sizeof(float));
            for (int k = 0; k != network->layer_size[l - 1]; k++) {
                write(fd, &network->weights[l][j][k], sizeof(float));
                write(fd, ",", 1);
            }
            write(fd, "\n", 1);
        }
//        write(fd, "\n", 1);
    }

    close(fd);
}

void free_all(neural_network_t *network) {

    for (int i = 0; i != network->layers_nbr; i++) {
        free(network->activations[i]);

        if (i != 0) {
            for (int j = network->layer_size[i] - 1; j != -1; j--)
                free(network->weights[i][j]);
            free(network->errors_temp[i]);
            free(network->weights[i]);
            free(network->errors[i]);
            free(network->bias[i]);
        }
    }

    for (int i = 0; i != network->cycles; i++)
        free(network->cost_average[i]);

    free(network->bias);
    free(network->errors);
    free(network->weights);
    free(network->layer_size);
    free(network->errors_temp);
    free(network->activations);
    free(network->expectation);
    free(network->cost_average);
    free(network->costs);
}

void randomize(neural_network_t *network) {
    srand(time(NULL));

    for (int i = 0; i != network->layers_nbr - 1; i++)
        for (int j = 0; j != network->layer_size[i]; j++)
            for (int k = 0; k != network->layer_size[i + 1]; k++)
                network->weights[i][j][k] = (float)rand() / (float)(RAND_MAX) * WEIGHT_RANGE - WEIGHT_RANGE / 2;

    for (int i = 1; i != network->layers_nbr; i++)
        for (int j = 0; j != network->layer_size[i]; j++)
            network->bias[i][j] = (float)rand() / (float)(RAND_MAX) * BIAS_RANGE - BIAS_RANGE / 2;
}

void print_activations(neural_network_t *network) {
    for (int i = network->layers_nbr - 1; i != network->layers_nbr; i++) {
        for (int j = 0; j != network->layer_size[i]; j++)
            printf("%f ", network->activations[i][j]);
        printf("\r\n");
    }
}

void print_bias(neural_network_t *network) {
    for (int i = 1; i != network->layers_nbr; i++) {
        for (int j = 0; j != network->layer_size[i]; j++) {
            printf("%f ", network->bias[i][j]);
        }
        printf("\r\n");
    }
}

void print_weights(neural_network_t *network) {
    for (int i = 1; i != network->layers_nbr; i++) {
        printf("Level %d\r\n", i);
        for (int j = 0; j != network->layer_size[i]; j++) {
            printf("%d\t", j);
            for (int k = 0; k != network->layer_size[i - 1]; k++)
                printf("%f ", network->weights[i][j][k]);
            printf("\r\n");
        }
        printf("\r\n");
    }
}

void print_costs(neural_network_t *network) {
    /*printf("Cost:\r\n");

    for (int i = 0; i != network->layer_size[network->layers_nbr - 1]; i++)
            printf("%f\t", network->costs[i]);
    printf("\r\n");*/
}

float sigmoid(float x) {
    return 1 / (1 + exp(-x));
/*
    f(x) = x / (1 + abs(x))

    if (x < 0) // absolute
        return x / (1 - x);
    else
        return x / (1 + x);
*/
}

float sigmoid_derivative(float x) {
    x = sigmoid(x);
    return x * (1 - x);
}

float ai_z(neural_network_t *network, int l, int j) {
    float z = 0.0;

    for (int k = network->layer_size[l - 1] - 1; k != -1; k--)
        z += network->weights[l][j][k] * network->activations[l - 1][k] + network->bias[l][j];

    return z;
}

void ia_compute_cost(neural_network_t *network) {
    int len = network->layer_size[network->layers_nbr - 1];

    for (int i = 0; i != len; i++)
        network->cost_average[network->cycle][i] = powf(network->expectation[i] - network->activations[network->layers_nbr - 1][i], 2.0);
}

void final_cost(neural_network_t *network) {
    float node_cost;

    network->cost = 0.0;

    for (int i = network->layer_size[network->layers_nbr - 1] - 1; i != -1; i--) {
        node_cost = 0.0;

        for (int j = network->cycles - 1; j != -1; j--)
            node_cost += network->cost_average[j][i];

        node_cost /= network->cycle;
        network->costs[i] = node_cost;
        network->cost += node_cost;
    }

    printf("Average cost: %f\r\n", network->cost);
}

void ia_forward_propagation(neural_network_t *network) {
    for (int l = 1; l != network->layers_nbr; l++)
        for (int j = 0; j != network->layer_size[l]; j++)
            network->activations[l][j] = sigmoid(ai_z(network, l, j));
}

// error computation
void ia_backward_propagation(neural_network_t *network) {
    int l = network->layers_nbr - 1;
    float sig, delta_cost;

    for (int j = 0; j != network->layer_size[l]; j++) {
        network->errors_temp[l][j] = network->errors[l][j];

        if (network->cycle == 0)
            network->errors[l][j] = (network->activations[l][j] - network->expectation[j]) * sigmoid_derivative(ai_z(network, l, j));
        else
            network->errors[l][j] += (network->activations[l][j] - network->expectation[j]) * sigmoid_derivative(ai_z(network, l, j));
    }
    l--;

    // backward propagation of the error
    while (l != 0) {
        for (int j = 0; j != network->layer_size[l]; j++) {
            network->errors_temp[l][j] = 0.0;
            sig = sigmoid_derivative(ai_z(network, l, j));

            for (int a = 0; a != network->layer_size[l + 1]; a++) {
                delta_cost = 0.0;

                for (int b = 0; b != network->layer_size[l]; b++)
                    delta_cost += network->errors_temp[l + 1][a] * network->weights[l + 1][a][b];
                network->errors_temp[l][j] += delta_cost * sig;
            }

            if (network->cycle == 0)
                network->errors[l][j] = network->errors_temp[l][j];
            else
                network->errors[l][j] += network->errors_temp[l][j];
        }
        l--;
    }
}

void ia_adjustment(neural_network_t *network) {
    for (int l = 1; l != network->layers_nbr; l++)
        for (int j = 0; j != network->layer_size[l]; j++) {
            for (int k = 0; k != network->layer_size[l - 1]; k++)
                network->weights[l][j][k] = network->weights[l][j][k] - network->errors[l][j] * network->activations[l - 1][k];
            network->bias[l][j] -= network->errors[l][j];
        }
    network->cycle = 0;
}

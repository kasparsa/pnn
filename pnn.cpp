#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <pliba/pliba.h>
#include <pregex/pregex.h>
#include <cuda_runtime.h>
#include <iostream>
#include <SDL2/SDL.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include "kernel_feed_forward.h"


int esample_input_count = -1;		/* Input field count in dataset */
int esample_output_count = -1;	/* Output field count in dataset */
int eLayer_Count = -1;
int eprm_max_neuron_count_in_layer = -1;
int esample_count = -1;
int esensor_max_count = -1;     /* Max sensor count is max neuron count ^ 2 */

#define INPUTS(sample, input) inputs[(sample) * esample_input_count + (input)]
#define OUTPUTS(sample, output) outputs[(sample) * esample_output_count + (output)]
#define NC(layer) neuron_count_per_layer[layer]
#define WEIGHTED_SUM(layer,neuron) weighted_sum[layer*eprm_max_neuron_count_in_layer+neuron]
#define OUTPUT_AXON_VAL_DENORMALIZED(layer,neuron) output_axon_val_denormalized[layer*eprm_max_neuron_count_in_layer+neuron]
#define OUTPUT_AXON_VAL(layer,neuron) output_axon_val[(layer) * eprm_max_neuron_count_in_layer + (neuron)]
#define SENSOR_COUNT(layer,neuron) sensor_count[layer*eprm_max_neuron_count_in_layer+neuron]
#define DELTA(layer, neuron, sample) delta[(((layer) * eprm_max_neuron_count_in_layer * esample_count) + ((neuron) * esample_count) + (sample))]
#define WEIGHTED_SUM_SAMPLE(layer, neuron, sample) weighted_sum_sample[(((layer) * eprm_max_neuron_count_in_layer * esample_count) + ((neuron) * esample_count) + (sample))]
#define OUTPUT_AXON_SAMPLE(layer, neuron, sample) output_axon_sample[(((layer) * eprm_max_neuron_count_in_layer * esample_count) + ((neuron) * esample_count) + (sample))]
#define SENSOR(layer, neuron, sample) sensor[(((layer) * eprm_max_neuron_count_in_layer * esensor_max_count) + ((neuron) * esensor_max_count) + (sample))]
#define WEIGHT(layer, neuron, sample) weight[(((layer) * eprm_max_neuron_count_in_layer * esensor_max_count) + ((neuron) * esensor_max_count) + (sample))]
#define PREV_WEIGHT(layer, neuron, sample) prev_weight[(((layer) * eprm_max_neuron_count_in_layer * esensor_max_count) + ((neuron) * esensor_max_count) + (sample))]
#define PREV_GRADIENT(layer, neuron, sample) prev_gradient[(((layer) * eprm_max_neuron_count_in_layer * esensor_max_count) + ((neuron) * esensor_max_count) + (sample))]
#define GRADIENT(layer, neuron, sample) gradient[(((layer) * eprm_max_neuron_count_in_layer * esensor_max_count) + ((neuron) * esensor_max_count) + (sample))]
#define RPROP_DELTA(layer, neuron, sample) rprop_delta[(((layer) * eprm_max_neuron_count_in_layer * esensor_max_count) + ((neuron) * esensor_max_count) + (sample))]
#define MEAN_SQUARE_GRADIENT(layer, neuron, sample) mean_square_gradient[(((layer) * eprm_max_neuron_count_in_layer * esensor_max_count) + ((neuron) * esensor_max_count) + (sample))]
#define M(layer, neuron, sample) m[(((layer) * eprm_max_neuron_count_in_layer * esensor_max_count) + ((neuron) * esensor_max_count) + (sample))]
#define WEIGHT_AGE(layer, neuron, sample) weight_age[(((layer) * eprm_max_neuron_count_in_layer * esensor_max_count) + ((neuron) * esensor_max_count) + (sample))]


class nnet;

nnet *Net;

#define EPSILON_MATCH 0.09  // define a suitable small value

int areFloatsEqual1(float a, float b);
void Init();
void Run(int sample, float e, float r );
void DrawMsqe(int counter, float msqe );
void WaitInput();
void Clear();
void Render();
void Destroy();

#define UNIVERSE_WIDTH 2000
#define UNIVERSE_HEIGHT 1000

int pX[UNIVERSE_WIDTH*2];
int pY[UNIVERSE_HEIGHT*2];

SDL_Event sdlEvent;
SDL_Window *sdlWindow;
SDL_Renderer *sdlRenderer;
SDL_Texture *sdlTexture;
int quit = 0;

#define DDEBUG 1
// #define ENABLE


#define EPSILON 0.000001  // define a suitable small value

#define ACTIV_FN_SIGMOID   0
#define ACTIV_FN_TANH      1
#define ACTIV_FN_GAUSSIAN  2
#define ACTIV_FN_SIN       3

#define MAX_LAYER_COUNT 5
#define NEURON_COUNT_PER_LAYER 10
#define DETECTION_ITERATION_COUNT 100

#define LAYER_COUNT 10
#define LAYER_NEURON_COUNT 100

int *neuron_count_per_layer;
int *sensor_count;		/* input value count*/
float *inputs;
float *outputs;
float *output_axon_val;
float *output_axon_val_denormalized;
float *weighted_sum;
float *sensor;		/* dendrite value */
float *weight;		   /* dendrite weight */
float *gradient;       /* dendrite weight */
float *prev_gradient;
float *prev_weight;
float *rprop_delta;
float *weight_age;
float *mean_square_gradient;
float *m;
float *delta;           /*delta storage for each sample for gradient calculation*/
float *output_axon_sample;  /*axon storage for each sample for gradient calculation*/
float *weighted_sum_sample;
float devi_squarsum;

float rprop_delta_plus;
float rprop_delta_minus;
float rprop_delta_max;
float rprop_delta_min;



// GPU pointers (prefixed with d_ for clarity)
int *d_prm_activation_fn;
int *d_neuron_count_per_layer;
int *d_sensor_count;
float *d_inputs;
float *d_outputs;
float *d_output_axon_val;
float *d_output_axon_val_denormalized;
float *d_weighted_sum;
float *d_sensor;
float *d_weight;
float *d_gradient;
float *d_prev_gradient;
float *d_prev_weight;
float *d_rprop_delta;
float *d_weight_age;
float *d_mean_square_gradient;
float *d_m;
float *d_delta;
float *d_output_axon_sample;
float *d_weighted_sum_sample;
float *d_devi_squarsum;

class nndataset;


void create_input_sensors(int LYI, int LNI, int neuron_input_count, int sample_count );

void layer_construct();
void layer_destruct();

void configure_neurons(int LYI, int sensor_count, nndataset *dataset );
void apply_settings(int LYI, nndataset *dataset );

int sign(float x);
float weight_sum(int LYI, int LNI);

float transferfn_sigmoid( float w_sum );
float transferfn_tanh( float w_sum );
float transferfn_gaussian( float w_sum );
float transferfn_sinusoid( float w_sum );

class nndataset
{
private:
    void SwapInput( int src, int dst );
public:
    nndataset();
    ~nndataset();

    int sample_count;			/* Dataset sample count */
    int sample_input_count;		/* Input field count in dataset */
    int sample_output_count;	/* Output field count in dataset */

    char fileNameOrig[256];         /* Loaded sample filename */
    char fileNameWoExt[256];         /* Loaded sample filename */
    char fileNameCSV[256];           /* Loaded sample filename */
    char fileNamePNN[256];           /* Loaded sample filename */

    // nndata *samples;			/* Dataset samples */
    // nndata *tmp;

    float dataset_min;
    float dataset_max;

    float dataset_input_min;
    float dataset_input_max;

    float dataset_output_min;
    float dataset_output_max;

    int prm_net_layer_type_rnn;
    int prm_net_architecture;   /* Net architecture. Layer count. Neuron count in hidden layers */
    int prm_training_algorithm; /* 0 - incremental, 1 - batch, 2 - rprop 3 - RMSprop */

    bool prm_kubernetes;

    float prm_rprop_delta_plus;
    float prm_rprop_delta_minus;
    float prm_rprop_delta_max;
    float prm_rprop_delta_min;

    float prm_normalize_max;
    float prm_normalize_min;


    int prm_total_neuron_count;
    int prm_max_neuron_count_in_layer;
    int prm_net_layer_count;
    int *prm_net_layer;

    int prm_report;
    int prm_print;			    /* 0 - Silent, 1 - Epoch Msqe, 2 - Detailed output */
    int prm_print_input;		/* 0 - Silent, 1 - Epoch Msqe, 2 - Detailed output */
    int prm_print_output;		/* 0 - Silent, 1 - Epoch Msqe, 2 - Detailed output */
    int prm_print_net_arch;
    int prm_stagnation;			/* Learning process iteartion count since last Msqe decreasing  */
    int prm_stuck;			    /* Learning process iteartion count without frozen Msqe */
    int prm_activation_fn;  	/* Activation fuction type */
    float prm_steepness;		/* Trashold */
    float prm_learning_rate_for_decay;	/* Learning rate */
    float prm_lambda_for_decay;           /* lambda = regularization strength */
    float prm_bias;			    /* Bias value */
    float prm_momentum;			/* Momentum step */
    float prm_final_msqe;		/* Expected Msqe */
    int prm_final_epo;			/* Maximumum learning iteration epoch count */
//     int prm_train_incremental;  /* Weights are updated after each training pattern */
//     int prm_train_batch;        /* Weights are updated after calculating the mean square error for the whole training set */
    float prm_bias_neuron_val;  /* Bias neuron output value */


    void ProcessFileName(const char *fileName);
    void LoadDataset( const char *fileName );
    void LoadProcessDataset( const char *fileName );


    void NormalizeInputData( float a, float b );
    float DenormalizeOutput( float value, float a, float b);


    void PrintInputData();


    int GetPrintLevel();
    void SetPrintLevel( int debug_level );
};

class nnet
{
public:
    int msqeDrawCounter;

    bool was_shoot;
    float PrevReported_MsqE;
    float MsqE;
    float MsqELowest;
    float MsqEPrevLowest;
    float MsqEHighest;
    float reward;

    int ItMsqEHighest;
    int ItMsqELowest;

    float LearningRatio;

    int Iteration;
    int ReportCounter;

    float MsqELast;

    float MsqEDelta; // Difference between previous and currend mse

    bool Learning;

    /*Counts how many iterations MsqE remains unchanged*/
    int StuckCounter;

    /*Counts of iterations since last LowestMsqE decrease*/
    int StagnationCounter;

    int ConvergeCounter;
    int DivergeCounter;

    int Layer_Count;

    int closeMatchCnt;

    float Learn();
    void Process();

public:
    nnet();
    ~nnet();

    nndataset *ds;

    bool converging;

    void allocMemory();

    bool shouldContinueLearning();

    void processNeuralNet();
    void ffw_processNeuralNet();

    void processLayers();

    void postProcessIteration();
    void reportStatus();
    void trackConvergence();

    void LoadDataset( const char *fileName );
    void LoadDatasetData( const char *fileName );

    void SaveNet(const char *fileName);
    void LoadNet(const char *fileName);

    void CreateLayers( int Count );

    void SetupLayers( int layer_count, int *neuron_count_in_layer );

    float compute_loss(int neuron);
    int areFloatsEqual(float a, float b);

    void set_initial_input_val( int sample );

    float LearStart();

    void PrintConclusion();
    void PrintProcessingConclusion();

    void printStopConditions();
    void printNetworkArchitecture();
    void printSampleOutput(int sample, int &under_ten, FILE *file);
    void printSampleOutputProcess(int sample);
    void interfaceDenormalization(int sample);

    void drawSampleOutput();
    void drawSampleOutputProcessing( int sample );
};



char fileName[256];
bool Learn = false;
bool Process = false;
bool Continue = false;
bool Detect = false;
int detectNeuronCountUpTo = 0;


void configure_neurons(int LYI, int sensor_count, nndataset *dataset )
{
    for ( int i = 0; i < NC(LYI); i++ ) {

        create_input_sensors(LYI, i, sensor_count, dataset->sample_count);

    }
}


void apply_settings(int LYI, nndataset* dataset)
{
    rprop_delta_plus= dataset->prm_rprop_delta_plus;
    rprop_delta_minus = dataset->prm_rprop_delta_minus;
    rprop_delta_min = dataset->prm_rprop_delta_min;
    rprop_delta_max = dataset->prm_rprop_delta_max;
}

void create_input_sensors(int LYI, int LNI, int neuron_input_count, int sample_count )
{
    SENSOR_COUNT(LYI,LNI) = neuron_input_count;

    for ( int i = 0; i < sample_count; i++ ) {
        DELTA(LYI,LNI,i) = 0;
        OUTPUT_AXON_SAMPLE(LYI,LNI,i) = 0;
        WEIGHTED_SUM_SAMPLE(LYI,LNI,i) = 0;
    }

    for ( int i = 0; i < SENSOR_COUNT(LYI,LNI); i++ ) {
        SENSOR(LYI,LNI,i) = ((double)rand() / RAND_MAX) * 0.8 - 0.4;
        WEIGHT(LYI,LNI,i) = ((double)rand() / RAND_MAX) * 0.8 - 0.4;
        PREV_WEIGHT(LYI,LNI,i) = ((double)rand() / RAND_MAX) * 0.8 - 0.4;;
        GRADIENT(LYI,LNI,i) = ((double)rand() / RAND_MAX) * 0.8 - 0.4;;
        PREV_GRADIENT(LYI,LNI,i) = ((double)rand() / RAND_MAX) * 0.8 - 0.4;
        MEAN_SQUARE_GRADIENT(LYI,LNI,i)= 0;//((double)rand() / RAND_MAX) * 0.8 - 0.4;
        M(LYI,LNI,i) = 0;
        RPROP_DELTA(LYI,LNI,i) = 0.125;
        WEIGHT_AGE(LYI,LNI,i) = 0;
    }
}

int sign(float x) {
    if (x > 0.0f) return 1;
    if (x < 0.0f) return -1;
    return 0;
}

float weight_sum(int LYI, int LNI)
{
    WEIGHTED_SUM(LYI,LNI) = 0;

    for ( int i = 0; i < SENSOR_COUNT(LYI,LNI); i++ ) {
        WEIGHTED_SUM(LYI,LNI) += SENSOR(LYI,LNI,i) * WEIGHT(LYI,LNI,i);
    }

    return WEIGHTED_SUM(LYI,LNI);
}

float transferfn_sigmoid( float w_sum )
{
    return 1 / ( 1 + powf( M_E, -w_sum ) );
}


float transferfn_tanh(float w_sum)
{
    return tanhf(w_sum);
}

float transferfn_gaussian(float w_sum)
{
    return exp(-w_sum * w_sum );
}

float transferfn_sinusoid( float w_sum )
{
    return sinf(w_sum);
}

nnet::nnet()
{
    ds = nullptr;

    MsqE = 0;
    PrevReported_MsqE = 0;
    was_shoot = 0;
    Iteration = 0;
    ReportCounter = 0;

    MsqELowest = 0;
    MsqEHighest = 0;
    MsqEDelta = 0;
    MsqELast = 0;
    MsqEPrevLowest = 0;

    reward = 0;

    StagnationCounter = 0;

    converging = false;

    Layer_Count = 0;

    ds = new nndataset;

    ConvergeCounter = 1;
    DivergeCounter = 1;

    Learning = false;

    msqeDrawCounter = 0;




    ItMsqEHighest = 0;
    ItMsqELowest = 0;

    LearningRatio = 0;

    StuckCounter = 0;

    closeMatchCnt = -1;
}


nnet::~nnet()
{
    // printf("Net Destructor\n");
    delete ds;
    //if (ly) delete ly;
    // printf("Net Destructor Done!\n");
    free(neuron_count_per_layer);
    free(sensor_count);
    free(output_axon_val_denormalized);
    free(output_axon_val);
    free(weighted_sum);
    free(sensor);
    free(weight);
    free(gradient);
    free(prev_gradient);
    free(prev_weight);
    free(rprop_delta);
    free(weight_age);
    free(mean_square_gradient);
    free(m);
    free(delta);
    free(output_axon_sample);
    free(weighted_sum_sample);
}


void nnet::LoadDataset( const char *fileName )
{
    ds->LoadDataset( fileName );
}


void nnet::LoadDatasetData(const char* fileName)
{
    if ( DDEBUG ) printf("Loading input data\n");

    ds->LoadProcessDataset( fileName );

    if ( DDEBUG ) printf("Done!\n");

    if ( ds->prm_print_input == 1 ) ds->PrintInputData();
}

void nnet::CreateLayers( int Count )
{
    Layer_Count = Count;
    eLayer_Count = Layer_Count;

    printf("ALLOCATING MEMORY 0\n");

    neuron_count_per_layer = (int*)calloc(eLayer_Count,sizeof(int));
}

void nnet::allocMemory()
{
    printf("ALLOCATING MEMORY 1\n");

    weighted_sum = (float*)calloc(eLayer_Count*eprm_max_neuron_count_in_layer,sizeof(float));

    output_axon_val_denormalized = (float*)calloc(eLayer_Count*eprm_max_neuron_count_in_layer,sizeof(float));

    output_axon_val= (float*)calloc(eLayer_Count*eprm_max_neuron_count_in_layer,sizeof(float));

    sensor_count = (int*)calloc(eLayer_Count*eprm_max_neuron_count_in_layer,sizeof(int));

    delta = (float*)calloc(eLayer_Count * eprm_max_neuron_count_in_layer * esample_count,sizeof(float));

    output_axon_sample = (float*)calloc(eLayer_Count * eprm_max_neuron_count_in_layer * esample_count,sizeof(float));

    weighted_sum_sample = (float*)calloc(eLayer_Count * eprm_max_neuron_count_in_layer * esample_count,sizeof(float));

    sensor = (float*)calloc(eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count,sizeof(float));

    weight = (float*)calloc(eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count,sizeof(float));

    prev_weight = (float*)calloc(eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count,sizeof(float));

    gradient = (float*)calloc(eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count,sizeof(float));

    prev_gradient = (float*)calloc(eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count,sizeof(float));

    mean_square_gradient = (float*)calloc(eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count,sizeof(float));

    m = (float*)calloc(eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count,sizeof(float));

    rprop_delta = (float*)calloc(eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count,sizeof(float));

    weight_age = (float*)calloc(eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count,sizeof(float));
}

void nnet::SetupLayers( int layer_count, int *neuron_count_in_layer )
{
    /* Input layer neuron count */
    NC(0) = ds -> sample_input_count;

    /* Hidden layer neuron count */
    for ( int i = 1; i <= layer_count; i++ ) NC(i) = neuron_count_in_layer[i];

    /* Output layer neuron count */
    NC(Layer_Count - 1) = ds -> sample_output_count;

    allocMemory();

    for ( int i = 0; i < Layer_Count; i++ ) {

        if ( i == 0 ) configure_neurons(i, 1, ds );
        else configure_neurons(i, NC(i - 1), ds );

        apply_settings(i, ds );

    }
}


void nnet::set_initial_input_val( int sample )
{
    for ( int i = 0; i < NC(0); i++ ) {

        SENSOR(0,i,0) = INPUTS(sample,i);

    }
}

int areFloatsEqual(float a, float b) {
    return fabsf(a - b) < EPSILON;
}


float nnet::Learn()
{
    if (ds->prm_print_net_arch) {
        printNetworkArchitecture();
    }

    MsqE = 1000000000;

    allocMemoryCUDA();

    initializeMemoryCUDA();

    while ( shouldContinueLearning() ) {

        while (shouldContinueLearning()) {

            Iteration++;
            ReportCounter++;

            processNeuralNet();

            if (ReportCounter == ds->prm_report) reportStatus();

            trackConvergence();
        }

        if (StagnationCounter >= ds->prm_stagnation ) {
            StagnationCounter = 0;
        }
    }

    freeMemoryCUDA();

    Learning = false;

    return MsqE;
}

// float nnet::Learn()
// {
//     if (ds->prm_print_net_arch) {
//         printNetworkArchitecture();
//     }
//
//     MsqE = 1000000000;
//
//     while ( shouldContinueLearning() ) {
//
//         while (shouldContinueLearning()) {
//
//             Iteration++;
//             ReportCounter++;
//
//             processNeuralNet();
//
//             if (ReportCounter == ds->prm_report) reportStatus();
//
//             trackConvergence();
//         }
//
//         if (StagnationCounter >= ds->prm_stagnation ) {
//             StagnationCounter = 0;
//         }
//     }
//
//     Learning = false;
//
//     return MsqE;
// }

bool nnet::shouldContinueLearning() {
    return Learning &&
           MsqE > ds->prm_final_msqe &&
           StagnationCounter < ds->prm_stagnation &&
           StuckCounter < ds->prm_stuck &&
           Iteration < ds->prm_final_epo;
}


void nnet::Process()
{
    printf("Processing :)))))))))))))))))))))))))\n");

    PrintProcessingConclusion();

    // PrintConclusion();
}






void allocMemoryCUDA()
{
    printf("ALLOCATING GPU MEMORY\n");

    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&d_prm_activation_fn, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_neuron_count_per_layer, eLayer_Count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sensor_count, eLayer_Count * eprm_max_neuron_count_in_layer * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_inputs, esample_count * esample_input_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_outputs, esample_count * esample_output_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_axon_val, eLayer_Count * eprm_max_neuron_count_in_layer * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_axon_val_denormalized, eLayer_Count * eprm_max_neuron_count_in_layer * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weighted_sum, eLayer_Count * eprm_max_neuron_count_in_layer * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sensor, eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weight, eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gradient, eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_prev_gradient, eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_prev_weight, eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rprop_delta, eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weight_age, eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean_square_gradient, eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m, eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_delta, eLayer_Count * eprm_max_neuron_count_in_layer * esample_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_axon_sample, eLayer_Count * eprm_max_neuron_count_in_layer * esample_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weighted_sum_sample, eLayer_Count * eprm_max_neuron_count_in_layer * esample_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_devi_squarsum, sizeof(float)));
}

void initializeMemoryCUDA()
{
    printf("INITIALIZING GPU MEMORY\n");

    // Zero-initialize arrays (mimicking calloc)
    CUDA_CHECK(cudaMemset(d_prm_activation_fn, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_neuron_count_per_layer, 0, eLayer_Count * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_sensor_count, 0, eLayer_Count * eprm_max_neuron_count_in_layer * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_inputs, 0, esample_count * esample_input_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_outputs, 0, esample_count * esample_output_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_output_axon_val, 0, eLayer_Count * eprm_max_neuron_count_in_layer * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_output_axon_val_denormalized, 0, eLayer_Count * eprm_max_neuron_count_in_layer * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_weighted_sum, 0, eLayer_Count * eprm_max_neuron_count_in_layer * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sensor, 0, eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_weight, 0, eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_gradient, 0, eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_prev_gradient, 0, eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_prev_weight, 0, eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_rprop_delta, 0, eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_weight_age, 0, eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_mean_square_gradient, 0, eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_m, 0, eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_delta, 0, eLayer_Count * eprm_max_neuron_count_in_layer * esample_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_output_axon_sample, 0, eLayer_Count * eprm_max_neuron_count_in_layer * esample_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_weighted_sum_sample, 0, eLayer_Count * eprm_max_neuron_count_in_layer * esample_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_devi_squarsum, 0, sizeof(float)));
}


void copyHostToDeviceCUDA()
{
    // printf("COPYING HOST DATA TO GPU MEMORY\n");

    // Copy input data from host to device
    CUDA_CHECK(cudaMemcpy(d_neuron_count_per_layer, neuron_count_per_layer, eLayer_Count * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sensor_count, sensor_count, eLayer_Count * eprm_max_neuron_count_in_layer * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inputs, inputs, esample_count * esample_input_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_outputs, outputs, esample_count * esample_output_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output_axon_val, output_axon_val, eLayer_Count * eprm_max_neuron_count_in_layer * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output_axon_val_denormalized, output_axon_val_denormalized, eLayer_Count * eprm_max_neuron_count_in_layer * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weighted_sum, weighted_sum, eLayer_Count * eprm_max_neuron_count_in_layer * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sensor, sensor, eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, weight, eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gradient, gradient, eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prev_gradient, prev_gradient, eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prev_weight, prev_weight, eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rprop_delta, rprop_delta, eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight_age, weight_age, eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mean_square_gradient, mean_square_gradient, eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_m, m, eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_delta, delta, eLayer_Count * eprm_max_neuron_count_in_layer * esample_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output_axon_sample, output_axon_sample, eLayer_Count * eprm_max_neuron_count_in_layer * esample_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weighted_sum_sample, weighted_sum_sample, eLayer_Count * eprm_max_neuron_count_in_layer * esample_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_devi_squarsum, &devi_squarsum, sizeof(float), cudaMemcpyHostToDevice));
}


void copyDeviceToHostCUDA()
{
    // printf("COPYING GPU MEMORY TO HOST\n");

    //Copy data from device to host
    CUDA_CHECK(cudaMemcpy(neuron_count_per_layer, d_neuron_count_per_layer, eLayer_Count * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sensor_count, d_sensor_count, eLayer_Count * eprm_max_neuron_count_in_layer * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(inputs, d_inputs, esample_count * esample_input_count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(outputs, d_outputs, esample_count * esample_output_count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(output_axon_val, d_output_axon_val, eLayer_Count * eprm_max_neuron_count_in_layer * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(output_axon_val_denormalized, d_output_axon_val_denormalized, eLayer_Count * eprm_max_neuron_count_in_layer * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(weighted_sum, d_weighted_sum, eLayer_Count * eprm_max_neuron_count_in_layer * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sensor, d_sensor, eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(weight, d_weight, eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gradient, d_gradient, eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(prev_gradient, d_prev_gradient, eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(prev_weight, d_prev_weight, eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(rprop_delta, d_rprop_delta, eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(weight_age, d_weight_age, eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mean_square_gradient, d_mean_square_gradient, eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(m, d_m, eLayer_Count * eprm_max_neuron_count_in_layer * esensor_max_count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(delta, d_delta, eLayer_Count * eprm_max_neuron_count_in_layer * esample_count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(output_axon_sample, d_output_axon_sample, eLayer_Count * eprm_max_neuron_count_in_layer * esample_count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(weighted_sum_sample, d_weighted_sum_sample, eLayer_Count * eprm_max_neuron_count_in_layer * esample_count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&devi_squarsum, d_devi_squarsum, sizeof(float), cudaMemcpyDeviceToHost));
}


void freeMemoryCUDA()
{
    printf("FREEING GPU MEMORY\n");

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_prm_activation_fn));
    CUDA_CHECK(cudaFree(d_neuron_count_per_layer));
    CUDA_CHECK(cudaFree(d_sensor_count));
    CUDA_CHECK(cudaFree(d_inputs));
    CUDA_CHECK(cudaFree(d_outputs));
    CUDA_CHECK(cudaFree(d_output_axon_val));
    CUDA_CHECK(cudaFree(d_output_axon_val_denormalized));
    CUDA_CHECK(cudaFree(d_weighted_sum));
    CUDA_CHECK(cudaFree(d_sensor));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_gradient));
    CUDA_CHECK(cudaFree(d_prev_gradient));
    CUDA_CHECK(cudaFree(d_prev_weight));
    CUDA_CHECK(cudaFree(d_rprop_delta));
    CUDA_CHECK(cudaFree(d_weight_age));
    CUDA_CHECK(cudaFree(d_mean_square_gradient));
    CUDA_CHECK(cudaFree(d_m));
    CUDA_CHECK(cudaFree(d_delta));
    CUDA_CHECK(cudaFree(d_output_axon_sample));
    CUDA_CHECK(cudaFree(d_weighted_sum_sample));
    CUDA_CHECK(cudaFree(d_devi_squarsum));
}


void nnet::processNeuralNet()
{
    devi_squarsum = 0;

    // #pragma omp parallel for reduction(+:devi_squarsum) num_threads(16)
    for ( int sample = 0; sample < ds->sample_count; sample++) {

        /* Input data loading */
        for ( int i = 0; i < NC(0); i++ ) SENSOR(0,i,0) = INPUTS(sample,i);

        /* Feed forward */
        for (int i = 0; i < Layer_Count; i++) {
            // If it's not the first layer, feed forward
            if (i > 0) {

                    // If feed forward layer
                    for (int n = 0; n < NC(i); n++) {
                        for (int s = 0; s < SENSOR_COUNT(i,n); s++) {
                            SENSOR(i,n,s) = OUTPUT_AXON_VAL(i-1,s);
                        }
                    }
            }

            /* Process the current layer - Weighted Sum and Activation */
            for ( int n = 0; n < NC(i); n++ ) {

                WEIGHTED_SUM(i,n) = 0;

                for ( int k = 0; k < SENSOR_COUNT(i,n); k++ ) {
                    WEIGHTED_SUM(i,n) += SENSOR(i,n,k) * WEIGHT(i,n,k);
                }

                switch ( ds->prm_activation_fn ) {
                    case ACTIV_FN_SIGMOID:
                            OUTPUT_AXON_VAL(i,n) = 1 / ( 1 + powf( M_E, -WEIGHTED_SUM(i,n) ) );
                            OUTPUT_AXON_SAMPLE(i,n,sample) = OUTPUT_AXON_VAL(i,n);
                            WEIGHTED_SUM_SAMPLE(i,n,sample) = WEIGHTED_SUM(i,n);
                        break;
                    case ACTIV_FN_TANH:
                            OUTPUT_AXON_VAL(i,n) = tanhf( WEIGHTED_SUM(i,n) );
                            OUTPUT_AXON_SAMPLE(i,n,sample) = OUTPUT_AXON_VAL(i,n);
                            WEIGHTED_SUM_SAMPLE(i,n,sample) = WEIGHTED_SUM(i,n);
                        break;
                    case ACTIV_FN_GAUSSIAN:
                            OUTPUT_AXON_VAL(i,n) = exp(-WEIGHTED_SUM(i,n) * WEIGHTED_SUM(i,n) );
                            OUTPUT_AXON_SAMPLE(i,n,sample) = OUTPUT_AXON_VAL(i,n);
                            WEIGHTED_SUM_SAMPLE(i,n,sample) = WEIGHTED_SUM(i,n);
                        break;
                    case ACTIV_FN_SIN:
                            OUTPUT_AXON_VAL(i,n) = sinf( WEIGHTED_SUM(i,n) );
                            OUTPUT_AXON_SAMPLE(i,n,sample) = OUTPUT_AXON_VAL(i,n);
                            WEIGHTED_SUM_SAMPLE(i,n,sample) = WEIGHTED_SUM(i,n);
                        break;
                    default:
                        printf("ERROR: unknown activation function\n");
                        throw;
                }
            }
        }


        // float deviation_sum = 0;
        float sample_squared_error = 0;
        for ( int j = 0; j < ds->sample_output_count; j++) {
            float axon_output = OUTPUT_AXON_VAL(Layer_Count - 1,j);
            float sample_output = OUTPUTS(sample,j);
            // Calculate the square of the difference
            // deviation_sum += (axon_output - sample_output) * (axon_output - sample_output);
            sample_squared_error += (axon_output - sample_output) * (axon_output - sample_output);
        }

        // Average the sum of squared deviations
        // deviation_sum /= ds->sample_output_count;
        devi_squarsum += sample_squared_error;

        // devi_squarsum += powf(deviation_sum, 2);
    }

    // Compute final MSE
    // MsqE = devi_squarsum / ds->sample_count;
    MsqE = devi_squarsum / (ds->sample_count * ds->sample_output_count);

    #ifdef ENABLE
    #pragma omp parallel for num_threads(16)
    #endif
    for (int sample = 0; sample < ds->sample_count; sample++) {
        for (int layer = Layer_Count - 1; layer >= 0; layer--) {
            for (int neuron = 0; neuron < NC(layer); neuron++) {

                float derivative = 0;
                float value;

                switch ( ds->prm_activation_fn ) {
                    case ACTIV_FN_SIGMOID:
                        value = WEIGHTED_SUM_SAMPLE(layer,neuron,sample);
                        derivative = (1 / ( 1 + powf( M_E, -value ) )) * ( 1 - (1 / ( 1 + powf( M_E, -value ) )) );
                        break;
                    case ACTIV_FN_TANH:
                        value = tanhf(WEIGHTED_SUM_SAMPLE(layer,neuron,sample));
                        derivative = 1.0f - value * value;
                        break;
                    case ACTIV_FN_GAUSSIAN:
                        value = WEIGHTED_SUM_SAMPLE(layer,neuron,sample);
                        derivative = -2 * value * exp( -value * value );;
                        break;
                    case ACTIV_FN_SIN:
                        derivative = cosf(WEIGHTED_SUM_SAMPLE(layer,neuron,sample));
                        break;
                    default:
                        printf("ERROR: Unknown activation function derivate\n");
                        throw;
                };

                float delta_var;

                if (layer == Layer_Count - 1) {
                    float target_output = OUTPUTS(sample,neuron);
                    float neuron_output = OUTPUT_AXON_SAMPLE(layer,neuron,sample);
                    delta_var = (neuron_output - target_output) * derivative;
                } else {
                    delta_var = 0;
                    for (int next_neuron = 0; next_neuron < NC(layer + 1); next_neuron++) {
                        float weight_to_next_neuron = WEIGHT(layer + 1,next_neuron,neuron);
                        float delta_next_neuron = DELTA(layer + 1,next_neuron,sample);
                        delta_var += (delta_next_neuron * weight_to_next_neuron) * derivative;
                    }
                }

                DELTA(layer,neuron,sample) = delta_var;
            }
        }
    }

    #ifdef ENABLE
    #pragma omp parallel for num_threads(16)
    #endif
    for (int i = 0; i < Layer_Count; i++) {
        for (int j = 0; j < NC(i); j++) {
            for (int k = 0; k < SENSOR_COUNT(i,j); k++) {
                PREV_GRADIENT(i,j,k) = GRADIENT(i,j,k);

                float accumulated_gradient = 0;
                #ifdef ENABLE
                #pragma omp parallel for reduction(+:accumulated_gradient) num_threads(16)   //Tested with rosenback
                #endif
                for ( int sample = 0; sample < ds->sample_count; sample++)
                {
                    float delta_var = DELTA(i,j,sample);

                    float input_value;
                    if (i == 0)
                    {
                        input_value = INPUTS(sample,k);
                    }
                    else
                    {
                        input_value = OUTPUT_AXON_SAMPLE(i - 1,k,sample);
                    }

                    accumulated_gradient += delta_var * input_value;
                }

                GRADIENT(i,j,k) = accumulated_gradient / ds->sample_count;
            }
        }
    }

    #ifdef ENABLE
    #pragma omp parallel for num_threads(16)
    #endif
    for (int i = 0; i < Layer_Count; i++) {
        for (int j = 0; j < NC(i); j++) {
            for (int k = 0; k < SENSOR_COUNT(i,j); k++) {

                //apply_rmsprop_correction(i,j,k);
                float learning_rate = 0.001;
                float beta = 0.9;
                float epsilon = 0.00000001;

                // apply_adam_correction(i,j,k, Iteration);
                float m_hat;
                float v_hat;
                float blearning_rate = 0.001;
                float bbeta1 = 0.9;  // Momentum decay
                float bbeta2 = 0.999; // RMSprop-like decay
                // float bepsilon = 1e-8;

                // apply_adagrad_momentum_correction(i,j,k);
                float clearning_rate = 0.01;
                float cmomentum = 0.9;
                float cepsilon = 1e-8;

                // apply_irpropplus_correction(i,j,k);
                float dmomentum = 0.5;  // Lowered to balance updates

                switch ( ds->prm_training_algorithm ) {
                    case 0:
                        break;
                    case 1:
                        break;
                    case 2:
                        // apply_rpropplus_correction(i,j,k);
                        if ( PREV_WEIGHT(i,j,k) == WEIGHT(i,j,k) ) WEIGHT_AGE(i,j,k) += 1;
                        else WEIGHT_AGE(i,j,k) = 0;

                        PREV_WEIGHT(i,j,k) = WEIGHT(i,j,k);
                        if (PREV_GRADIENT(i,j,k) * GRADIENT(i,j,k) > 0) {
                             RPROP_DELTA(i,j,k) = fminf( RPROP_DELTA(i,j,k) * rprop_delta_plus, rprop_delta_max);
                            WEIGHT(i,j,k) += -sign(GRADIENT(i,j,k)) *  RPROP_DELTA(i,j,k);
                        } else if (PREV_GRADIENT(i,j,k) * GRADIENT(i,j,k) < 0) {
                             RPROP_DELTA(i,j,k) = fmaxf( RPROP_DELTA(i,j,k) * rprop_delta_minus, rprop_delta_min);
                            WEIGHT(i,j,k) = PREV_WEIGHT(i,j,k);
                            GRADIENT(i,j,k) = 0;
                        } else if (PREV_GRADIENT(i,j,k) * GRADIENT(i,j,k) == 0) {
                            WEIGHT(i,j,k) += -sign(GRADIENT(i,j,k)) *  RPROP_DELTA(i,j,k);
                        }

                        break;
                    case 3:
                        // apply_rmsprop_correction(i,j,k);
                        // Initialize moving average of squared gradients if needed
                        if ( MEAN_SQUARE_GRADIENT(i,j,k) == 0) {
                             MEAN_SQUARE_GRADIENT(i,j,k) = GRADIENT(i,j,k) * GRADIENT(i,j,k);
                        }
                        // Update mean squared gradient with RMSprop formula
                         MEAN_SQUARE_GRADIENT(i,j,k) = beta *  MEAN_SQUARE_GRADIENT(i,j,k) + (1 - beta) * GRADIENT(i,j,k) * GRADIENT(i,j,k);

                        // Update weight based on RMSprop adaptive learning rate
                        WEIGHT(i,j,k) -= learning_rate * GRADIENT(i,j,k) / (sqrtf( MEAN_SQUARE_GRADIENT(i,j,k)) + epsilon);

                        // Store current gradient for potential further usage
                        PREV_GRADIENT(i,j,k) = GRADIENT(i,j,k);
                        break;
                    case 4:
                        // apply_adam_correction(i,j,k, Iteration);
                        // Initialize if needed
                        // if (m[i] == 0) m[i] = 0;  // First moment
                        if ( MEAN_SQUARE_GRADIENT(i,j,k) == 0)  MEAN_SQUARE_GRADIENT(i,j,k) = 0;  // Second moment

                        // Update biased first moment estimate
                        M(i,j,k) = bbeta1 * M(i,j,k) + (1 - bbeta1) * GRADIENT(i,j,k);
                        // Update biased second moment estimate
                         MEAN_SQUARE_GRADIENT(i,j,k) = bbeta2 *  MEAN_SQUARE_GRADIENT(i,j,k) + (1 - bbeta2) * GRADIENT(i,j,k) * GRADIENT(i,j,k);

                        /* iteration counts (iteration) recommended by Grok */
                        m_hat = M(i,j,k) / (1 - powf(bbeta1, Iteration));
                        v_hat =  MEAN_SQUARE_GRADIENT(i,j,k) / (1 - powf(bbeta2, Iteration));

                        // Update weight
                        WEIGHT(i,j,k) -= blearning_rate * m_hat / (sqrtf(v_hat) + epsilon);

                        PREV_GRADIENT(i,j,k) = GRADIENT(i,j,k);
                        break;
                    case 5:
                        // apply_adagrad_momentum_correction(i,j,k);

                        if (M(i,j,k) == 0) M(i,j,k) = 0;  // Momentum term
                         MEAN_SQUARE_GRADIENT(i,j,k) += GRADIENT(i,j,k) * GRADIENT(i,j,k);  // Accumulate (no decay)

                        M(i,j,k) = cmomentum * M(i,j,k) + clearning_rate * GRADIENT(i,j,k) / (sqrtf( MEAN_SQUARE_GRADIENT(i,j,k)) + cepsilon);
                        WEIGHT(i,j,k) -= M(i,j,k);

                        PREV_GRADIENT(i,j,k) = GRADIENT(i,j,k);
                        break;
                    case 6:
                        // apply_irpropplus_correction(i,j,k);
                        if (PREV_WEIGHT(i,j,k) == WEIGHT(i,j,k)) WEIGHT_AGE(i,j,k) += 1;
                        else WEIGHT_AGE(i,j,k) = 0;

                        PREV_WEIGHT(i,j,k) = WEIGHT(i,j,k);

                        // Ensure m[i] is initialized elsewhere (e.g., constructor)
                        float step = (1 - dmomentum) * sign(GRADIENT(i,j,k)) *  RPROP_DELTA(i,j,k);
                        if (PREV_GRADIENT(i,j,k) * GRADIENT(i,j,k) > 0) {
                             RPROP_DELTA(i,j,k) = fminf( RPROP_DELTA(i,j,k) * rprop_delta_plus, rprop_delta_max);
                            M(i,j,k) = dmomentum * M(i,j,k) - step;
                            WEIGHT(i,j,k) += M(i,j,k);
                        } else if (PREV_GRADIENT(i,j,k) * GRADIENT(i,j,k) < 0) {
                             RPROP_DELTA(i,j,k) = fmaxf( RPROP_DELTA(i,j,k) * rprop_delta_minus, rprop_delta_min);
                            WEIGHT(i,j,k) = PREV_WEIGHT(i,j,k);  // Revert only if needed
                            // Don’t zero gradient[i] here unless your loop expects it
                        } else {
                            M(i,j,k) = dmomentum * M(i,j,k) - step;
                            WEIGHT(i,j,k) += M(i,j,k);
                        }
                        break;
                }


                /* Weight Decay (Regularization): Add L2 penalty to optimization methods */
                WEIGHT(i,j,k) -= ds->prm_learning_rate_for_decay * (GRADIENT(i,j,k) + ds->prm_lambda_for_decay * WEIGHT(i,j,k));  // lambda = regularization strength
            }
        }
    }
}


void nnet::processLayers()
{
    #ifdef ENABLE
    #pragma omp parallel for num_threads(16)
    #endif
    for (int i = 0; i < Layer_Count; i++) {
        // If it's not the first layer, feed forward
        if (i > 0) {
            for (int n = 0; n < NC(i); n++) {
                for (int s = 0; s < SENSOR_COUNT(i,n); s++) {
                    SENSOR(i,n,s) = OUTPUT_AXON_VAL(i-1,s);
                }
            }
        }

        // Process the current layer

        for ( int j = 0; j < NC(i); j++ ) {

            switch ( ds->prm_activation_fn ) {
                case ACTIV_FN_SIGMOID:
                        OUTPUT_AXON_VAL(i,j) = transferfn_sigmoid( weight_sum(i,j) );
                    break;
                case ACTIV_FN_TANH:
                        OUTPUT_AXON_VAL(i,j) = transferfn_tanh( weight_sum(i,j) );
                    break;
                case ACTIV_FN_GAUSSIAN:
                        OUTPUT_AXON_VAL(i,j) = transferfn_gaussian( weight_sum(i,j) );
                    break;
                case ACTIV_FN_SIN:
                        OUTPUT_AXON_VAL(i,j) = transferfn_sinusoid( weight_sum(i,j) );
                    break;
                default:
                    printf("ERROR: unknown activation function\n");
                    throw;
            }
        }
    }
}


void nnet::postProcessIteration() {
}


int findFirstNonZeroDigitPosition(double number)
{
    int position = 0;
    number = number - (int)number; // Remove the integer part, if any

    // Guard against the number being zero or negative
    if (number <= 0) {
        return -1; // Indicates no non-zero digit found or invalid input
    }

    while (number > 0) {
        number *= 10; // Shift decimal to the right
        int digit = (int)number; // Get the digit before the decimal point
        position++;
        if (digit != 0) {
            return position; // Return the position of the first non-zero digit
        }
        number -= digit; // Remove the digit just checked
    }

    return -1; // Just in case, but we should never get here for valid inputs
}


int findFirstMismatchPosition(float num1, float num2)
{
    // Remove the integer part, if any
    num1 = num1 - (int)num1;
    num2 = num2 - (int)num2;

    int position = 0;

    while (num1 > 0 || num2 > 0) {
        num1 *= 10;
        num2 *= 10;

        int digit1 = (int)num1;
        int digit2 = (int)num2;

        position++;

        if (digit1 != digit2) {
            return position; // Return the position of the first mismatch
        }

        // Remove the integer part for the next iteration
        num1 -= digit1;
        num2 -= digit2;
    }

    return -1; // Indicates no mismatch found (or both numbers are zero)
}

// Function to shift the decimal place of the fractional part to the right
float shiftDecimalRight(float number) {
    int integerPart = (int)number; // Extract integer part
    float fractionalPart = number - integerPart; // Extract fractional part

    fractionalPart *= 10; // Shift fractional part right
    fractionalPart = fmod(fractionalPart, 1.0); // Keep only the new fractional part

    return integerPart + fractionalPart; // Recombine and return
}

// Function to shift the decimal place of the fractional part to the left
float shiftDecimalLeft(float number) {
    int integerPart = (int)number; // Extract integer part
    float fractionalPart = number - integerPart; // Extract fractional part

    fractionalPart /= 10; // Shift fractional part left

    return integerPart + fractionalPart; // Recombine and return
}



void nnet::reportStatus() {
    ReportCounter = 0;

    if (ds->prm_print == 1) {

        int pos = -1;
        pos = findFirstNonZeroDigitPosition(MsqE);
        int mismatch = -1;
        mismatch = findFirstMismatchPosition(MsqE, PrevReported_MsqE);

        if ( mismatch - pos >= 3 ) {
            ds->prm_rprop_delta_plus = shiftDecimalRight(ds->prm_rprop_delta_plus);

            //ds->prm_rprop_delta_minus *= 10;
            // for ( int i = 1; i < Layer_Count; i++ ) ly[i].apply_settings( ds );
            was_shoot = true;
        } else {
            if (( mismatch - pos <= 2 ) && was_shoot ) {
                ds->prm_rprop_delta_plus = shiftDecimalLeft(ds->prm_rprop_delta_plus);

                //ds->prm_rprop_delta_minus /= 10;
                // for ( int i = 1; i < Layer_Count; i++ ) ly[i].apply_settings( ds );
                was_shoot = false;
            }
        }

        printf("%10d, Lowest MSQE %19.16f %s pos: %2d match: %d cnvrgcnt: %3d\n", Iteration, MsqE, MsqE < MsqELast ? "-" : "+",pos, closeMatchCnt, ConvergeCounter );
        // send_to_influx(Iteration, MsqE, pos, closeMatchCnt);



        /* Graphical reporting */
        if ( !ds->prm_kubernetes ) {

            Clear();

            drawSampleOutput();

            if (msqeDrawCounter<UNIVERSE_WIDTH-1) msqeDrawCounter++;
            else msqeDrawCounter = 0;

            DrawMsqe(msqeDrawCounter,MsqE);
            Render();

        }






        PrevReported_MsqE = MsqE;
    }
}

void nnet::trackConvergence() {

    // if current MSQE is by factor of 10 smaller then previous

    // if (MsqE < MsqELast) {
    if (MsqE < ( MsqELast / 10 ) ) {
        ConvergeCounter++;
    } else {
        DivergeCounter++;
    }

    converging = false;

    if (MsqE < MsqELowest) {
        MsqEPrevLowest = MsqELowest;
        MsqELowest = MsqE;
        ItMsqELowest = Iteration;
        converging = true;
        StagnationCounter = 0;
    }

    if (MsqE > MsqEHighest) {
        MsqEHighest = MsqE;
        ItMsqEHighest = Iteration;
    }

    if (MsqE == MsqELast) {
        StuckCounter++;
    } else {
        StuckCounter = 0;
    }

    MsqEDelta = fabs(MsqE - MsqELast);

    MsqELast = MsqE;
    StagnationCounter++;
}


float nnet::LearStart()
{
    if ( Learning ) {

        printf( "INFO: Learning process is already started\n" );

        return MsqE;

    }

    MsqELast = MsqEPrevLowest;

    StuckCounter = 0;

    MsqELowest = 1000000000;
    MsqEPrevLowest = MsqELowest;
    MsqEHighest = 0;

    ItMsqEHighest = -1;
    ItMsqELowest = -1;



    Iteration = 0;
    ReportCounter = 0;

    ConvergeCounter = 1;
    DivergeCounter = 1;


    Learning = true;

    return Learn();
}



void nnet::printSampleOutput(int sample, int &under_ten, FILE *file)
{
    fprintf(file,"S: %6d ",sample);
    printf(" S: %3d",sample);
    for ( int j = 0; j < ds->sample_output_count; j++) {
        float e = OUTPUTS(sample,j);
        float r = OUTPUT_AXON_VAL(Layer_Count - 1,j);
        e = ds->DenormalizeOutput(e,ds->prm_normalize_min, ds->prm_normalize_max);
        r = ds->DenormalizeOutput( r, ds->prm_normalize_min, ds->prm_normalize_max );

        // OUTPUT_AXON_VAL_DENORMALIZED(Layer_Count - 1,j) = r;

        float x = (e != 0) ? ( ( r - e ) / e ) * 100 : 0;




        if (fabs(x) < 10) under_ten++;

        printf(" D: %7.3f R: %7.3f ( %5.f%% )", e, r, x);

        fprintf(file,"D: %16.8f R: %16.8f ", e, r);
    }

    printf("\n");
    fprintf(file,"\n");
}

void nnet::printSampleOutputProcess(int sample)
{
    printf(" S: %3d",sample);

    for ( int j = 0; j < ds->sample_output_count; j++) {

        float e = OUTPUTS(sample,j);
        float r = OUTPUT_AXON_VAL(Layer_Count - 1,j);

        e = ds->DenormalizeOutput(e,ds->prm_normalize_min, ds->prm_normalize_max);
        r = ds->DenormalizeOutput( r, ds->prm_normalize_min, ds->prm_normalize_max );

        // OUTPUT_AXON_VAL_DENORMALIZED(Layer_Count - 1,j) = r;

        printf(" D:%.1f R:%.1f\n", e, r);
    }

    printf("\n");
}

void nnet::interfaceDenormalization(int sample)
{
    for (int j = 0; j < ds->sample_output_count; j++) {

        float r = OUTPUT_AXON_VAL(Layer_Count - 1,j);

        r = ds->DenormalizeOutput( r, ds->prm_normalize_min, ds->prm_normalize_max );

        // OUTPUT_AXON_VAL_DENORMALIZED(Layer_Count - 1,j) = r;
    }
}

void nnet::drawSampleOutput()
{
    closeMatchCnt = 0;

    float dataset_scale_factor = (float)UNIVERSE_WIDTH / ds->sample_count;
    float scale_factor = (float)(UNIVERSE_HEIGHT/2) / ( ds->dataset_output_max - ds->dataset_output_min );

    for (int sample = 0; sample < ds->sample_count; sample++) {

        for (int j = 0; j < ds->sample_output_count; j++) {

            float e = OUTPUTS(sample,j);

            float r = OUTPUT_AXON_SAMPLE(Layer_Count - 1,j,sample);

            e = ds->DenormalizeOutput( e, ds->prm_normalize_min, ds->prm_normalize_max );
            r = ds->DenormalizeOutput( r, ds->prm_normalize_min, ds->prm_normalize_max );

            int Ye = ( e - ((ds->dataset_min + ds->dataset_output_max)/2)) * scale_factor + UNIVERSE_HEIGHT/2;
            int Yr = ( r - ((ds->dataset_min + ds->dataset_output_max)/2)) * scale_factor + UNIVERSE_HEIGHT/2;

            if ( areFloatsEqual1(e,r) ) closeMatchCnt++;

            // Clear();
            Run( sample*dataset_scale_factor, Ye, Yr );
            // Render();
        }
    }
}

void nnet::drawSampleOutputProcessing( int sample )
{
    closeMatchCnt = 0;

    float dataset_scale_factor = (float)UNIVERSE_WIDTH / ds->sample_count;
    float scale_factor = (float)(UNIVERSE_HEIGHT/2) / ( ds->dataset_output_max - ds->dataset_output_min );

    for (int j = 0; j < ds->sample_output_count; j++) {

        float e = OUTPUTS(sample,j);

        float r = OUTPUT_AXON_VAL(Layer_Count - 1,j);

        e = ds->DenormalizeOutput( e, ds->prm_normalize_min, ds->prm_normalize_max );
        r = ds->DenormalizeOutput( r, ds->prm_normalize_min, ds->prm_normalize_max );

        int Ye = ( e - ((ds->dataset_min + ds->dataset_output_max)/2)) * scale_factor + UNIVERSE_HEIGHT/2;
        int Yr = ( r - ((ds->dataset_min + ds->dataset_output_max)/2)) * scale_factor + UNIVERSE_HEIGHT/2;

        if ( areFloatsEqual1(e,r) ) closeMatchCnt++;

        // Clear();
        Run( sample*dataset_scale_factor, Ye, Yr );
        Render();
    }
}

void nnet::printNetworkArchitecture() {
    printf("\nNeuron Net arch:\n");
    for (int i = 0; i < Layer_Count; i++) {
        printf("Layer %d neuron count %d", i, NC(i));
        if (i == 0) printf(" Input\n");
        else if (i == Layer_Count - 1) printf(" Output\n");
        else printf("\n");
        for (int j = 0; j < NC(i); j++)
            printf("   Neuron %d Sensor count %d\n", j, SENSOR_COUNT(i,j));
    }
    printf("\n");
}

void nnet::printStopConditions() {
    const char *format_str = "Stop Condition: %s ( MsqELowest : %.10f ) ( MsqE : %.10f ) ( It : %5d ) ( Lc: %2d ) Algo=%d \"%s\" Fn=%d Match=%d CnvrgCnt=%d\n";
    if (StagnationCounter >= ds->prm_stagnation)
        printf(format_str, "STAGLOCMIN  ", MsqELowest, MsqE, Iteration, ds->prm_net_layer_count, ds->prm_training_algorithm, ds->fileNameWoExt,ds->prm_activation_fn,closeMatchCnt, ConvergeCounter);
    if (StuckCounter >= ds->prm_stuck)
        printf(format_str, "STUCK       ", MsqELowest, MsqE, Iteration, ds->prm_net_layer_count, ds->prm_training_algorithm, ds->fileNameWoExt,ds->prm_activation_fn,closeMatchCnt, ConvergeCounter);
    if (MsqE <= ds->prm_final_msqe)
        printf(format_str, "SUCCESS     ", MsqELowest, MsqE, Iteration, ds->prm_net_layer_count, ds->prm_training_algorithm, ds->fileNameWoExt,ds->prm_activation_fn,closeMatchCnt, ConvergeCounter);
    if (Iteration >= ds->prm_final_epo)
        printf(format_str, "MAXITERATION", MsqELowest, MsqE, Iteration, ds->prm_net_layer_count, ds->prm_training_algorithm, ds->fileNameWoExt,ds->prm_activation_fn,closeMatchCnt, ConvergeCounter);
}

void nnet::PrintConclusion()
{
    int under_ten = 0;
    FILE *fResult = fopen("result.dat","w");

    // Clear();
    for (int sample = 0; sample < ds->sample_count; sample++) {

            set_initial_input_val(sample);
            processLayers();

        if ( ds->prm_print ) printSampleOutput(sample, under_ten,fResult);
    }
    // Render();
    fclose(fResult);

    if ( ds->prm_print ) printf("From %6d samples %6d are close to truth (+/-5%%)\n", ds->sample_count, under_ten);

    if (ds->prm_print_net_arch) {
        printNetworkArchitecture();
    }

    LearningRatio = (float)(ConvergeCounter - DivergeCounter) / (float)(ConvergeCounter + DivergeCounter);

    printStopConditions();

    if ( ds->prm_print )
    {
        printf(" \n ( Iterations       : %d ) \n ( Last MSQE        : %.16f )\n ( Lowest MSQE      : %.16f )\n ( Recommended MSQE : %.16f )\n\n", Iteration, MsqE, MsqELowest, MsqEPrevLowest);
        printf("Convergence-Divergence Ratio: %d:%d %1.1f\n", ConvergeCounter, DivergeCounter, LearningRatio);
    }
}

void nnet::PrintProcessingConclusion()
{
    for( int sample = 0; sample < ds->sample_count; sample++) {

            set_initial_input_val(sample);

            processLayers();

            printSampleOutputProcess(sample); // this prints out desired vs received

            drawSampleOutputProcessing(sample);
    }
}


void nnet::SaveNet(const char *fileName)
{
    FILE *fnet = fopen( fileName, "w" );

    fprintf( fnet, "info_sample_count = %d\n", ds->sample_count );
    fprintf( fnet, "info_iteration_count = %d\n", Iteration );
    fprintf( fnet, "info_last_msqe = %f\n", MsqE);
    fprintf( fnet, "info_lowest_msqe = %f\n", MsqELowest);
    fprintf( fnet, "info_highest_msqe = %f\n", MsqEHighest);
    fprintf( fnet, "info_lowest_msqe_it = %d\n", ItMsqELowest);
    fprintf( fnet, "info_highest_msqe_it = %d\n", ItMsqEHighest);
    fprintf( fnet, "info_ds_param_prm_report = %d\n", ds->prm_report);
    fprintf( fnet, "info_ds_param_prm_print = %d\n", ds->prm_print);			/* 0 - Silent, 1 - Epoch Msqe, 2 - Detailed output */
    fprintf( fnet, "info_ds_param_prm_print_input = %d\n", ds->prm_print_input);		/* 0 - Silent, 1 - Epoch Msqe, 2 - Detailed output */
    fprintf( fnet, "info_ds_param_prm_print_output = %d\n", ds->prm_print_output);		/* 0 - Silent, 1 - Epoch Msqe, 2 - Detailed output */
    fprintf( fnet, "info_ds_param_prm_print_net_arch = %d\n", ds->prm_print_net_arch);
    fprintf( fnet, "info_ds_param_prm_stagnation = %d\n", ds->prm_stagnation);			/* Learning process iteartion count since last Msqe decreasing  */
    fprintf( fnet, "info_ds_param_prm_stuck = %d\n", ds->prm_stuck);			/* Learning process iteartion count without frozen Msqe */
    fprintf( fnet, "prm_activation_fn = %d\n", ds->prm_activation_fn);		/* Activation fuction type */
    fprintf( fnet, "info_ds_param_prm_steepness = %f\n", ds->prm_steepness);		/* Trashold */
    fprintf( fnet, "info_ds_param_prm_learning_rate = %f\n", ds->prm_learning_rate_for_decay);		/* Learning rate */
    fprintf( fnet, "info_ds_param_prm_bias = %f\n", ds->prm_bias);			/* Bias value */
    fprintf( fnet, "info_ds_param_prm_momentum = %f\n", ds->prm_momentum);			/* Momentum step */
    fprintf( fnet, "info_ds_param_prm_final_msqe = %f\n", ds->prm_final_msqe);		/* Expected Msqe */
    fprintf( fnet, "info_ds_param_prm_final_epo = %d\n", ds->prm_final_epo );			/* Maximumum learning iteration epoch count */
    fprintf( fnet, "info_ds_param_prm_dataset_min = %f\n", ds->dataset_min);
    fprintf( fnet, "info_ds_param_prm_dataset_max = %f\n", ds->dataset_max);
    fprintf( fnet, "dataset_input_min = %f\n", ds->dataset_input_min);
    fprintf( fnet, "dataset_input_max = %f\n", ds->dataset_input_max);
    fprintf( fnet, "dataset_output_min = %f\n", ds->dataset_output_min);
    fprintf( fnet, "dataset_output_max = %f\n", ds->dataset_output_max);
    fprintf( fnet, "prm_normalize_min = %f\n", ds->prm_normalize_min);
    fprintf( fnet, "prm_normalize_max = %f\n", ds->prm_normalize_max);
    fprintf( fnet, "prm_training_algorithm = %d\n", ds->prm_training_algorithm);
    fprintf( fnet, "info_ds_param_prm_sample_input_count = %d\n", ds->sample_input_count);
    fprintf( fnet, "info_ds_param_prm_sample_output_count = %d\n", ds->sample_output_count);
    fprintf( fnet, "layer_count = %d\n", Layer_Count );

    for ( int i = 0; i < Layer_Count; i++ ) {

        fprintf( fnet, "nc_ly_%d = %d\n", i, NC(i) );

    }

    for ( int l = 0; l < Layer_Count; l++ ) {

        for ( int n = 0; n < NC(l); n++ ) {

            for ( int s = 0; s < SENSOR_COUNT(l,n); s++ ) {

                fprintf( fnet, "ly%dn%dsensor%d_weight = %19.16f\n", l, n, s, WEIGHT(l,n,s) );

            }
        }
    }
}


void nnet::LoadNet(const char *fileName)
{
    PRegEx RegExParm;

    RegExParm.init();
    RegExParm.verbosity(0);

    if ( DDEBUG ) printf("Loading data from file %s...\n",fileName);

    RegExParm.load( fileName );

    if ( DDEBUG ) printf("Parsing parameter section...\n");

    RegExParm.regextract( (char*)"\\(^\\w+\\)\\s*=\\s*\\([0-9.e-]+\\)" );

    if ( DDEBUG ) printf("Loading parameters..\n");

    for ( unsigned long i = 0; i < RegExParm.getrowcount(); i++ ) {

        if ( strcmp(RegExParm.getval(i,0),"info_ds_param_prm_dataset_min")         == 0 ) ds->dataset_min            = StrToFloat( RegExParm.getval(i,1) );
        if ( strcmp(RegExParm.getval(i,0),"info_ds_param_prm_dataset_max")         == 0 ) ds->dataset_max            = StrToFloat( RegExParm.getval(i,1) );
        if ( strcmp(RegExParm.getval(i,0),"dataset_input_min")                     == 0 ) ds->dataset_input_min      = StrToFloat( RegExParm.getval(i,1) );
        if ( strcmp(RegExParm.getval(i,0),"dataset_input_max")                     == 0 ) ds->dataset_input_max      = StrToFloat( RegExParm.getval(i,1) );
        if ( strcmp(RegExParm.getval(i,0),"dataset_output_min")                    == 0 ) ds->dataset_output_min     = StrToFloat( RegExParm.getval(i,1) );
        if ( strcmp(RegExParm.getval(i,0),"dataset_output_max")                    == 0 ) ds->dataset_output_max     = StrToFloat( RegExParm.getval(i,1) );
        if ( strcmp(RegExParm.getval(i,0),"prm_normalize_min")                     == 0 ) ds->prm_normalize_min      = StrToFloat( RegExParm.getval(i,1) );
        if ( strcmp(RegExParm.getval(i,0),"prm_normalize_max")                     == 0 ) ds->prm_normalize_max      = StrToFloat( RegExParm.getval(i,1) );
        if ( strcmp(RegExParm.getval(i,0),"info_ds_param_prm_sample_input_count")  == 0 ) ds->sample_input_count     = StrToFloat( RegExParm.getval(i,1) );
        if ( strcmp(RegExParm.getval(i,0),"info_ds_param_prm_sample_output_count") == 0 ) ds->sample_output_count    = StrToFloat( RegExParm.getval(i,1) );
        if ( strcmp(RegExParm.getval(i,0),"prm_training_algorithm")                == 0 ) ds->prm_training_algorithm = StrToInt( RegExParm.getval(i,1) );
        if ( strcmp(RegExParm.getval(i,0),"prm_activation_fn")                     == 0 ) ds->prm_activation_fn      = StrToInt( RegExParm.getval(i,1) );
        if ( strcmp(RegExParm.getval(i,0),"layer_count")                           == 0 ) Layer_Count                = StrToInt( RegExParm.getval(i,1) );

    }

    if ( DDEBUG ) printf("Creating %d Layers..\n",Layer_Count);

    CreateLayers( Layer_Count );

    if ( DDEBUG ) printf("Creating saved net..\n");

    char param[256];

    /* Detecting maximal neuron count per layer in net */
    for ( int i = 0; i < Layer_Count; i++ ) {

        sprintf( param, "nc_ly_%d", i );

        for ( unsigned long j = 0; j < RegExParm.getrowcount(); j++ ) {

            if ( strcmp(RegExParm.getval(j,0), param ) == 0 ) {

                NC(i) = StrToInt( RegExParm.getval(j,1) );

                if ( NC(i) > ds->prm_max_neuron_count_in_layer ) ds->prm_max_neuron_count_in_layer = NC(i);
            }
        }
    }

    eprm_max_neuron_count_in_layer = ds->prm_max_neuron_count_in_layer;

    esensor_max_count = eprm_max_neuron_count_in_layer;

    allocMemory();

    /* Loading net */
    for ( int i = 0; i < Layer_Count; i++ ) {

        sprintf( param, "nc_ly_%d", i );

        for ( unsigned long j = 0; j < RegExParm.getrowcount(); j++ ) {

            if ( strcmp(RegExParm.getval(j,0), param ) == 0 ) {

                NC(i) = StrToInt( RegExParm.getval(j,1) );

                if ( i == 0 ) configure_neurons(i, 1, ds );
                else configure_neurons(i, NC(i - 1), ds );

                for ( int n = 0; n < NC(i); n++ ) {

                    for ( int s = 0; s < SENSOR_COUNT(i,n); s++ ) {

                        sprintf( param, "ly%dn%dsensor%d_weight", i, n, s );

                        for ( unsigned long k = 0; k < RegExParm.getrowcount(); k++ ) {

                            if ( strcmp(RegExParm.getval(k,0), param ) == 0 ) {

                                WEIGHT(i,n,s) = StrToFloat( RegExParm.getval(k,1) );

                                break;
                            }
                        }
                    }
                }

                break;
            }
        }
    }

    if ( DDEBUG ) printf("Net Loaded! %d\n",ds->prm_max_neuron_count_in_layer);
}






#include <pregex/pregex.h>

#define DEBUG 0

nndataset::nndataset()
{
    sample_count = 0;

    sample_input_count = 0;
    sample_output_count = 0;

    prm_training_algorithm = 2;

    prm_rprop_delta_plus = 1.2f;   //1.2
    prm_rprop_delta_minus = 0.5f;   //0.5
    prm_rprop_delta_max = 100000.0f;    //50
    prm_rprop_delta_min = 0.000001f;//0.000001f

    prm_normalize_min = 0.01;
    prm_normalize_max = 0.8;

    prm_total_neuron_count = -1;
    prm_max_neuron_count_in_layer = -1;
    prm_net_layer_count = 0;
    prm_net_layer = nullptr;

    prm_print = 1;
    prm_print_input = 0;
    prm_print_output = 0;
    prm_print_net_arch = 0;

    prm_net_layer_type_rnn = -1;

    prm_stagnation = 100;
    prm_stuck = 100;
    prm_activation_fn = 2;

    prm_steepness = 0.5;
    prm_learning_rate_for_decay = 0.00001;
    prm_lambda_for_decay = 1.0;
    prm_bias = 0.1;
    prm_momentum = 0.2;

    prm_final_msqe = 0.001;
    prm_final_epo = 1000000;

    prm_report = 10;

    prm_bias_neuron_val = 1.0F;

    dataset_input_min = 10000000;
    dataset_input_max = -10000000;

    dataset_output_min = 10000000;
    dataset_output_max = -10000000;

    prm_kubernetes = false;
}


nndataset::~nndataset()
{
    // printf("Dataset Destructor\n");
    delete [] prm_net_layer;

    free(inputs);
    free(outputs);
}


void nndataset::ProcessFileName(const char* fileName)
{
    char buffer[256];

    extractFileName(fileName, buffer );

    strcpy( (char*)fileNameOrig, fileName );

    strcpy( (char*)fileNameWoExt, (const char*)buffer );

    sprintf( buffer, "%s.pnn", fileNameWoExt );

    strcpy( (char*)fileNamePNN, (const char*)buffer );

    sprintf( buffer, "%s.csv", fileNameWoExt );

    strcpy( (char*)fileNameCSV, (const char*)buffer );

    // printf("Processed filenames:\n\t%s\n\t%s\n\t%s\n\t%s\n",fileName,fileNameWoExt,fileNamePNN,fileNameCSV);
}


void nndataset::LoadDataset( const char *fileName )
{
    PRegEx RegExParm;

    sample_count = 0;

    sample_input_count = 0;
    sample_output_count = 0;

    prm_total_neuron_count = 0;
    prm_max_neuron_count_in_layer = 0;

    RegExParm.init();
    RegExParm.verbosity(0);

    if ( DEBUG ) printf("Loading data from file %s...\n",fileName);
    RegExParm.load(fileName );

    char *buf = RegExParm.getbuf();

    int j = 0;

    while( !(( buf[j]=='I' ) && ( buf[j+1]==',' )) ) j++; // Seek data header line

    if ( DEBUG ) printf("Parsing parameter section...\n");

    RegExParm.regextract( (char*)"\\(^\\w\\w+\\),\\([A-Za-z0-9.-]+$\\)" );

    for ( int i = 0; i < (int)RegExParm.getrowcount(); i++ )
    {

        if ( strcmp(RegExParm.getval(i,0),"prm_net_layer_count")               == 0 ) {

            prm_net_layer_count  = StrToInt(RegExParm.getval(i,1));

            prm_net_layer = new int[prm_net_layer_count+1];     // Array element 0 is not utilized
        }


        for ( j = 1; j <= prm_net_layer_count; j++) {

            char str[255];

            sprintf(str,"prm_net_layer%d",j);

            if ( strcmp(RegExParm.getval(i,0),str) == 0 ) {

                prm_net_layer[j] = StrToInt(RegExParm.getval(i,1));

                if ( prm_net_layer[i] > prm_max_neuron_count_in_layer ) prm_max_neuron_count_in_layer = prm_net_layer[i];

                prm_total_neuron_count += prm_net_layer[j];
            }

        }


        // if ( strcmp(RegExParm.getval(i,0),"netarch")               == 0 ) prm_net_architecture  = StrToInt(RegExParm.getval(i,1));


        if ( strcmp(RegExParm.getval(i,0),"prm_net_layer_type_rnn") == 0 ) prm_net_layer_type_rnn = StrToInt(RegExParm.getval(i,1));
        if ( strcmp(RegExParm.getval(i,0),"prm_training_algorithm") == 0 ) prm_training_algorithm = StrToFloat(RegExParm.getval(i,1));
        if ( strcmp(RegExParm.getval(i,0),"prm_rprop_delta_plus")   == 0 ) prm_rprop_delta_plus   = StrToFloat(RegExParm.getval(i,1));
        if ( strcmp(RegExParm.getval(i,0),"prm_rprop_delta_minus")  == 0 ) prm_rprop_delta_minus  = StrToFloat(RegExParm.getval(i,1));
        if ( strcmp(RegExParm.getval(i,0),"prm_rprop_delta_max")    == 0 ) prm_rprop_delta_max    = StrToFloat(RegExParm.getval(i,1));
        if ( strcmp(RegExParm.getval(i,0),"prm_rprop_delta_min")    == 0 ) prm_rprop_delta_min    = StrToFloat(RegExParm.getval(i,1));
        if ( strcmp(RegExParm.getval(i,0),"prm_normalize_max")      == 0 ) prm_normalize_max      = StrToFloat(RegExParm.getval(i,1));
        if ( strcmp(RegExParm.getval(i,0),"prm_normalize_min")      == 0 ) prm_normalize_min      = StrToFloat(RegExParm.getval(i,1));
        if ( strcmp(RegExParm.getval(i,0),"prm_activation_fn")      == 0 ) prm_activation_fn      = StrToInt(RegExParm.getval(i,1));
        if ( strcmp(RegExParm.getval(i,0),"prm_print_net_arch")     == 0 ) prm_print_net_arch     = StrToInt(RegExParm.getval(i,1));
        if ( strcmp(RegExParm.getval(i,0),"print")                  == 0 ) prm_print              = StrToInt(RegExParm.getval(i,1));
        if ( strcmp(RegExParm.getval(i,0),"report")                 == 0 ) prm_report             = StrToInt(RegExParm.getval(i,1));
        if ( strcmp(RegExParm.getval(i,0),"msqe")                   == 0 ) prm_final_msqe         = StrToFloat(RegExParm.getval(i,1));
        if ( strcmp(RegExParm.getval(i,0),"stagnation")             == 0 ) prm_stagnation         = StrToInt(RegExParm.getval(i,1));
        if ( strcmp(RegExParm.getval(i,0),"stuck")                  == 0 ) prm_stuck              = StrToInt(RegExParm.getval(i,1));
        if ( strcmp(RegExParm.getval(i,0),"epochs")                 == 0 ) prm_final_epo          = StrToInt(RegExParm.getval(i,1));
    }


    if ( DEBUG ) printf("Retriving input and output count...\n");
    // Seek data start line and count input/output
    while( !(( buf[j]=='O' ) && ( buf[j+1]=='\n' )) )
    {
        if(buf[j]=='I') sample_input_count++;
        if(buf[j]=='O') sample_output_count++;
        j++;
    }
    sample_output_count+=1;
    j+=2;


    prm_total_neuron_count += sample_input_count;
    prm_total_neuron_count += sample_output_count;

    if ( sample_input_count > prm_max_neuron_count_in_layer ) prm_max_neuron_count_in_layer = sample_input_count;
    if ( sample_output_count > prm_max_neuron_count_in_layer ) prm_max_neuron_count_in_layer = sample_output_count;

    eprm_max_neuron_count_in_layer = prm_max_neuron_count_in_layer;
    esensor_max_count = eprm_max_neuron_count_in_layer;

    if ( DEBUG ) printf("Maximum neuron count in layer: %d\n", prm_max_neuron_count_in_layer);

    if ( DEBUG ) printf("Total Neuron count in network: %d\n",prm_total_neuron_count);

    esample_input_count = sample_input_count;		/* Input field count in dataset */
    esample_output_count = sample_output_count;	/* Output field count in dataset */


    PRegEx RegExDat;

	// tmatch *match_data;
	// int col_cnt_data;
	// int row_cnt_data;


    RegExDat.init();
    RegExDat.verbosity(0);
    RegExDat.load( fileName );

    if ( DEBUG ) printf("Preparing regex pattern...\n");
    char str1[] = "\\(^[0-9.-]+\\),";
    char str[] = "\\([0-9.-]+\\),";

    int len = sample_input_count * strlen(str) + sample_output_count * strlen(str) + 1; // 1 for ^ symbol

    char *pattern = (char*)calloc(len,sizeof(char*));

    pattern[0]='\0';

    strcat(pattern,str1);
    for( int i = 1; i < ( sample_input_count + sample_output_count ); i++ ) strcat(pattern,str);

    pattern[len-1]='\0';

    if ( DEBUG ) printf("Parsing dataset...\n");
    RegExDat.regextract( pattern );

    sample_count = RegExDat.getrowcount();

    esample_count = sample_count;

    if ( DEBUG ) printf("Allocating memory for samples...\n");

    printf("ALLOCATING MEMORY 3\n");

    inputs = (float*)calloc(sample_count * sample_input_count, sizeof(float));
    outputs = (float*)calloc(sample_count * sample_output_count, sizeof(float));

    if ( DEBUG ) printf( "\nINFO: Input file \"%s\" contains %d samples with Inputs: %d Outputs: %d\n\n", fileName, sample_count, sample_input_count, sample_output_count );

    if ( DEBUG ) printf("Formatting and populating dataset...\n");


    float data_min = 1000000000;
    float data_max = -1000000000;

    for ( int i = 0; i < sample_count; i++ )
    {
        for( int k = 0; k < (int)RegExDat.getcolcount(); k++ ) {
            float val = StrToFloat(RegExDat.getval(i,k));

            if ( val > data_max ) data_max = val;
            if ( val < data_min ) data_min = val;

            /*For inputs */
            if ( k < sample_input_count ) {
                if ( val > dataset_input_max ) dataset_input_max = val;
                if ( val < dataset_input_min ) dataset_input_min = val;
            } else {
            /*For outputs */
                if ( val > dataset_output_max ) dataset_output_max = val;
                if ( val < dataset_output_min ) dataset_output_min = val;
            }


            if ( k < sample_input_count ) INPUTS(i,k) = val;
            else OUTPUTS(i,RegExDat.getcolcount() - k - 1) = val;
        }
    }

    dataset_min = data_min;
    dataset_max = data_max;

    if ( DEBUG ) printf( "\nMIN: %f\nMAX: %f\n\n", dataset_min, dataset_max );
    if ( DEBUG ) printf( "\n\tInputs  MIN: %f\n\tInputs  MAX: %f\n\n", dataset_input_min, dataset_input_max );
    if ( DEBUG ) printf( "\n\tOutputs MIN: %f\n\tOutputs MAX: %f\n\n", dataset_output_min, dataset_output_max );

    if ( DEBUG ) printf("Normalizing dataset..\n");

    NormalizeInputData( prm_normalize_min, prm_normalize_max );

    if ( DEBUG ) printf("Load done!\n");
}




























void nndataset::LoadProcessDataset( const char *fileName )
{
    PRegEx RegExParm;

    sample_count = 0;

    sample_input_count = 0;

    int a_sample_output_count = 0;

    RegExParm.init();
    RegExParm.verbosity(0);

    if ( DEBUG ) printf("Loading data from file %s...\n",fileName);
    RegExParm.load( fileName );

    char *buf = RegExParm.getbuf();

    int j = 0;

    while( !(( buf[j]=='I' ) && ( buf[j+1]==',' )) ) j++; // Seek data header line

    if ( DEBUG ) printf("Parsing parameter section...\n");

    RegExParm.regextract( (char*)"\\(^\\w\\w+\\),\\([A-Za-z0-9.-]+$\\)" );

    for ( int i = 0; i < (int)RegExParm.getrowcount(); i++ )
    {

            // if ( strcmp(RegExParm.getval(i,0),"netarch")               == 0 ) prm_net_architecture  = StrToInt(RegExParm.getval(i,1));
            // if ( strcmp(RegExParm.getval(i,0),"prm_rprop_delta_plus")  == 0 ) prm_rprop_delta_plus  = StrToFloat(RegExParm.getval(i,1));
            // if ( strcmp(RegExParm.getval(i,0),"prm_rprop_delta_minus") == 0 ) prm_rprop_delta_minus = StrToFloat(RegExParm.getval(i,1));
            // if ( strcmp(RegExParm.getval(i,0),"prm_rprop_delta_max")   == 0 ) prm_rprop_delta_max   = StrToFloat(RegExParm.getval(i,1));
            // if ( strcmp(RegExParm.getval(i,0),"prm_rprop_delta_min")   == 0 ) prm_rprop_delta_min   = StrToFloat(RegExParm.getval(i,1));
            // if ( strcmp(RegExParm.getval(i,0),"prm_normalize_max")     == 0 ) prm_normalize_max     = StrToFloat(RegExParm.getval(i,1));
            // if ( strcmp(RegExParm.getval(i,0),"prm_normalize_min")     == 0 ) prm_normalize_min     = StrToFloat(RegExParm.getval(i,1));
        // if ( strcmp(RegExParm.getval(i,0),"prm_activation_fn")     == 0 ) prm_activation_fn     = StrToInt(RegExParm.getval(i,1));
            // if ( strcmp(RegExParm.getval(i,0),"print")                 == 0 ) prm_print             = StrToInt(RegExParm.getval(i,1));
            // if ( strcmp(RegExParm.getval(i,0),"report")                == 0 ) prm_report            = StrToInt(RegExParm.getval(i,1));
            // if ( strcmp(RegExParm.getval(i,0),"msqe")                  == 0 ) prm_final_msqe        = StrToFloat(RegExParm.getval(i,1));
            // if ( strcmp(RegExParm.getval(i,0),"stagnation")            == 0 ) prm_stagnation        = StrToInt(RegExParm.getval(i,1));
            // if ( strcmp(RegExParm.getval(i,0),"stuck")                 == 0 ) prm_stuck             = StrToInt(RegExParm.getval(i,1));
            // if ( strcmp(RegExParm.getval(i,0),"epochs")                == 0 ) prm_final_epo         = StrToInt(RegExParm.getval(i,1));
    }


    if ( DEBUG ) printf("Retriving input and output count...\n");
    // Seek data start line and count input/output
    while( !(( buf[j]=='O' ) && ( buf[j+1]=='\n' )) )
    {
        if(buf[j]=='I') sample_input_count++;
        if(buf[j]=='O') a_sample_output_count++;
        j++;
    }
    a_sample_output_count+=1;
    j+=2;

    esample_input_count = sample_input_count;		/* Input field count in dataset */
    esample_output_count = a_sample_output_count;	/* Output field count in dataset */

    PRegEx RegExDat;

    RegExDat.init();
    RegExDat.verbosity(0);
    RegExDat.load( fileName );

    if ( DEBUG ) printf("Preparing regex pattern...\n");
    if ( DEBUG ) printf("Preparing regex pattern...\n");
    char str1[] = "\\(^[0-9.-]+\\),";
    char str[] = "\\([0-9.-]+\\),";

    int len = sample_input_count * strlen(str) + sample_output_count * strlen(str) + 1; // 1 for ^ symbol

    char *pattern = (char*) calloc(len,sizeof(char*));

    pattern[0]='\0';

    strcat(pattern,str1);
    for( int i = 1; i < ( sample_input_count + sample_output_count ); i++ ) strcat(pattern,str);

    pattern[len-1]='\0';

    if ( DEBUG ) printf("Parsing dataset...\n");
    RegExDat.regextract( pattern );

    sample_count = RegExDat.getrowcount();

    if ( DEBUG ) printf("Allocating memory for samples...\n");

    printf("ALLOCATING MEMORY 4\n");

    inputs = (float*)calloc(sample_count * sample_input_count, sizeof(float));
    outputs = (float*)calloc(sample_count * a_sample_output_count, sizeof(float));


    if ( DEBUG ) printf( "\nINFO: Input file \"%s\" contains %d samples with Inputs: %d Outputs: %d\n\n", fileName, sample_count, sample_input_count, a_sample_output_count );

    if ( DEBUG ) printf("Formatting and populating dataset...\n");

    for( int i = 0; i < sample_count; i++ ) {

        for( int k = 0; k < (int)RegExDat.getcolcount(); k++ )
        {
            float val = StrToFloat(RegExDat.getval(i,k));

            if ( k < sample_input_count ) INPUTS(i,k) = val;
            else OUTPUTS(i,RegExDat.getcolcount() - k - 1) = val;
        }
    }

    if ( DEBUG ) printf( "\nMIN: %f\nMAX: %f\n\n", dataset_min, dataset_max );

    if ( DEBUG ) printf("Normalizing dataset..\n");

    NormalizeInputData( prm_normalize_min, prm_normalize_max );

    if ( DEBUG ) printf("Load done!\n");
}




void nndataset::NormalizeInputData( float a, float b )
{
// // Set the target normalization range
// float a = -1.0f;
// float b = 1.0f;

for ( int i = 0; i < sample_count; i++) {
    // Normalize inputs to [-1, 1] using input-specific min and max
    for (int j = 0; j < sample_input_count; j++) {
        INPUTS(i,j) = a + ((INPUTS(i,j) - dataset_input_min) * (b - a)) / (dataset_input_max - dataset_input_min);
    }

    // Normalize outputs to [-1, 1] using output-specific min and max
    for ( int j = 0; j < sample_output_count; j++) {
        OUTPUTS(i,j) = a + ((OUTPUTS(i,j) - dataset_output_min) * (b - a)) / (dataset_output_max - dataset_output_min);
    }
}

    // for ( int i = 0; i < sample_count; i++ )
    // {
    //     /*Cubic equation form -100 to 100 doesn't work with this otherwise this is good and tested*/
    //     // for ( int j = 0; j < sample_input_count; j++ ) samples[ i ].inputs[ j ] = a + ((samples[ i ].inputs[ j ] - dataset_min) * (b - a)) / (dataset_max - dataset_min);
    //     // for ( int j = 0; j < sample_output_count; j++ ) samples[ i ].outputs[ j ] = a + ((samples[ i ].outputs[ j ]- dataset_min) * (b - a)) / (dataset_max - dataset_min);
    //
    //     for ( int j = 0; j < sample_input_count; j++ ) samples[ i ].inputs[ j ] = a + ((samples[ i ].inputs[ j ] - dataset_input_min) * (b - a)) / (dataset_input_max - dataset_input_min);
    //     for ( int j = 0; j < sample_output_count; j++ ) samples[ i ].outputs[ j ] = a + ((samples[ i ].outputs[ j ]- dataset_output_min) * (b - a)) / (dataset_output_max - dataset_output_min);
    // }

    if ( DEBUG ) {
        for ( int i = 0; i < sample_count; i++ ) {

            for ( int j = 0; j < sample_input_count; j++ )  printf( "I: %.16f ", INPUTS(i,j) );
            for ( int j = 0; j < sample_output_count; j++ ) printf( "O: %.16f ", OUTPUTS(i,j) );
            printf("\n");

        }
    }
}

float nndataset::DenormalizeOutput(float value, float a, float b)
{
    /*Cubic equation form -100 to 100 doesn't work with this otherwise this is good and tested*/
    // return (value - a) * (dataset_max - dataset_min) / (b - a) + dataset_min;
    return (value - a) * (dataset_output_max - dataset_output_min) / (b - a) + dataset_output_min;
}

void nndataset::PrintInputData()
{
    for ( int i = 0; i < sample_count; i++ ) {

        for ( int j = 0; j < sample_input_count; j++ ) printf( "%.2f ", INPUTS(i,j) );
        for ( int j = 0; j < sample_output_count; j++ ) printf( "%.2f ", OUTPUTS(i,j) );

        printf( "\n" );
    }
}


int nndataset::GetPrintLevel()
{
    return prm_print;
}


void nndataset::SetPrintLevel( int debug_level )
{
    printf( "Output changed from %d to %d\n", prm_print, debug_level );
    prm_print = debug_level;
}




void handle_sigint(int sig)
{
    printf("Caught signal %d. Exiting...\n", sig);

    if ( !Net->ds->prm_kubernetes ) Net->PrintConclusion();

    printf("\tSaving net..\n");

    Net->SaveNet( Net->ds->fileNamePNN );

    printf("\tNet saved! Quitting..\n");

    delete Net;

    freeMemoryCUDA();

    exit(1);
}

void InfoPnnHelp()
{
    printf("\nPNN help (pnn)\n\n \
    usage: pnn [-h] -t dataset.csv -n hlc nc1 ncn | -p dataset.csv \n\n\
        -h, --help      -       this help screen \n\n\
        -t dataset      -       teach network \n\
        -l dataset      -       learn from dataset file \n\
        -p dataset      -       process dataset \n\n\
        -c dataset      -       continue teach net \n\n\
        -d dataset      -       detect optimal network structure \n\n\
        -v level        -       0 - Silent, 1 - Epoch Msqe, 2 - Detailed output\n\n\
        -o              -       print processed dataset \n\n\
        -a              -       print net architecture \n\n\
        -i              -       train incremental \n\n\
        -b              -       train batch \n\n\
        -r              -       train rprop \n\n\
    \n\n");

    exit( 0 );
}


void Process_CommandlineArguments( int argc, char ** argv )
{
    for (int x = 0; x < argc; x++ ) {

        if ( ( strcmp(argv[x],"-h") == 0 ) || ( strcmp( argv[x],"--help" ) == 0 ) || ( argc == 1 ) ) InfoPnnHelp();

        if ( strcmp( argv[x],"-d" ) == 0 ) {

            Detect = true;

            strcpy( (char*)fileName, argv[ x + 1 ] );

            if ( ! FileExist( fileName ) ) {

                printf( "File %s does not exist!\n", fileName );

                InfoPnnHelp();
            }
        }

        if ( strcmp( argv[x],"-l" ) == 0 ) {

            Learn = true;

            strcpy( (char*)fileName, argv[ x + 1 ] );

            if ( ! FileExist( fileName ) ) {

                printf( "File %s does not exist!\n", fileName );

                InfoPnnHelp();
            }

        }

        if ( strcmp( argv[x], "-p" ) == 0 ) {

            Process = true;

            strcpy( (char*)fileName, argv[ x + 1 ] );

            if ( ! FileExist( fileName ) ) {

                printf( "File %s does not exist!\n", fileName );

                InfoPnnHelp();
            }

            printf( "Processing dataset %s.\n", fileName );

        }

        if ( strcmp( argv[x], "-c" ) == 0 ) {

            Continue = true;

            strcpy( (char*)fileName, argv[ x + 1 ] );

            if ( ! FileExist( fileName ) ) {

                printf( "File %s does not exist!\n", fileName );

                InfoPnnHelp();
            }

            printf( "Continue training using dataset %s.\n", fileName );

        }

        if ( strcmp( argv[x],"-v" ) == 0 ) Net->ds->prm_print = StrToInt( argv[ x + 1 ] );

        if ( strcmp( argv[x],"-i" ) == 0 ) Net->ds->prm_print_input = 1;

        if ( strcmp( argv[x],"-o" ) == 0 ) Net->ds->prm_print_output = 1;

        if ( strcmp( argv[x],"-a" ) == 0 ) Net->ds->prm_print_net_arch = 1;


        if ( strcmp( argv[x],"-i" ) == 0 ) Net->ds->prm_training_algorithm = 0;

        if ( strcmp( argv[x],"-b" ) == 0 ) Net->ds->prm_training_algorithm = 1;

        if ( strcmp( argv[x],"-r" ) == 0 ) Net->ds->prm_training_algorithm = 2;

        if ( strcmp( argv[x],"-s" ) == 0 ) Net->ds->prm_training_algorithm = 3;

        if ( strcmp( argv[x],"-k" ) == 0 ) Net->ds->prm_kubernetes = true;

    }
}



void detectNormalizeMax( int argc, char **argv, float *detectedLowestMSE, int detectedBestAlgorithm, int detectedBestLayerCount, int neuronCount, int detectedFn, int deltaMax, float deltaMin, float deltaPlus, float deltaMinus, float normalizeMin, float *normalizeMax )
{
    float MSE = *detectedLowestMSE;
    int detectedIt = 0;

    for ( float delta = 0.9; delta > 0.3; delta -= 0.1 ) {
        Net = new nnet;

        Process_CommandlineArguments( argc, argv );

        Net->ds->ProcessFileName(fileName);

        Net->LoadDataset( fileName );

        // Net->ds->sample_count *= 0.1;   // While detecting uses on 10% of dataset

        Net->ds->prm_training_algorithm = detectedBestAlgorithm;

        Net->ds->prm_print = 0; // Dont print Iteration

        Net->ds->prm_final_epo = DETECTION_ITERATION_COUNT;

        Net->ds->prm_final_msqe = 0.000000000001;  // Set to unrela value to run through max iteration count

        Net->ds->prm_activation_fn = detectedFn;

        Net->ds->prm_net_layer_count = detectedBestLayerCount;

        Net->ds->prm_net_layer = new int[Net->ds->prm_net_layer_count+1];     // Array element 0 is not utilized

        for ( int j = 1; j <= Net->ds->prm_net_layer_count; j++) {

            Net->ds->prm_net_layer[j] = neuronCount;

        }

        Net->CreateLayers( Net->ds->prm_net_layer_count + 2 );

        Net->SetupLayers( Net->ds->prm_net_layer_count, Net->ds->prm_net_layer );

        Net->ds->prm_rprop_delta_max = deltaMax;
        Net->ds->prm_rprop_delta_min = deltaMin;

        Net->ds->prm_rprop_delta_plus = deltaPlus;
        Net->ds->prm_rprop_delta_minus = deltaMinus;

        Net->ds->prm_normalize_min = normalizeMin;

        Net->ds->prm_normalize_max = delta;

        printf("Testing with normalize max: %f\n",delta);

        MSE = Net->LearStart();

        if ( MSE < *detectedLowestMSE )
        {
            *detectedLowestMSE = MSE;
            detectedIt = Net->Iteration;
            *normalizeMax = Net->ds->prm_normalize_max;
        }

        Net->PrintConclusion();

        delete Net;
    }

    printf("\n\n\tDetected lowest MSE: %.10f at normalize max: %f at itteration : %d\n\n",*detectedLowestMSE,*normalizeMax,detectedIt);
}

void detectNormalizeMin( int argc, char **argv, float *detectedLowestMSE, int detectedBestAlgorithm, int detectedBestLayerCount, int neuronCount, int detectedFn, int deltaMax, float deltaMin, float deltaPlus, float deltaMinus, float *normalizeMin )
{
    float MSE = *detectedLowestMSE;
    int detectedIt = 0;

    for ( float delta = 0.0000001; delta < 0.1; delta *= 10 ) {
        Net = new nnet;

        Process_CommandlineArguments( argc, argv );

        Net->ds->ProcessFileName(fileName);

        Net->LoadDataset( fileName );

        // Net->ds->sample_count *= 0.1;   // While detecting uses on 10% of dataset

        Net->ds->prm_training_algorithm = detectedBestAlgorithm;

        Net->ds->prm_print = 0; // Dont print Iteration

        Net->ds->prm_final_epo = DETECTION_ITERATION_COUNT;

        Net->ds->prm_final_msqe = 0.000000000001;  // Set to unrela value to run through max iteration count

        Net->ds->prm_activation_fn = detectedFn;

        Net->ds->prm_net_layer_count = detectedBestLayerCount;

        Net->ds->prm_net_layer = new int[Net->ds->prm_net_layer_count+1];     // Array element 0 is not utilized

        for ( int j = 1; j <= Net->ds->prm_net_layer_count; j++) {

            Net->ds->prm_net_layer[j] = neuronCount;

        }

        Net->CreateLayers( Net->ds->prm_net_layer_count + 2 );

        Net->SetupLayers( Net->ds->prm_net_layer_count, Net->ds->prm_net_layer );

        Net->ds->prm_rprop_delta_max = deltaMax;
        Net->ds->prm_rprop_delta_min = deltaMin;

        Net->ds->prm_rprop_delta_plus = deltaPlus;
        Net->ds->prm_rprop_delta_minus = deltaMinus;

        Net->ds->prm_normalize_min = delta;

        printf("Testing with normalize min: %f\n",delta);

        MSE = Net->LearStart();

        if ( MSE < *detectedLowestMSE )
        {
            *detectedLowestMSE = MSE;
            detectedIt = Net->Iteration;
            *normalizeMin = Net->ds->prm_normalize_min;
        }

        Net->PrintConclusion();

        delete Net;
    }

    printf("\n\n\tDetected lowest MSE: %.10f at normalize min: %f at itteration : %d\n\n",*detectedLowestMSE,*normalizeMin,detectedIt);
}

void detectDeltaMinus( int argc, char **argv, float *detectedLowestMSE, int detectedBestAlgorithm, int detectedBestLayerCount, int neuronCount, int detectedFn, int deltaMax, float deltaMin, float deltaPlus, float *deltaMinus )
{
    float MSE = *detectedLowestMSE;
    int detectedIt = 0;

    for ( float delta = 0.0000005; delta < 0.5; delta *= 10 ) {
        Net = new nnet;

        Process_CommandlineArguments( argc, argv );

        Net->ds->ProcessFileName(fileName);

        Net->LoadDataset( fileName );

        // Net->ds->sample_count *= 0.1;   // While detecting uses on 10% of dataset

        Net->ds->prm_training_algorithm = detectedBestAlgorithm;

        Net->ds->prm_print = 0; // Dont print Iteration

        Net->ds->prm_final_epo = DETECTION_ITERATION_COUNT;

        Net->ds->prm_final_msqe = 0.000000000001;  // Set to unrela value to run through max iteration count

        Net->ds->prm_activation_fn = detectedFn;

        Net->ds->prm_net_layer_count = detectedBestLayerCount;

        Net->ds->prm_net_layer = new int[Net->ds->prm_net_layer_count+1];     // Array element 0 is not utilized

        for ( int j = 1; j <= Net->ds->prm_net_layer_count; j++) {

            Net->ds->prm_net_layer[j] = neuronCount;

        }

        Net->CreateLayers( Net->ds->prm_net_layer_count + 2 );

        Net->SetupLayers( Net->ds->prm_net_layer_count, Net->ds->prm_net_layer );

        Net->ds->prm_rprop_delta_max = deltaMax;
        Net->ds->prm_rprop_delta_min = deltaMin;

        Net->ds->prm_rprop_delta_plus = deltaPlus;
        Net->ds->prm_rprop_delta_minus = delta;

        printf("Testing with delta minus: %f\n",delta);

        MSE = Net->LearStart();

        if ( MSE < *detectedLowestMSE )
        {
            *detectedLowestMSE = MSE;
            detectedIt = Net->Iteration;
            *deltaMinus = Net->ds->prm_rprop_delta_minus;
        }

        Net->PrintConclusion();

        delete Net;
    }

    printf("\n\n\tDetected lowest MSE: %.10f at delta minus: %f at itteration : %d\n\n",*detectedLowestMSE,*deltaMinus,detectedIt);
}

void detectDeltaPlus( int argc, char **argv, float *detectedLowestMSE, int detectedBestAlgorithm, int detectedBestLayerCount, int neuronCount, int detectedFn, int deltaMax, float deltaMin, float *deltaPlus )
{
    float MSE = *detectedLowestMSE;
    int detectedIt = 0;

    for ( float delta = 0.0000002; delta < 0.2; delta *= 10 ) {
        Net = new nnet;

        Process_CommandlineArguments( argc, argv );

        Net->ds->ProcessFileName(fileName);

        Net->LoadDataset( fileName );

        // Net->ds->sample_count *= 0.1;   // While detecting uses on 10% of dataset

        Net->ds->prm_training_algorithm = detectedBestAlgorithm;

        Net->ds->prm_print = 0; // Dont print Iteration

        Net->ds->prm_final_epo = DETECTION_ITERATION_COUNT;

        Net->ds->prm_final_msqe = 0.000000000001;  // Set to unrela value to run through max iteration count

        Net->ds->prm_activation_fn = detectedFn;

        Net->ds->prm_net_layer_count = detectedBestLayerCount;

        Net->ds->prm_net_layer = new int[Net->ds->prm_net_layer_count+1];     // Array element 0 is not utilized

        for ( int j = 1; j <= Net->ds->prm_net_layer_count; j++) {

            Net->ds->prm_net_layer[j] = neuronCount;

        }

        Net->CreateLayers( Net->ds->prm_net_layer_count + 2 );

        Net->SetupLayers( Net->ds->prm_net_layer_count, Net->ds->prm_net_layer );

        Net->ds->prm_rprop_delta_max = deltaMax;
        Net->ds->prm_rprop_delta_min = deltaMin;

        Net->ds->prm_rprop_delta_plus = 1 + delta;

        printf("Testing with delta plus: %f\n",1+delta);

        MSE = Net->LearStart();

        if ( MSE < *detectedLowestMSE )
        {
            *detectedLowestMSE = MSE;
            detectedIt = Net->Iteration;
            *deltaPlus = Net->ds->prm_rprop_delta_plus;
        }

        Net->PrintConclusion();

        delete Net;
    }

    printf("\n\n\tDetected lowest MSE: %.10f at delta plus: %f at itteration : %d\n\n",*detectedLowestMSE,*deltaPlus,detectedIt);
}

void detectDeltaMin( int argc, char **argv, float *detectedLowestMSE, int detectedBestAlgorithm, int detectedBestLayerCount, int neuronCount, int detectedFn, int deltaMax, float *deltaMin  )
{
    float MSE = *detectedLowestMSE;
    int detectedIt = 0;

    for ( float delta = 0.0000001; delta < 0.1; delta *= 10 ) {
        Net = new nnet;

        Process_CommandlineArguments( argc, argv );

        Net->ds->ProcessFileName(fileName);

        Net->LoadDataset( fileName );

        // Net->ds->sample_count *= 0.1;   // While detecting uses on 10% of dataset

        Net->ds->prm_training_algorithm = detectedBestAlgorithm;

        Net->ds->prm_print = 0; // Dont print Iteration

        Net->ds->prm_final_epo = DETECTION_ITERATION_COUNT;

        Net->ds->prm_final_msqe = 0.000000000001;  // Set to unrela value to run through max iteration count

        Net->ds->prm_activation_fn = detectedFn;

        Net->ds->prm_net_layer_count = detectedBestLayerCount;

        Net->ds->prm_net_layer = new int[Net->ds->prm_net_layer_count+1];     // Array element 0 is not utilized

        for ( int j = 1; j <= Net->ds->prm_net_layer_count; j++) {

            Net->ds->prm_net_layer[j] = neuronCount;

        }

        Net->CreateLayers( Net->ds->prm_net_layer_count + 2 );

        Net->SetupLayers( Net->ds->prm_net_layer_count, Net->ds->prm_net_layer );

        Net->ds->prm_rprop_delta_max = deltaMax;
        Net->ds->prm_rprop_delta_min = delta;

        printf("Testing with delta min: %f\n",delta);

        MSE = Net->LearStart();

        if ( MSE < *detectedLowestMSE )
        {
            *detectedLowestMSE = MSE;
            detectedIt = Net->Iteration;
            *deltaMin = Net->ds->prm_rprop_delta_min;
        }

        Net->PrintConclusion();

        delete Net;
    }

    printf("\n\n\tDetected lowest MSE: %.10f at delta min: %f at itteration : %d\n\n",*detectedLowestMSE,*deltaMin,detectedIt);
}


void detectDeltaMax( int argc, char **argv, float *detectedLowestMSE, int detectedBestAlgorithm, int detectedBestLayerCount, int neuronCount, int detectedFn, int *deltaMax  )
{
    float MSE = *detectedLowestMSE;
    int detectedIt = 0;

    for ( int delta = 100000; delta > 1 ; delta /= 10 ) {
        Net = new nnet;

        Process_CommandlineArguments( argc, argv );

        Net->ds->ProcessFileName(fileName);

        Net->LoadDataset( fileName );

        // Net->ds->sample_count *= 0.1;   // While detecting uses on 10% of dataset

        Net->ds->prm_training_algorithm = detectedBestAlgorithm;

        Net->ds->prm_print = 0; // Dont print Iteration

        Net->ds->prm_final_epo = DETECTION_ITERATION_COUNT;

        Net->ds->prm_final_msqe = 0.000000000001;  // Set to unrela value to run through max iteration count

        Net->ds->prm_activation_fn = detectedFn;

        Net->ds->prm_net_layer_count = detectedBestLayerCount;

        Net->ds->prm_net_layer = new int[Net->ds->prm_net_layer_count+1];     // Array element 0 is not utilized

        for ( int j = 1; j <= Net->ds->prm_net_layer_count; j++) {

            Net->ds->prm_net_layer[j] = neuronCount;

        }

        Net->CreateLayers( Net->ds->prm_net_layer_count + 2 );

        Net->SetupLayers( Net->ds->prm_net_layer_count, Net->ds->prm_net_layer );

        Net->ds->prm_rprop_delta_max = delta;

         printf("Testing with delta max: %d\n",delta);

        MSE = Net->LearStart();

        if ( MSE < *detectedLowestMSE )
        {
            *detectedLowestMSE = MSE;
            detectedIt = Net->Iteration;
            *deltaMax = Net->ds->prm_rprop_delta_max;
        }

        Net->PrintConclusion();

        delete Net;
    }

    printf("\n\n\tDetected lowest MSE: %.10f at delta max: %d at itteration : %d\n\n",*detectedLowestMSE,*deltaMax,detectedIt);
}

void detectNeuronCount( int argc, char **argv, float *detectedLowestMSE, int detectedBestAlgorithm,  int detectedBestLayerCount, int detectedFn, int *neuronCount  )
{
    float MSE = *detectedLowestMSE;
    int detectedIt = 0;

    for ( int nc = 1; nc < ( detectNeuronCountUpTo * 2 ); nc++ ) {
        Net = new nnet;

        Process_CommandlineArguments( argc, argv );

        Net->ds->ProcessFileName(fileName);

        Net->LoadDataset( fileName );

        // Net->ds->sample_count *= 0.1;   // While detecting uses on 10% of dataset

        Net->ds->prm_training_algorithm = detectedBestAlgorithm;

        Net->ds->prm_print = 0; // Dont print Iteration

        Net->ds->prm_final_epo = DETECTION_ITERATION_COUNT;

        Net->ds->prm_final_msqe = 0.000000000001;  // Set to unrela value to run through max iteration count

        Net->ds->prm_activation_fn = detectedFn;

        Net->ds->prm_net_layer_count = detectedBestLayerCount;

        Net->ds->prm_net_layer = new int[Net->ds->prm_net_layer_count+1];     // Array element 0 is not utilized

        for ( int j = 1; j <= Net->ds->prm_net_layer_count; j++) {

            Net->ds->prm_net_layer[j] = nc;

        }

        Net->CreateLayers( Net->ds->prm_net_layer_count + 2 );

        Net->SetupLayers( Net->ds->prm_net_layer_count, Net->ds->prm_net_layer );

        printf("Testing with neuron count: %d\n",nc);

        MSE = Net->LearStart();

        if ( MSE < *detectedLowestMSE )
        {
            *detectedLowestMSE = MSE;
            detectedIt = Net->Iteration;
            *neuronCount = nc;
        }

        Net->PrintConclusion();

        delete Net;
    }

    printf("\n\n\tDetected lowest MSE: %.10f at neuron count: %d at itteration : %d\n\n",*detectedLowestMSE,*neuronCount,detectedIt);
}


void detectAlgorithmLayerCountAndFunction( int argc, char **argv, float *detectedLowestMSE, int *detectedBestAlgorithm, int *detectedBestLayerCount, int *detectedFn  )
{
    float MSE = *detectedLowestMSE;
    int detectedIt = 0;

    for ( int alg = 2; alg < 7; alg++ ) {
        for ( int lc = 1; lc < MAX_LAYER_COUNT; lc++ ) {
            for ( int fn = 0; fn < 4; fn++ ) {

                Net = new nnet;

                Process_CommandlineArguments( argc, argv );

                Net->ds->ProcessFileName(fileName);

                Net->LoadDataset( fileName );

                if ( Net->ds->sample_input_count < 10 ) detectNeuronCountUpTo = NEURON_COUNT_PER_LAYER;
                else detectNeuronCountUpTo = Net->ds->sample_input_count * 2;

                //Net->ds->sample_count *= 0.1;   // While detecting uses on 10% of dataset

                Net->ds->prm_training_algorithm = alg;

                Net->ds->prm_print = 0; // Dont print Iteration

                Net->ds->prm_final_epo = DETECTION_ITERATION_COUNT;

                Net->ds->prm_final_msqe = 0.000000000001;  // Set to unrela value to run through max iteration count

                Net->ds->prm_activation_fn = fn;

                Net->ds->prm_net_layer_count = lc;

                Net->ds->prm_net_layer = new int[Net->ds->prm_net_layer_count+1];     // Array element 0 is not utilized

                for ( int j = 1; j <= Net->ds->prm_net_layer_count; j++) {

                    Net->ds->prm_net_layer[j] = detectNeuronCountUpTo;

                }

                Net->CreateLayers( Net->ds->prm_net_layer_count + 2 );

                Net->SetupLayers( Net->ds->prm_net_layer_count, Net->ds->prm_net_layer );

                MSE = Net->LearStart();

                if ( MSE < *detectedLowestMSE )
                {
                    *detectedLowestMSE = MSE;
                    *detectedBestAlgorithm = alg;
                    *detectedBestLayerCount = lc;
                    *detectedFn = fn;
                    detectedIt = Net->Iteration;
                }

                Net->PrintConclusion();

                delete Net;
            }
        }
    }

    printf("\n\n\tDetected lowest MSE: %.10f at layer count: %d with activation prm_activation_fn: %d at itteration : %d\n\n",*detectedLowestMSE,*detectedBestLayerCount,*detectedFn,detectedIt);
}



int areFloatsEqual1(float a, float b) {
    return fabsf(a - b) < EPSILON_MATCH;
}

void Init() {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("SDL_Init failed: %s\n", SDL_GetError());
        exit(1);
    }

    sdlWindow = SDL_CreateWindow(
        "Terrain Generator v0.01",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        UNIVERSE_WIDTH, UNIVERSE_HEIGHT,
        SDL_WINDOW_SHOWN
    );
    if (!sdlWindow) {
        printf("Window creation failed: %s\n", SDL_GetError());
        SDL_Quit();
        exit(1);
    }

    sdlRenderer = SDL_CreateRenderer(sdlWindow, -1, SDL_RENDERER_ACCELERATED);
    if (!sdlRenderer) {
        printf("Renderer creation failed: %s\n", SDL_GetError());
        SDL_DestroyWindow(sdlWindow);
        SDL_Quit();
        exit(1);
    }

    // Create a texture with TARGET access
    sdlTexture = SDL_CreateTexture(
        sdlRenderer,
        SDL_PIXELFORMAT_RGBA8888,
        SDL_TEXTUREACCESS_TARGET, // Use TARGET for render-to-texture
        UNIVERSE_WIDTH, UNIVERSE_HEIGHT
    );
    if (!sdlTexture) {
        printf("Texture creation failed: %s\n", SDL_GetError());
        SDL_DestroyRenderer(sdlRenderer);
        SDL_DestroyWindow(sdlWindow);
        SDL_Quit();
        exit(1);
    }

    // Set texture as render target and clear it
    if (SDL_SetRenderTarget(sdlRenderer, sdlTexture) != 0) {
        printf("Set render target failed: %s\n", SDL_GetError());
        SDL_DestroyTexture(sdlTexture);
        SDL_DestroyRenderer(sdlRenderer);
        SDL_DestroyWindow(sdlWindow);
        SDL_Quit();
        exit(1);
    }

    SDL_SetRenderDrawColor(sdlRenderer, 0, 0, 0, 255);
    SDL_RenderClear(sdlRenderer);
}

void Run(int sample, float e, float r) {
    if (sample > UNIVERSE_WIDTH - 1) {
        printf("Invalid sample: %d\n", sample);
        return;
    }
    if (e > UNIVERSE_HEIGHT - 1 || r > UNIVERSE_HEIGHT - 1) {
        printf("Invalid e: %f or r: %f\n", e, r);
        return;
    }

    // Ensure texture is the render target
    SDL_SetRenderTarget(sdlRenderer, sdlTexture);

    if (areFloatsEqual1(e, r)) {
        SDL_SetRenderDrawColor(sdlRenderer, 255, 0, 0, 255);
        SDL_RenderDrawPoint(sdlRenderer, sample, (int)r);
    } else {
        SDL_SetRenderDrawColor(sdlRenderer, 0, 255, 255, 255);
        SDL_RenderDrawPoint(sdlRenderer, sample, (int)r);
        SDL_SetRenderDrawColor(sdlRenderer, 0, 0, 255, 255);
        SDL_RenderDrawPoint(sdlRenderer, sample, (int)e);
    }
}

void DrawMsqe(int counter, float msqe) {
    // Ensure texture is the render target
    SDL_SetRenderTarget(sdlRenderer, sdlTexture);

    SDL_SetRenderDrawColor(sdlRenderer, 0, 255, 0, 255);
    int x = counter;
    int y = (int)(logf(msqe) + UNIVERSE_HEIGHT - 100);

    if (x > UNIVERSE_WIDTH - 1 || y > UNIVERSE_HEIGHT - 1) {
        printf("Invalid x: %d or y: %d\n", x, y);
        return;
    }

    SDL_RenderDrawPoint(sdlRenderer, x, y);
}

void Clear() {
    // Clear the texture (not the screen)
    SDL_SetRenderTarget(sdlRenderer, sdlTexture);
    SDL_SetRenderDrawColor(sdlRenderer, 0, 0, 0, 255);
    SDL_RenderClear(sdlRenderer);
}

void Render() {
    // Switch to default render target (screen)
    if (SDL_SetRenderTarget(sdlRenderer, NULL) != 0) {
        printf("Reset render target failed: %s\n", SDL_GetError());
        return;
    }

    // Clear the screen
    SDL_SetRenderDrawColor(sdlRenderer, 0, 0, 0, 255);
    SDL_RenderClear(sdlRenderer);

    // Copy texture to screen
    SDL_Rect dest = {0, 0, UNIVERSE_WIDTH, UNIVERSE_HEIGHT};
    SDL_RenderCopy(sdlRenderer, sdlTexture, NULL, &dest);

    // Present the frame
    SDL_RenderPresent(sdlRenderer);
}

void WaitInput() {
    while (!quit) {
        SDL_PollEvent(&sdlEvent);
        switch (sdlEvent.type) {
            case SDL_MOUSEMOTION:
                // Handle mouse motion if needed
                break;
            case SDL_KEYDOWN:
                switch (sdlEvent.key.keysym.sym) {
                    case SDLK_ESCAPE:
                        quit = 1;
                        break;
                }
                break;
            case SDL_QUIT:
                quit = 1;
                break;
        }
        Render();
    }
}

void Destroy() {
    SDL_DestroyTexture(sdlTexture);
    SDL_DestroyRenderer(sdlRenderer);
    SDL_DestroyWindow(sdlWindow);
    SDL_Quit();
}

int main(int argc, char ** argv)
{
    signal(SIGINT, handle_sigint);

    srand( 12345 );
    // srand(time(NULL));

    Net = new nnet;

    strcpy( (char*)fileName, "csv/rosenbrock.csv");

    Process_CommandlineArguments( argc, argv );

    Net->ds->ProcessFileName(fileName);

    float detectedLowestMSE = 1000000;
    int detectedBestAlgorithm = -1;
    int detectedBestLayerCount = -1;
    int detectedFn = -1;
    int deltaMax = -1;
    float deltaMin = -1;
    float deltaPlus = -1;
    float deltaMinus = -1;
    float normalizeMin = -1;
    float normalizeMax = -1;
    int neuronCount = -1;

    if ( Detect ) {

        detectAlgorithmLayerCountAndFunction( argc, argv, &detectedLowestMSE, &detectedBestAlgorithm, &detectedBestLayerCount, &detectedFn );

        detectedLowestMSE = 1000000;
        detectNeuronCount( argc, argv, &detectedLowestMSE, detectedBestAlgorithm, detectedBestLayerCount, detectedFn, &neuronCount );

        detectedLowestMSE = 1000000;
        detectDeltaMax( argc, argv, &detectedLowestMSE, detectedBestAlgorithm, detectedBestLayerCount, neuronCount, detectedFn, &deltaMax );

        detectedLowestMSE = 1000000;
        detectDeltaMin( argc, argv, &detectedLowestMSE, detectedBestAlgorithm, detectedBestLayerCount, neuronCount, detectedFn, deltaMax, &deltaMin );

        detectedLowestMSE = 1000000;
        detectDeltaPlus( argc, argv, &detectedLowestMSE, detectedBestAlgorithm, detectedBestLayerCount, neuronCount, detectedFn, deltaMax, deltaMin, &deltaPlus );

        detectedLowestMSE = 1000000;
        detectDeltaMinus( argc, argv, &detectedLowestMSE, detectedBestAlgorithm, detectedBestLayerCount, neuronCount, detectedFn, deltaMax, deltaMin, deltaPlus, &deltaMinus );

        detectedLowestMSE = 1000000;
        detectNormalizeMin( argc, argv, &detectedLowestMSE, detectedBestAlgorithm, detectedBestLayerCount, neuronCount, detectedFn, deltaMax, deltaMin, deltaPlus, deltaMinus, &normalizeMin );

        detectedLowestMSE = 1000000;
        detectNormalizeMax( argc, argv, &detectedLowestMSE, detectedBestAlgorithm, detectedBestLayerCount, neuronCount, detectedFn, deltaMax, deltaMin, deltaPlus, deltaMinus, normalizeMin, &normalizeMax );

        printf("Optimal algorithm: %d, architecture: lc: %d, nc: %d, fn: %d, deltaMax: %d, deltaMin: %f, deltaPlus %f, deltaMinus: %f, normMin: %f, normMax: %f\n", detectedBestAlgorithm, detectedBestLayerCount, neuronCount, detectedFn, deltaMax, deltaMin, deltaPlus, deltaMinus, normalizeMin, normalizeMax);
    }

    if ( Learn ) {

        if ( !Net->ds->prm_kubernetes ) Init();

        Net->LoadDataset( fileName );
        if ( !Net->ds->prm_kubernetes ) printf("Dataset Loaded\n");

        Process_CommandlineArguments( argc, argv );

        Net->CreateLayers( Net->ds->prm_net_layer_count + 2 );
        if ( !Net->ds->prm_kubernetes ) printf("Layers created\n");

        Net->SetupLayers( Net->ds->prm_net_layer_count, Net->ds->prm_net_layer );

        if ( !Net->ds->prm_kubernetes ) printf("Layers configured\n");

        Net->LearStart();

        if ( !Net->ds->prm_kubernetes )Net->PrintConclusion();

        Net->SaveNet( Net->ds->fileNamePNN );

        // WaitInput();

        if ( !Net->ds->prm_kubernetes ) Destroy();
    }

    if ( Process ) {

        if ( !Net->ds->prm_kubernetes ) Init();

        Net->LoadNet(Net->ds->fileNamePNN);

        Net->LoadDatasetData(Net->ds->fileNameCSV);

        Net->Process();

        WaitInput();

        // if ( !Net->ds->prm_kubernetes )Net->PrintConclusion();

        if ( !Net->ds->prm_kubernetes ) Destroy();
    }

    delete Net;

    return 0;
}


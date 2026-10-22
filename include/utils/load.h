#pragma once

#include "../include/config/params.h"


double filter[params.FILTER_SIZE];
double img[params.IMG_SIZE];
double output[params.OUTPUT_SIZE];

void load_filter();
void load_img();

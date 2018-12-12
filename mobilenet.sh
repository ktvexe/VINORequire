#!/bin/bash

# Copyright (c) 2018 Intel Corporation
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

error() {
	local code="${3:-1}"
	if [[ -n "$2" ]];then
		echo "Error on or near line $1: $2; exiting with status ${code}"
	else
		echo "Error on or near line $1; exiting with status ${code}"
	fi
	exit "${code}" 
}
trap 'error ${LINENO}' ERR

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR=$ROOT_DIR/..

if [[ $EUID -ne 0 ]]; then
	echo "ERROR: to install CV SDK dependencies, you must run this script as root." >&2
    echo "Please try again with "sudo -E $0", or as root." >&2
    exit 1
fi

model_name="mobilenet"
#target_device="CPU"
#target_device="GPU"
target_device="MYRIAD"
target_precision="FP16"
#target_precision="FP32"
target_image_path="$ROOT_DIR/demo/bulldog.jpg"
#target_image_path="$ROOT_DIR/demo/testdata"


run_again="Then run the script again\n\n"
dashes="\n\n###################################################\n\n"

# Step 1. Download the Caffe model and the prototxt of the model
printf "${dashes}"
printf "\n\nuse predonwload model"

model_dir="${model_name}"
ir_dir="ir/${model_name}"
cur_path=$PWD


# Step 2. Configure Model Optimizer
printf "${dashes}"
printf "Configure Model Optimizer\n\n"

if [[ -z "${INTEL_CVSDK_DIR}" ]]; then
        printf "\n\nINTEL_CVSDK_DIR environment variable is not set. Trying to run ./setvars.sh to set it. \n"

    if [ -e "$ROOT_DIR/inference_engine/bin/setvars.sh" ]; then # for Intel Deep Learning Deployment Toolkit package
        setvars_path="$ROOT_DIR/inference_engine/bin/setvars.sh"
    elif [ -e "$ROOT_DIR/../bin/setupvars.sh" ]; then # for Intel CV SDK package
        setvars_path="$ROOT_DIR/../bin/setupvars.sh"
    elif [ -e "$ROOT_DIR/../setupvars.sh" ]; then # for Intel GO SDK package
        setvars_path="$ROOT_DIR/../setupvars.sh"
    else
        printf "Error: setvars.sh is not found\n"
    fi 
    if ! source $setvars_path ; then
        printf "Unable to run ./setvars.sh. Please check its presence. ${run_again}"
        exit 1
    fi
fi

cvsdk_install_dir="${INTEL_CVSDK_DIR}"

prereqs_mo_path="${cvsdk_install_dir}/deployment_tools/model_optimizer/install_prerequisites"
prereqs_script="install_prerequisites.sh"

if [ ! -e "${prereqs_mo_path}/../venv" ]; then
        cd $prereqs_mo_path
        mkdir "../venv"
        if ! source $prereqs_script ; then
                printf "\n\nUnable to create virtual environment. Do you want to install dependencies globally?\n"
                printf "\nWARNING: this can overwrite your globally installed Python packages.\n"

                read -p "Type 'y' to install dependencies globally or 'n' to exit: " -n 1 -r
                echo    # (optional) move to a new line
                if [[ ! $REPLY =~ ^[Yy]$ ]]
                then
                    [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1 # handle exits from shell or function but don't exit interactive shell
                else 
                        rm -rf ../venv
                        sudo -E $pip_binary install -r ../requirements.txt
                fi
        fi
        cd $cur_path
else
        printf "Found existing environment. Skipping installing dependencies for Model Optimizer.\n"
        printf "If you want to install again, remove venv directory. ${run_again}"
fi
source $prereqs_mo_path/../venv/bin/activate

# Step 3. Convert a model with Model Optimizer
printf "${dashes}"
printf "Convert a model with Model Optimizer\n\n"

mo_path="${cvsdk_install_dir}/deployment_tools/model_optimizer/mo.py"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp
printf "Run $python_binary $mo_path --input_model $ROOT_DIR/demo/classification/mobilenet/mobilenet.caffemodel --output_dir $ir_dir --data_type $target_precision  0\n\n"
$python_binary $mo_path --batch 32 --input_model "$ROOT_DIR/demo/classification/mobilenet/mobilenet.caffemodel" --output_dir $ir_dir --data_type $target_precision
#$python_binary $mo_path --batch 1 --input_model "$ROOT_DIR/demo/classification/resnet_18/resnet_18.caffemodel" --output_dir $ir_dir --data_type $target_precision    & pyflame -o prof.txt -p $!
pid=$!
printf "pid=${pid}"
wait $pid
# Step 4. Build samples
printf "${dashes}"
printf "Build Inference Engine samples\n\n"

samples_path="${cvsdk_install_dir}/deployment_tools/inference_engine/samples"
cd $samples_path

if ! command -v cmake &>/dev/null; then
    printf "\n\nCMAKE is not installed. It is required to build Inference Engine samples. Please install it. ${run_again}"
    exit 1
fi

build_dir="${ROOT_DIR}/inference_engine/samples/build"
mkdir -p $build_dir
cd $build_dir
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8 classification_sample_async

# Step 5. Run samples
printf "${dashes}"
printf "Run Inference Engine classification inception\n\n"

binaries_dir="${ROOT_DIR}/inference_engine/samples/build/intel64/Release"
cd $binaries_dir

printf "Run ./classification_resnet_50 "

./classification_sample_async -d $target_device -i $target_image_path -m "$ROOT_DIR/demo/${ir_dir}/mobilenet.xml" -pc -ni 100 -nt 10 -nireq 2



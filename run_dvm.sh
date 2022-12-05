#!/bin/bash
Displaynumber=1
CODEDIR="$( cd "$( dirname "$0" )" && pwd -P )"
DATADIR="/media/hyomin/HDD6/DATA"

nvidia-docker container run \
	--runtime=nvidia		\
	-it \
	--rm						\
	-e GRANT_SUDO=yes --user root \
	-p 55300:55450 \
	-v ~/.ssh:/root/.ssh		\
	-e DISPLAY=:$Displaynumber \
	--gpus 'all'	\
	--shm-size='200G' \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix" \
	--volume="${CODEDIR}:/code" \
	--volume="${DATADIR}:/DATA" \
	--name lapFusion_test	\
	min00001/dvm_run \
	bash
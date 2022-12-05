#!/bin/bash

# Before execute this script, Ensure to run the pose estimation (preprocessing/fit_smplx.py)
# 
JOBS=(
    # "00114"
    "hyomin_example"
    )
for subj in ${JOBS[@]}; do
    echo "Start processing"
    echo "->" ${subj}
    CONF=$(cat ./scripts/jobs.json | jq '.['"\"${subj}\""']')

    TAR_GENDER=$(echo ${CONF} | jq '.["target_gender"]' | sed 's/\"//g')
    USE_RGBD=$(echo ${CONF} | jq '.["RGBD"]')
    FLAT_HAND=$(echo ${CONF} | jq '.["flathand"]')

    ## Preprocessing (smpl, approx lap coords)
    python ./preprocessing/fit_smplx.py --target_subj ${subj}
    python ./preprocessing/approximate_lap.py --target_subj ${subj}
    LAP_REG_BASE=$(echo ${CONF} | jq '.["lap_reg_weight_base"]')

    ## main processing
    #### learn offset
    LAP_REG_OFFSET=$(echo ${CONF} | jq '.["lap_reg_weight_offset"]')
    NOISE_OFFSET=$(echo ${CONF} | jq '.["noise_offset"]')
    EPOCH_OFFSET=$(echo ${CONF} | jq '.["epoch_offset"]')
    python ./learning/learn_offset.py --target_subj ${subj} --target_gender ${TAR_GENDER} --RGBD ${USE_RGBD} --add_noise ${NOISE_OFFSET} --lap_reg_weight ${LAP_REG_OFFSET} --flathand ${FLAT_HAND}
    python ./inference/pose_blend_base_mesh.py --target_subj ${subj} --target_gender ${TAR_GENDER} --RGBD ${USE_RGBD} --flathand ${FLAT_HAND} --epoch ${EPOCH_OFFSET}

    python ./learning/lap_projecting.py --target_subj ${subj} --target_gender ${TAR_GENDER} --RGBD ${USE_RGBD} --flathand ${FLAT_HAND}

    #### learn details
    NOISE_LAP=$(echo ${CONF} | jq '.["noise_lap"]')
    EPOCH_LAP=$(echo ${CONF} | jq '.["epoch_lap"]')
    python ./learning/learn_details.py --target_subj ${subj} --target_gender ${TAR_GENDER} --RGBD ${USE_RGBD} --add_noise ${NOISE_OFFSET} --flathand ${FLAT_HAND}
    python ./inference/pose_dependent_details.py --target_subj ${subj} --target_gender ${TAR_GENDER} --RGBD ${USE_RGBD} --flathand ${FLAT_HAND} --epoch ${EPOCH_LAP}
done

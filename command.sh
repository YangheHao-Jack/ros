python3 live_seg_diff_node.py \
  --video-output-path           /home/jack/experiments/segmentation_overlay.mp4 \
  --diff-overlay-output-path    /home/jack/experiments/differential_overlay.mp4 \
  --diff-mask-video-output-path /home/jack/experiments/differential_mask.mp4 \
  --mask-output-dir             /home/jack/experiments/seg_masks \
  --topic                        /zed/zed_node/left/image_rect_color/compressed \
  --num-pos                      30 \
  --num-neg                      30 \
  --pth                          0.5 \
  --alpha                        0.3 \
  --kernel-size                  7 \
  --fps                          30.0 \
  --model-path                   .cache/torch/hub/checkpoints/roboreg \
  --model-name                   model.pt \
  --model-url                    https://drive.google.com/uc?id=1_byUJRzTtV5FQbqvVRTeR8FVY6nF87ro \
  --sam2-config                  configs/sam2.1/sam2.1_hiera_l \
  --sam2-checkpoint             ~/repos/sam2/checkpoints/sam2.1_hiera_large.pt \
  --device 
python3 check_camera_base_tf.py \
  /home/jack/experiments/25_05_16_motion_capture \
  --parent-frame base_link \
  --child-frame zed_left_camera_frame
python3 check_camera_base_tf.py /home/jack/experiments/25_05_16_motion_capture \
  --list-only

python3 check_ht.py \
  --ht-init   /home/jack/experiments/transforms/tf_0.npy \
  --ht-final  /home/jack/experiments/transforms/tf_565.npy

python3 extract_zed_lbr.py  /home/jack/experiments/25_05_16_motion_capture /home/jack/experiments 

python3 read_npy.py /home/jack/experiments/25_05_16_roboreg_measure_0

ros2 run tf2_tools view_frames --ros-args -p bagfile:=/home/jack/experiments/25_05_16_motion_capture


python3 extract_zed_lbr2.py  \
  /home/jack/experiments/25_05_16_motion_capture \
  /home/jack/experiments  \
  --camera-info-file /home/jack/experiments/25_05_16_roboreg_measure_1/camera.image.camera_info_4.yaml \
  --sam2-checkpoint ~/repos/sam2/checkpoints/sam2.1_hiera_large.pt \
  --model-path .cache/torch/hub/checkpoints/roboreg \
  --model-name model.pt \
  --num-components 1 \
  --pth 0.5 \
  --alpha 0.3 \
  --device cuda

python3 offline_sam2_extractor.py \
  /home/jack/experiments/25_05_16_motion_capture\
  --out-dir /home/jack/experiments/25_05_21_ouput_extraction_sam2 \
  --image-topic /zed/zed_node/left/image_rect_color/compressed \
  --joint-topic /lbr/joint_states \
  --num-pos 30 \
  --num-neg 30 \
  --num-components 1 \
  --pth 0.5 \
  --alpha 0.3 \
  --device cuda \
  --model-path .cache/torch/hub/checkpoints/roboreg \
  --model-name model.pt \
  --model-url https://drive.google.com/uc?id=1_byUJRzTtV5FQbqvVRTeR8FVY6nF87ro \
  --sam2-config configs/sam2.1/sam2.1_hiera_l \
  --sam2-checkpoint ~/repos/sam2/checkpoints/sam2.1_hiera_large.pt


rr-mono-dr \
    --optimizer SGD \
    --lr 0.01 \
    --max-iterations 100 \
    --display-progress \
    --ros-package lbr_description \
    --xacro-path urdf/med7/med7.xacro \
    --root-link-name lbr_link_0 \
    --end-link-name lbr_link_7 \
    --camera-info-file /home/jack/experiments/25_05_16_roboreg_measure_1/camera.image.camera_info_3.yaml \
    --extrinsics-file /home/jack/experiments/25_05_16_roboreg_measure_0/ht.npy \
    --path /home/jack/experiments/25_06_09_DR_trails_on_video \
    --image-pattern camera_image_left_*.png \
    --joint-states-pattern joint_state_left*.npy \
    --mask-pattern mask_left_*.png \
    --output-file HT_dr.npy
    
python3 offline_tf_check.py \
  /home/jack/experiments/25_05_16_motion_capture \
  --image-topic /zed/zed_node/left/image_rect_color/compressed \
  --tf-topic /tf_static \
  --out-dir /home/jack/experiments/tf_output \
  --tol 1e-6

python3 offline_sam2_extractor.py \
  --bag-dir             /home/jack/experiments/25_05_16_motion_capture \
  --out-dir             /home/jack/experiments/25_05_21_output_extraction_sam2 \
  --camera-info-file    /home/jack/experiments/25_05_16_roboreg_measure_1/camera.image.camera_info_4.yaml \
  --num-pos             30 \
  --num-neg             30 \
  --num-components      1 \
  --pth                 0.5 \
  --alpha               0.3 \
  --device              cuda \
  --sam2-config         configs/sam2.1/sam2.1_hiera_l \
  --sam2-checkpoint     ~/repos/sam2/checkpoints/sam2.1_hiera_large.pt


python3 image_folder_extractor.py \
  --image-dir /home/jack/experiments/25_05_16_roboreg_measure_0  \
  --camera-info-file /home/jack/experiments/25_05_16_roboreg_measure_0/camera.image.camera_info_0.yaml \
  --out-dir /home/jack/experiments/25_05_16_roboreg_measure_0 \
  --sam2-checkpoint ~/repos/sam2/checkpoints/sam2.1_hiera_large.pt \
  --device cuda






rr-mono-dr \
    --optimizer Adam \
    --lr 0.01 \
    --max-iterations 1000 \
    --display-progress \
    --ros-package lbr_description \
    --xacro-path urdf/med7/med7.xacro \
    --root-link-name lbr_link_0 \
    --end-link-name lbr_link_7 \
    --camera-info-file /home/jack/experiments/25_05_16_roboreg_measure_0/camera.image.camera_info_3.yaml \
    --extrinsics-file /home/jack/experiments/25_05_16_roboreg_measure_0/ht.npy \
    --path /home/jack/experiments/25_06_09_DR_trails_on_video \
    --image-pattern camera_image_left_*.png \
    --joint-states-pattern joint_state_left*.npy \
    --mask-pattern mask_left_*.png \
    --mode segmentation \
    --output-file HT_dr.npy
    
git update-index --assume-unchanged /home/jack/roboreg/roboreg/cli/rr_mono_dr.py

./rr-mono-dr.py \
    --optimizer Adam \
    --lr 0.01 \
    --max-iterations 100 \
    --display-progress \
    --ros-package lbr_description \
    --xacro-path urdf/med7/med7.xacro \
    --root-link-name lbr_link_0 \
    --end-link-name lbr_link_7 \
    --camera-info-file /home/jack/experiments/25_05_16_roboreg_measure_0/camera.image.camera_info_3.yaml \
    --extrinsics-file /home/jack/output/refined_extrinsics/average/avg_camera_to_base.npy \
    --path /home/jack/experiments/25_06_09_DR_trails_on_video \
    --image-pattern camera_image_left_*.png \
    --joint-states-pattern joint_state_left*.npy \
    --mask-pattern mask_left_*.png \
    
./rr-mono-dr3.py \
    --optimizer AdamW \
    --lr 0.001 \
    --max-iterations 1000 \
    --display-progress \
    --ros-package lbr_description \
    --xacro-path urdf/med7/med7.xacro \
    --root-link-name lbr_link_0 \
    --end-link-name lbr_link_7 \
    --camera-info-file /home/jack/experiments/25_05_16_roboreg_measure_0/camera.image.camera_info_3.yaml \
    --extrinsics-file /home/jack/experiments/25_05_16_roboreg_measure_0/ht.npy \
    --path /home/jack/experiments/25_06_09_DR_trails_on_video \
    --image-pattern camera_image_left_*.png \
    --joint-states-pattern joint_state_left*.npy \
    --mask-pattern mask_left_*.png \
    --mode segmentation

./rr-mono-dr4.py \
    --optimizer scipy_lbfgs \
    --lr 0.001 \
    --max-iterations 1000 \
    --display-progress \
    --ros-package lbr_description \
    --xacro-path urdf/med7/med7.xacro \
    --root-link-name lbr_link_0 \
    --end-link-name lbr_link_7 \
    --camera-info-file /home/jack/experiments/25_05_16_roboreg_measure_0/camera.image.camera_info_3.yaml \
    --extrinsics-file /home/jack/experiments/25_05_16_roboreg_measure_0/ht.npy \
    --path /home/jack/experiments/25_06_09_DR_trails_on_video \
    --image-pattern camera_image_left_*.png \
    --joint-states-pattern joint_state_left*.npy \
    --mask-pattern mask_left_*.png \
    --mode segmentation
    

./rr-mono-dr4.py \
    --optimizer scipy_lbfgs \
    --lr 0.001 \
    --max-iterations 1000 \
    --display-progress \
    --ros-package lbr_description \
    --xacro-path urdf/med7/med7.xacro \
    --root-link-name lbr_link_0 \
    --end-link-name lbr_link_7 \
    --camera-info-file /home/jack/experiments/25_05_16_roboreg_measure_0/zed_right_camera_info.yaml \
    --extrinsics-file /home/jack/experiments/25_05_16_roboreg_measure_0/ht_right.npy \
    --path /home/jack/experiments/25_06_12_DR_trails_on_video_right \
    --image-pattern camera_image_right_*.png \
    --joint-states-pattern joint_state_right*.npy \
    --mask-pattern mask_right_*.png \
    --mode segmentation

./rr-mono-dr3.py \
  --optimizer AdamW \
  --lr 0.1 \
  --weight-decay 1e-4 \
  --scheduler-type cosine \
  --patience 20 \
  --clip-norm 1.0 \
  --gamma 0.5 \
  --max-iterations 2000 \
  --display-progress \
  --ros-package lbr_description \
  --xacro-path urdf/med7/med7.xacro \
  --root-link-name lbr_link_0 \
  --end-link-name lbr_link_7 \
  --camera-info-file /home/jack/experiments/25_05_16_roboreg_measure_0/camera.image.camera_info_3.yaml \
  --extrinsics-file /home/jack/experiments/25_05_16_roboreg_measure_0/ht.npy \
  --final-extrinsics-file /home/jack/experiments/25_05_16_roboreg_measure_1/ht.npy \
  --path /home/jack/experiments/25_06_09_DR_trails_on_video \
  --image-pattern camera_image_left_*.png \
  --joint-states-pattern joint_state_left*.npy \
  --mask-pattern mask_left_*.png \
  --mode segmentation

rr-mono-dr \
    --optimizer AdamW \
    --lr 0.001 \
    --max-iterations 2000 \
    --display-progress \
    --ros-package lbr_description \
    --xacro-path urdf/med7/med7.xacro \
    --root-link-name lbr_link_0 \
    --end-link-name lbr_link_7 \
    --camera-info-file /home/jack/experiments/25_05_16_roboreg_measure_0/camera.image.camera_info_3.yaml \
    --extrinsics-file /home/jack/experiments/25_05_16_roboreg_measure_0/ht.npy \
    --path /home/jack/experiments/25_06_09_DR_trails_on_video \
    --image-pattern camera_image_left_*.png \
    --joint-states-pattern joint_state_left*.npy \
    --mask-pattern mask_left_*.png \
    --mode segmentation

./rr-stereo-dr.py \
  --optimizer        scipy_lbfgs \
  --lr               1e-4 \
  --max-iterations   300 \
  --mode             segmentation \
  --display-progress \
  --ros-package      lbr_description \
  --xacro-path       urdf/med7/med7.xacro \
  --root-link-name   lbr_link_0 \
  --end-link-name    lbr_link_7 \
  --collision-meshes \
  --camera-info-file         /home/jack/experiments/25_05_16_roboreg_measure_0/camera.image.camera_info_3.yaml  \
  --extrinsics-file          /home/jack/experiments/25_05_16_roboreg_measure_0/ht.npy  \
  --right-camera-info-file   /home/jack/experiments/25_05_16_roboreg_measure_0/zed_right_camera_info.yaml \
  --right-extrinsics-file    /home/jack/experiments/25_05_16_roboreg_measure_0/ht_right.npy\
  --path             /home/jack/experiments/25_06_12_DR_trails_on_video_stereo \
  --image-pattern   camera_image_left_*.png \
  --right-image-pattern  camera_image_right_*.png \
  --mask-pattern         mask_left_*.png \
  --right-mask-pattern   mask_right_*.png \
  --joint-states-pattern joint_state_left_*.npy \
  --output-file       HT_left_stereo_dr.npy \
  --right-output-file HT_right_stereo_dr.npy \
  --max-jobs          2

./refine_cam2base.py \
  --handeye         /home/jack/experiments/25_05_16_roboreg_measure_0/ht.npy \
  --camera-info     /home/jack/experiments/25_05_16_roboreg_measure_0/camera.image.camera_info_3.yaml \
  --ros-package     lbr_description \
  --xacro           urdf/med7/med7.xacro \
  --root-link       lbr_link_0 \
  --end-link        lbr_link_7 \
  --masks-dir       /home/jack/experiments/segmentation/masks/left \
  --joints-dir      /home/jack/experiments/joint_states/left \
  --imgs-dir        /home/jack/experiments/images/left \
  --output-dir      /home/jack/output/refined_extrinsics \
  --mode            segmentation \
  --optimizer       AdamW \
  --lr              0.001 \
  --iters           5 \
  --device          cuda


python3 check_ht.py   --ht-init   /home/jack/experiments/25_06_09_DR_trails_on_video/per_frame/camera_to_base_1.npy --ht-final  /home/jack/experiments/25_06_09_DR_trails_on_video/per_frame/camera_to_base_200.npy

python3 print_npy_joints.py /home/jack/experiments/25_05_16_roboreg_measure_1/joint_states_0.npy

python3 print_npy_joints.py /home/jack/experiments/joint_states/left/joint_state_left_00.npy

./left_to_right.py \
  --bag-dir        /home/jack/experiments/25_05_16_motion_capture \
  --extrinsics-left /home/jack/experiments/25_05_16_roboreg_measure_0/ht.npy \
  --common-frame   zed_camera_center \
  --left-frame     zed_left_camera_frame \
  --right-frame    zed_right_camera_frame \
  --output-file    /home/jack/experiments/25_05_16_roboreg_measure_0/ht_right.npy

./left_to_right_camera.py \
  --bag-dir           /home/jack/experiments/25_05_16_motion_capture \
  --extrinsics-left   /home/jack/experiments/25_05_16_roboreg_measure_0/ht.npy \
  --camera-info-left  /home/jack/experiments/25_05_16_roboreg_measure_0/camera.image.camera_info_3.yaml \
  --common-frame      zed_camera_center \
  --left-frame        zed_left_camera_frame \
  --right-frame       zed_right_camera_frame \
  --camera-info-right /home/jack/experiments/25_05_16_roboreg_measure_0/zed_right_camera_info.yaml
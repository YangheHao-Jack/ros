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
    --optimizer SGD\
    --lr 0.001 \
    --max-iterations 1000 \
    --display-progress \
    --ros-package lbr_description \
    --xacro-path urdf/med7/med7.xacro \
    --root-link-name lbr_link_0 \
    --end-link-name lbr_link_7 \
    --camera-info-file /home/jack/experiments/25_05_16_roboreg_measure_1/camera.image.camera_info_4.yaml \
    --extrinsics-file /home/jack/experiments/25_05_16_roboreg_measure_1/ht.npy \
    --path /home/jack/experiments/25_05_16_roboreg_measure_1 \
    --image-pattern left_image_reg_*.png \
    --joint-states-pattern joint_states_*.npy \
    --mask-pattern mask_sam2_reg_*.png \
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
  --image-dir /home/jack/experiments/25_05_16_roboreg_measure_1  \
  --camera-info-file /home/jack/experiments/25_05_16_roboreg_measure_1/camera.image.camera_info_4.yaml \
  --out-dir /home/jack/experiments/25_05_16_roboreg_measure_1 \
  --sam2-checkpoint ~/repos/sam2/checkpoints/sam2.1_hiera_large.pt \
  --device cuda
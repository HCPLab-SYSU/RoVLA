#!/bin/bash
# set -x -e
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
tasks=(
    adjust_bottle
    handover_mic
    pick_dual_bottles
    place_fan
    scan_object
    hanging_mug
    place_a2b_left
    place_mouse_pad
    shake_bottle_horizontally
    beat_block_hammer
    place_a2b_right
    place_object_basket
    shake_bottle
    blocks_ranking_rgb
    lift_pot
    place_bread_basket
    place_object_scale
    stack_blocks_three
    blocks_ranking_size
    move_can_pot
    place_bread_skillet
    place_object_stand
    stack_blocks_two
    click_alarmclock
    move_pillbottle_pad
    place_burger_fries
    place_phone_stand
    stack_bowls_three
    click_bell
    move_playingcard_away
    place_can_basket
    place_shoe
    stack_bowls_two
    dump_bin_bigbin
    move_stapler_pad
    place_cans_plasticbox
    press_stapler
    stamp_seal
    open_laptop
    place_container_plate
    put_bottles_dustbin
    turn_switch
    grab_roller
    open_microwave
    place_dual_shoes
    put_object_cabinet
    handover_block
    pick_diverse_bottles
    place_empty_cup
    rotate_qrcode
)
cd ./scripts/lerobot_conversion
ROBOTWIN2_DATA_DIR=/vla/users/luojingzhou/data/datasets/robotwin2_aloha_agilex_randomized_500
ROBOTWIN2_LEROBOT_DATA_DIR=/vla/users/luojingzhou/data/datasets/robotwin2_aloha_agilex_randomized_500_lerobotV2
for task in "${tasks[@]}"; do
    python convert_robotwin2_to_v2.py --input_dir $ROBOTWIN2_DATA_DIR/$task/demo_randomized --output_dir $ROBOTWIN2_LEROBOT_DATA_DIR/$task
done

ROBOTWIN2_DATA_DIR=/vla/users/luojingzhou/data/datasets/robotwin2_aloha_agilex_clean
ROBOTWIN2_LEROBOT_DATA_DIR=/vla/users/luojingzhou/data/datasets/robotwin2_aloha_agilex_clean_lerobotV2
for task in "${tasks[@]}"; do
    python convert_robotwin2_to_v2.py --input_dir $ROBOTWIN2_DATA_DIR/$task/demo_clean --output_dir $ROBOTWIN2_LEROBOT_DATA_DIR/$task
done
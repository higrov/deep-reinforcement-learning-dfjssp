#!/bin/bash

levels=("open-divider_salad", "partial-divider_salad" "full-divider_salad" "block-divider_salad" "ring-divider_salad" "open-divider_onion-salad" "block-divider_onion-salad")

#levels=("open-divider_salad", "partial-divider_salad" "full-divider_salad" "cross-divider_salad", "block-divider_salad", "ring-divider_salad" "open-divider_onion-salad" "block-divider_onion-salad")

nagents=1
config="evaluation/1x3"

for level in "${levels[@]}"; do
    echo python main.py --config $config --num-agents $nagents --level $level
    python main.py --num-agents $nagents --level $level
    sleep 5
    done
done

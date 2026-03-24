#!/bin/bash

img_names=(
    CD0 CD1 CD2 CD3
    CI1 CI2 CI3
    FD0 FD1 FD2 FD3
    FI1 FI2 FI3
    SD0 SD1 SD2 SD3
)

for img_name in "${img_names[@]}"; do
    python ldi_inpaiting.py --config "Antoine/ldi.yaml" --img_name "$img_name"
done
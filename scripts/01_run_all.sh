#!/bin/bash
# Run multiple experiments for SphericalDreamer

repo_dir="/home/a.schnepf/phd/SphericalDreamer"
script="$repo_dir/spherical_independant_dreams.py"

# Ensure the script exists
if [ ! -f "$script" ]; then
    echo "❌ Error: Python script not found at $script"
    exit 1
fi

# Run experiments
for exp_id in 0 1 2 3; do
    echo "🚀 Running experiment $exp_id..."
    python "$script" --exp_id "$exp_id" 
done

echo "✅ All experiments completed successfully."
ns-export gaussian-splat \
    --load-config OUTPUTS/SphericalDreamerRecurse/forest_v3/nerfstudio_chkpt/forest_v3/splatfacto/2025-12-30_085122/config.yml \
    --output-dir OUTPUTS/SphericalDreamerRecurse/forest_v3/splats/splats.ply


# IMPORTANT TODO: 
# this code must automaticall select the latest checkpoint for export
# Once the .ply has been generated, in splats/splats.ply, WE MUST also copy the associated dataparser_transform.json
cp -r ../../../dataloaders/3W/loader ./
cp -r ../../../models/tsdiffusion_transformed ./
${container_engine} build -t 3w:diffusion_transformed .
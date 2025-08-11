cp -r ../../../dataloaders/3W/loader ./
cp -r ../../../modules/models/tsdiffusion ./
${container_engine} build -t 3w:diffusion .
cp -r ../../../dataloaders/3W/loader ./
cp -r ../../../models/tsdiffusion ./
${container_engine} build -t 3w:capsnet .
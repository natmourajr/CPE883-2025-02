cp -r ../../../dataloaders/3W/loader ./
cp -r ../../../models/timeCaps ./
${container_engine} build -t 3w:diffusion .
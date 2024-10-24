export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/maxdiffusion-1023
docker tag maxdiffusion_base_image $LOCAL_IMAGE_NAME; docker push $LOCAL_IMAGE_NAME;
docker build -t my-rocm-pytorch-app .
docker run --rm -it --device=/dev/kfd --device=/dev/dri --group-add video my-rocm-pytorch-app
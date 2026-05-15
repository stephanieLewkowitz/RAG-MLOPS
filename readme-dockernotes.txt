1. Build docker- (starts with nvidia cuda image, but not necessary with gemini)
docker build --no-cache --progress=plain -t rag-gpu-jupyter .

2. Run docker
docker run --gpus all -it -p 8008:8008 -v "$(pwd):/notebooks" rag-gpu-jupyter
After run starts, a link for jupyter notebook is created 




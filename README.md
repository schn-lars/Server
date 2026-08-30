# Mr. Intenso - Server

This repository contains the backend of the iOS application ``Mr. Intenso``. (https://github.com/schn-lars/MrIntenso)

## Setup
The setup of the entire backend consists of setting up four Docker containers. These Docker containers must communicate with eachother.
We have solved this by connecting them to the same network. The setup is split into the different containers we have: ``bird-db``, ``location-db``, ``api`` and ``spark``.
Additionally, we have provided ``mock.env``-files in all directories. These will help you structure your passwords and usernames for the databases.

### Requirements
1. ``docker network create network``
2. Make sure, you have added an ``.env``-file in each directory where a ``mock.env`` has been placed and fill it out accordingly.

### Bird-DB
1. ``docker build -t bird-db ./database``
2. ``docker run -d --env-file ./database/.env --name bird-db --network network -p 5432:5432 -v bird_pgdata:/var/lib/postgresql/data bird-db``
3. ``nohup python3 ./database/integrate.py > output.log 2>&1 &``

### Location-DB
1. ``docker build -t location-db ./location``
2. ``docker run -d --env-file ./location/.env --name location-db --network network -p 6543:5432 -v location_pgdata:/var/lib/postgresql/data location-db``
3. ``nohup python3 ./location/locate.py > output.log 2>&1 &``
4. ``nohup python3 ./location/coordCalculator.py > output.log 2>&1 &``

### Spark
1. ``docker build -t spark ./spark``
2. ``docker run -d --name spark --network network -p 5050:5050 spark``

### API
1. ``docker build -t api ./api``
2. ``docker run -d --env-file ./api/.env --name api --network network --mount type=bind,source="$(pwd)/api",target=/app -p 6969:6969 api``

### Inference
1. Clone mobile sam repo onto node: ``git clone https://github.com/ChaoningZhang/MobileSAM.git``

### Smol, Qwen and minicpm
This container is designed to run in an environment with access to a GPU.
Since we are using NVIDIA GeForce RTX 4090, we needed to run some commands before.
#### Prerequisites:
1. Follow https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#with-apt-ubuntu-debian
2. Verify your Docker-container toolkit for nvidia using: ``docker run --rm --gpus all nvidia/cuda:12.2.0-runtime-ubuntu22.04 nvidia-smi``. If you see your GPU, then you can continue.
#### Actual Container:
1. ``cd smol``
2. ``sudo docker build --no-cache -t smol .``
3. ``sudo docker run --name smol --gpus all -p 8000:8000 smol``
4. Repeat for the other AI models as well. Beware that all the models run on the same port.
In case you want to run them in parallel, you need to add more endpoints.

After all this is done, the containers should be connected to each other and the API is accepting requests.

## Resources
- Data for: [Location-DB](https://www.swisstopo.admin.ch/de/amtliches-verzeichnis-der-gebaeudeadressen\#Download)
- Data for [Bird-DB](https://www.infospecies.ch/de/)
- Bird-Classifier used: [www.huggingface.co](https://huggingface.co/chriamue/bird-species-classifier)
- FastAPI: [Documentation](https://fastapi.tiangolo.com/)

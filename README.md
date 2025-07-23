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

After all this is done, the containers should be connected to each other and the API is accepting requests.

## Resources
- Data for: [Location-DB](https://www.swisstopo.admin.ch/de/amtliches-verzeichnis-der-gebaeudeadressen\#Download)
- Data for [Bird-DB](https://www.infospecies.ch/de/)
- Bird-Classifier used: [www.huggingface.co](https://huggingface.co/chriamue/bird-species-classifier)
- FastAPI: [Documentation](https://fastapi.tiangolo.com/)

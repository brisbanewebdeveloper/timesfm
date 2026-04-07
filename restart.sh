#!/bin/bash

docker compose down
# docker compose up -d --remove-orphans --build
docker compose up -d --remove-orphans
docker compose logs -f

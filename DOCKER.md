# Docker Usage

## Prerequisites
- Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Start Docker Desktop

## Commands

### Build and run
```bash
docker-compose up
```

### Rebuild after code changes
```bash
docker-compose up --build
```

### Run in background
```bash
docker-compose up -d
docker-compose logs -f  # View logs
```

### Stop the app
```bash
docker-compose down
```

### Remove everything (including volumes)
```bash
docker-compose down -v
```

## Access the app
Open browser: http://localhost:5000

## What persists
- `data/` folder — your generated datasets
- `models/` folder — saved model artifacts

These are mounted as volumes, so they survive container restarts.
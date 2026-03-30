MODEL_DIR=./models

build:
	docker-compose build

run:
	docker-compose up -d

stop:
	docker-compose down

convert:
	docker build -t cosyvoice-convert -f Dockerfile.convert .
	docker run --gpus all \
		-v $(MODEL_DIR):/models \
		cosyvoice-convert

clean:
	rm -rf models/*
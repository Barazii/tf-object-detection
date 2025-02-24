AWS_REGION = eu-north-1
ACCOUNT_ID = 482497089777
REPO_NAME = opencv
IMAGE_TAG = latest

REPO_URI = $(ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(REPO_NAME):$(IMAGE_TAG)

.PHONY: login build tag push

login:
	@echo "Logging into AWS ECR..."
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com

build:
	@echo "Building Docker image..."
	docker build -t $(REPO_NAME):$(IMAGE_TAG) .

tag:
	@echo "Tagging Docker image..."
	docker tag $(REPO_NAME):$(IMAGE_TAG) $(REPO_URI)

push: login build tag
	@echo "Pushing Docker image to ECR..."
	docker push $(REPO_URI)

help:
	@echo "Makefile commands:"
	@echo "  login  - Login to AWS ECR"
	@echo "  build  - Build the Docker image"
	@echo "  tag    - Tag the Docker image"
	@echo "  push   - Build, tag, and push the Docker image to ECR"
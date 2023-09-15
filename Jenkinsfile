pipeline {
    agent any

    options{
        buildDiscarder(logRotator(numToKeepStr: '5', daysToKeepStr: '5'))
        timestamps()
    }

    environment{
        registry = 'duong05102002/text-image-retrieval-serving'
        registryCredential = 'dockerhub'
    }

    stages {
        stage('Test') {
            agent {
                docker {
                    image 'python:3.8'
                }
            }
            steps {
                echo 'Testing model correctness..'
            }
        }
        stage('Build') {
            steps {
                script {
                    echo 'Building image for deployment..'
                    def imageName = "${registry}:${BUILD_NUMBER}"
                    def buildArgs = "--build-arg PINECONE_APIKEY=${PINECONE_APIKEY}"
                    // dockerImage = docker.build registry + ":$BUILD_NUMBER" + "--build-arg PINECONE_APIKEY=$PINECONE_APIKEY"
                    dockerImage = docker.build(imageName, "--file Dockerfile ${buildArgs} .")
                    echo 'Pushing image to dockerhub..'
                    docker.withRegistry( '', registryCredential ) {
                        dockerImage.push()
                    }
                }
            }
        }
        stage('Deploy') {
            steps {
                echo 'Deploying models..'
            }
        }
    }
}

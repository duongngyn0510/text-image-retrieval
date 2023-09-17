pipeline {
    agent any

    options{
        buildDiscarder(logRotator(numToKeepStr: '5', daysToKeepStr: '5'))
        timestamps()
    }

    environment{
        registry = 'duong05102002/retrieval-local-service'
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
                    def imageName = "${registry}:v1.${BUILD_NUMBER}"
                    def buildArgs = "--build-arg PINECONE_APIKEY=${PINECONE_APIKEY}"
                    // PINECONE_APIKEY env is set up on Jenkins dashboard

                    dockerImage = docker.build(imageName, "--file Dockerfile-jenkin ${buildArgs} .")
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

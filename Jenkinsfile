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
                sh 'echo ${PINECONE_APIKEY}'
            }
        }
        stage('Build') {
            steps {
                script {
                    // dockerImage = docker.build registry + "--build-arg PINECONE_APIKEY=$PINECONE_APIKEY" + ":v1.$BUILD_NUMBER"
                    def dockerImage = docker.build("registry:${BUILD_NUMBER}", "--build-arg PINECONE_APIKEY=${PINECONE_APIKEY} .")
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

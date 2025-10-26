pipeline {
   agent any


   environment {
       VENV_DIR = 'venv'
       GCP_PROJECT = 'mlops-project-1809'
       GCLOUD_PATH = '/google-cloud-sdk/bin'
   }


   stages {
       stage('Clone Github repo to Jenkins') {
           steps {
               script {
                   echo 'Cloning Github repo to Jenkins............'
                   checkout scmGit(
                       branches: [[name: '*/main']],
                       extensions: [],
                       userRemoteConfigs: [[
                           credentialsId: 'jenkins-github-token',
                           url: 'https://github.com/andd-1809/MLOPS-COURSE-PROJECT-01.git'
                       ]]
                   )
               }
           }
       }


       stage('Setting up our Virtual Environment and Installing Dependencies') {
           steps {
               script {
                   echo 'Cloning Github repo to Jenkins............'
                   sh '''
                       python3 -m venv ${VENV_DIR}
                       . ${VENV_DIR}/bin/activate
                       pip install --upgrade pip
                       pip install -e .
                   '''
               }
           }
       }


       stage('Building and Pushing Docker Image to GCR') {
           steps {
               withCredentials([file(credentialsId: 'gcp-key', variable: 'GOOGLE_APPLICATION_CREDENTIALS')]) {
                   script {
                       echo 'Building and Pushing Docker Image to GCR............'
                       sh '''
                           export PATH=$PATH:${GCLOUD_PATH}

                           gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}

                           gcloud config set project ${GCP_PROJECT}

                           gcloud auth configure-docker --quiet

                           IMAGE_NAME=gcr.io/${GCP_PROJECT}/ml-project:latest

                           docker build -t $IMAGE_NAME .

                           docker push $IMAGE_NAME
                       '''
                   }
               }
           }
       }


       stage('Deploy to Google Cloud Run') {
           steps {
               withCredentials([file(credentialsId: 'gcp-key', variable: 'GOOGLE_APPLICATION_CREDENTIALS')]) {
                   script {
                       echo 'Deploy to Google Cloud Run............'
                       sh '''
                           export PATH=$PATH:${GCLOUD_PATH}

                           gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}

                           gcloud config set project ${GCP_PROJECT}

                           IMAGE_NAME=gcr.io/${GCP_PROJECT}/ml-project:latest

                           gcloud run deploy ml-project \
                               --image=$IMAGE_NAME \
                               --platform=managed \
                               --region=us-central1 \
                               --allow-unauthenticated
                          
                       '''
                   }
               }
           }
       }
   }


   post {
       always {
           echo 'This will always run after the stages.'
       }
       success {
           echo 'This will run only if the pipeline succeeds.'
       }
       failure {
           echo 'This will run only if the pipeline fails.'
       }
   }
}
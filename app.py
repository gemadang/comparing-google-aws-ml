#export GOOGLE_APPLICATION_CREDENTIALS=baeri-test-project-77e8c8183336.json

from flask import Flask, render_template, request
import requests
import urllib, json
import httplib2
from apiclient.discovery import build
from oauth2client.client import GoogleCredentials

from googleapiclient import errors
from googleapiclient import discovery

import boto3

client = boto3.client('sagemaker')
DISCOVERY_URL = "https://ml.googleapis.com/$discovery/rest?version=v1"
app = Flask(__name__)
'''
http = httplib2.Http()
credentials = GoogleCredentials.get_application_default().create_scoped(
    ['http://www.googleapis.com/auth/cloud-platform']
)
http = httplib2.Http()
credentials.authorize(http)
'''
#################################### GOOGLE ML ENGINE ####################################
def get_mlengine_model(test):
    if test:
        return "test"
    ml = discovery.build('ml', 'v1')

    projectID_model = 'projects/{}'.format('baeri-test-project') + '/models/{}'.format('census')

    # Create a request to call projects.models.get
    request = ml.projects().models().get(name=projectID_model)

    # Make the call.
    try:
        response = request.execute()
        print(response)
        return response
    except errors.HttpError, err:
        # Something went wrong, print out some information.
        print('There was an error creating the model. Check the details:')
        print(err._get_reason())
    
    return "poops" 

def get_mlengine_job(test, job_name):
    if test:
        return "test"

    ml = discovery.build('ml', 'v1')

    projectID_model = 'projects/{}'.format('baeri-test-project') + '/jobs/{}'.format(job_name)

    # Create a request to call projects.models.get
    request = ml.projects().jobs().get(name=projectID_model)

    # Make the call.
    try:
        response = request.execute()
        print(response)
        return response
    except errors.HttpError, err:
        # Something went wrong, print out some information.
        print('There was an error creating the model. Check the details:')
        print(err._get_reason())
    
    return "poops" 

#################################### GOOGLE COMPUTE ENGINE ####################################




#################################### AWS SAGEMAKER ####################################
def invoke_sagemaker_endpoint(test):
    if test:
        return "test"

    response = client.invoke_endpoint(
        EndpointName='kmeans-2018-06-26-16-51-37-635',
        Body=b'bytes'|file,
        ContentType='string',
        Accept='string'
    )

def get_sagemaker_models(test):
    if test:
        return ("test", "test")

    response = client.list_models(
        SortBy='CreationTime',
        SortOrder='Descending',
        #NextToken='string',
        MaxResults=10,
        #NameContains='string',
        #CreationTimeBefore=datetime(2015, 1, 1),
        #CreationTimeAfter=datetime(2015, 1, 1)
    )

    description_list = []

    for model in response["Models"]:
        description = client.describe_model(
            ModelName= model["ModelName"]
        )
        print(description)
        description_list.append(description)
    
    return (response, description_list)
    

def get_sagemaker_jobs(test):
    if test:
        return ("test", "test")
    
    response = client.list_training_jobs(
        #NextToken='ListTrainingJobs',
        MaxResults=10,
        #CreationTimeAfter=datetime(2015, 1, 1),
        #CreationTimeBefore=datetime(2015, 1, 1),
        #LastModifiedTimeAfter=datetime(2015, 1, 1),
        #LastModifiedTimeBefore=datetime(2015, 1, 1),
        #NameContains='string',
        StatusEquals='Completed',
        SortBy='Name',
        SortOrder='Ascending'
    )

    description_list = []

    for job in response["TrainingJobSummaries"]:
        description = client.describe_training_job(
            TrainingJobName=job["TrainingJobName"]
        )
        description_list.append(description)
    
    return (response, description_list)

def get_sagemaker_hyper_paramater(test):

    if test:
        return ("test", "test")


    response = client.list_hyper_parameter_tuning_jobs(
        #NextToken='string',
        MaxResults=10,
        SortBy='Status',
        SortOrder='Descending',
        #NameContains='string',
        #CreationTimeAfter=datetime(2015, 1, 1),
        #CreationTimeBefore=datetime(2015, 1, 1),
        #LastModifiedTimeAfter=datetime(2015, 1, 1),
        #LastModifiedTimeBefore=datetime(2015, 1, 1),
        #StatusEquals='Completed'|'InProgress'|'Failed'|'Stopped'|'Stopping'
    )

    description_list = []

    for job in response["HyperParameterTuningJobSummaries"]:
        description = client.describe_hyper_parameter_tuning_job(
            HyperParameterTuningJobName=job["HyperParameterTuningJobName"]
        )

        description_list.append(description)
    
    return (response, description_list)




#################################### AWS EC2 ####################################


@app.route('/')
def main():
    '''Display the metrics
        CPU/ GPU/ memory usage
        Training/ Testing/ Validation Time
        Prediction Time
        Cost
    '''
    dont_call_api = True

    (sage_model, sage_model_description) = get_sagemaker_models(dont_call_api)
    (sage_job, sage_job_description) = get_sagemaker_jobs(dont_call_api)
    (sage_hyperparameter_job, sage_hyperparameter_description) = get_sagemaker_hyper_paramater(dont_call_api)

    #print(json.dumps(sage_hyperparameter_job, indent=1, sort_keys=True, default=str))

    mlengine_model = get_mlengine_model(dont_call_api)
    mlengine_job_single = get_mlengine_job(dont_call_api, 'census_single_2')
    mlengine_job_distributed = get_mlengine_job(dont_call_api, 'census_dist_1')
    mlengine_hyperparam = get_mlengine_job(dont_call_api, 'census_core_hptune_1')
    mlengine_predict = get_mlengine_job(dont_call_api, 'census_prediction_1')

    return render_template('index.html', sage_model=sage_model,
                            sage_model_description=sage_model_description,
                            sage_job=sage_job,
                            sage_job_description=sage_job_description,
                            sage_hyperparameter_job=sage_hyperparameter_job,
                            sage_hyperparameter_description=sage_hyperparameter_description,
                            mlengine_model=mlengine_model, 
                            mlengine_job_single=mlengine_job_single,
                            mlengine_job_distributed=mlengine_job_distributed,
                            mlengine_hyperparam=mlengine_hyperparam,
                            mlengine_predict=mlengine_predict
                        )









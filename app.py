#export GOOGLE_APPLICATION_CREDENTIALS=baeri-test-project-77e8c8183336.json

from flask import Flask, render_template, request
import requests
import urllib, json
import httplib2

import datetime
import dateutil.parser
from dateutil import relativedelta

from apiclient.discovery import build
from oauth2client.client import GoogleCredentials
from googleapiclient import errors
from googleapiclient import discovery

import boto3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple

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
GOOGLE_ML_SINGLE_TRAINING = 'census_single_2'
GOOGLE_ML_DISTRIBUTED_TRAINING = 'census_dist_1'
GOOGLE_ML_HYPERPARAM = 'census_core_hptune_1'
GOOGLE_ML_PREDICT = 'census_prediction_1'

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

        return response
    except errors.HttpError, err:
        # Something went wrong, print out some information.
        print('There was an error creating the model. Check the details:')
        print(err._get_reason())
    
    return "poops" 

def compute_mlengine_training_cost(job):
    mlUnits = job["trainingOutput"]["consumedMLUnits"]
    return mlUnits*0.49

def compute_mlengine_prediction_cost(job):
    return int(job["predictionOutput"]["predictionCount"])*job["predictionOutput"]["nodeHours"]

def get_mlengine_duration(job):
    createTime = job["createTime"]
    endTime = job["endTime"]

    create_time = datetime.datetime.strptime(createTime, "%Y-%m-%dT%H:%M:%SZ")
    #start_time = datetime.datetime.strptime(job["startTime"], "%Y-%m-%dT%H:%M:%SZ")
    end_time = datetime.datetime.strptime(endTime, "%Y-%m-%dT%H:%M:%SZ")

    diff = relativedelta.relativedelta(end_time, create_time)
    return "{} hours, {} minutes, {} seconds".format(diff.hours, diff.minutes, diff.seconds)
    

def get_mlengine_plot():
    n_groups = 5

    means_men = (20, 35, 30, 35, 27)
    std_men = (2, 3, 4, 1, 2)

    means_women = (25, 32, 34, 20, 25)
    std_women = (3, 5, 2, 3, 3)

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.35

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = ax.bar(index, means_men, bar_width,
                    alpha=opacity, color='b',
                    yerr=std_men, error_kw=error_config,
                    label='Men')

    rects2 = ax.bar(index + bar_width, means_women, bar_width,
                    alpha=opacity, color='r',
                    yerr=std_women, error_kw=error_config,
                    label='Women')

    ax.set_xlabel('Group')
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(('A', 'B', 'C', 'D', 'E'))
    ax.legend()

    fig.tight_layout()
    plt.show()

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
    

    ''' GOOGLE ML ENGINE
    '''
    mlengine_model = get_mlengine_model(dont_call_api)
    mlengine_job_single = get_mlengine_job(not dont_call_api, GOOGLE_ML_SINGLE_TRAINING)
    mlengine_job_distributed = get_mlengine_job(not dont_call_api, GOOGLE_ML_DISTRIBUTED_TRAINING)
    mlengine_hyperparam = get_mlengine_job(not dont_call_api, GOOGLE_ML_HYPERPARAM)
    mlengine_predict = get_mlengine_job(not dont_call_api, GOOGLE_ML_PREDICT)

    #TIME
    mlengine_job_single_duration = get_mlengine_duration(mlengine_job_single)
    mlengine_job_distributed_duration = get_mlengine_duration(mlengine_job_distributed)
    mlengine_hyperparam_duration = get_mlengine_duration(mlengine_hyperparam)
    mlengine_predict_duration = get_mlengine_duration(mlengine_predict)
    
    #MONEY
    mlengine_job_single_cost = compute_mlengine_training_cost(mlengine_job_single)
    mlengine_job_distributed_cost = compute_mlengine_training_cost(mlengine_job_distributed)
    mlengine_hyperparam_cost = compute_mlengine_training_cost(mlengine_hyperparam)
    mlengine_predict_cost = compute_mlengine_prediction_cost(mlengine_predict)

    #CPU USAGE

    #GPU USAGE

    #MEMORY USAGE

    get_mlengine_plot()

    '''
    print("SINGLE!!!!!")
    print(json.dumps(mlengine_job_single, indent=1, sort_keys=True, default=str))
    print("distribetud!!!!!")
    print(json.dumps(mlengine_job_distributed, indent=1, sort_keys=True, default=str))
    print("HYPERPARAM!!!")
    print(json.dumps(mlengine_hyperparam, indent=1, sort_keys=True, default=str))
    print("prediction!!!")
    print(json.dumps(mlengine_predict, indent=1, sort_keys=True, default=str))
    '''
    return render_template('index.html', sage_model=sage_model,
                            sage_model_description=sage_model_description,
                            sage_job=sage_job,
                            sage_job_description=sage_job_description,
                            sage_hyperparameter_job=sage_hyperparameter_job,
                            sage_hyperparameter_description=sage_hyperparameter_description,

                            mlengine_model=mlengine_model, 
                            mlengine_job_single_duration=mlengine_job_single_duration,
                            mlengine_job_distributed_duration=mlengine_job_distributed_duration,
                            mlengine_hyperparam_duration=mlengine_hyperparam_duration,
                            mlengine_predict_duration=mlengine_predict_duration,

                            mlengine_job_single_cost=mlengine_job_single_cost,
                            mlengine_job_distributed_cost=mlengine_job_distributed_cost,
                            mlengine_hyperparam_cost=mlengine_hyperparam_cost,
                            mlengine_predict_cost=mlengine_predict_cost
                        )









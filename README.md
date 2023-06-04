## Layout of the SageMaker ModelBuild Project Template

The template provides a starting point for bringing your SageMaker Pipeline development to production.

```
|-- codebuild-buildspec.yml
|-- CONTRIBUTING.md
|-- pipelines
|   |-- abalone
|   |   |-- evaluate.py
|   |   |-- __init__.py
|   |   |-- pipeline.py
|   |   `-- preprocess.py
|   |-- get_pipeline_definition.py
|   |-- __init__.py
|   |-- run_pipeline.py
|   |-- _utils.py
|   `-- __version__.py
|-- README.md
|-- sagemaker-pipelines-project.ipynb
|-- setup.cfg
|-- setup.py
|-- tests
|   `-- test_pipelines.py
`-- tox.ini
```

## Start here
This is a sample code repository that demonstrates how you can organize your code for an ML business solution. This code repository is created as part of creating a Project in SageMaker. 

In this example, we are solving the abalone age prediction problem using the abalone dataset (see below for more on the dataset). The following section provides an overview of how the code is organized and what you need to modify. In particular, `pipelines/pipelines.py` contains the core of the business logic for this problem. It has the code to express the ML steps involved in generating an ML model. You will also find the code for that supports preprocessing and evaluation steps in `preprocess.py` and `evaluate.py` files respectively.

Once you understand the code structure described below, you can inspect the code and you can start customizing it for your own business case. This is only sample code, and you own this repository for your business use case. Please go ahead, modify the files, commit them and see the changes kick off the SageMaker pipelines in the CICD system.

You can also use the `sagemaker-pipelines-project.ipynb` notebook to experiment from SageMaker Studio before you are ready to checkin your code.

A description of some of the artifacts is provided below:
<br/><br/>
Your codebuild execution instructions. This file contains the instructions needed to kick off an execution of the SageMaker Pipeline in the CICD system (via CodePipeline). You will see that this file has the fields definined for naming the Pipeline, ModelPackageGroup etc. You can customize them as required.

```
|-- codebuild-buildspec.yml
```

<br/><br/>
Your pipeline artifacts, which includes a pipeline module defining the required `get_pipeline` method that returns an instance of a SageMaker pipeline, a preprocessing script that is used in feature engineering, and a model evaluation script to measure the Mean Squared Error of the model that's trained by the pipeline. This is the core business logic, and if you want to create your own folder, you can do so, and implement the `get_pipeline` interface as illustrated here.

```
|-- pipelines
|   |-- abalone
|   |   |-- evaluate.py
|   |   |-- __init__.py
|   |   |-- pipeline.py
|   |   `-- preprocess.py

```
<br/><br/>
Utility modules for getting pipeline definition jsons and running pipelines (you do not typically need to modify these):

```
|-- pipelines
|   |-- get_pipeline_definition.py
|   |-- __init__.py
|   |-- run_pipeline.py
|   |-- _utils.py
|   `-- __version__.py
```
<br/><br/>
Python package artifacts:
```
|-- setup.cfg
|-- setup.py
```
<br/><br/>
A stubbed testing module for testing your pipeline as you develop:
```
|-- tests
|   `-- test_pipelines.py
```
<br/><br/>
The `tox` testing framework configuration:
```
`-- tox.ini
```

## AWS Tools Used
* Code Pipeline — automate continuous delivery pipelines 

## To Do
* Create simple preprocessing and xgboost pipeline on AWS
* https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-projects-walkthrough-3rdgit.html
* https://towardsdatascience.com/5-simple-steps-to-mlops-with-github-actions-mlflow-and-sagemaker-pipelines-19abf951a70
* Add Pre-Build to codebuild buildspec?
* Implement Continuous Training (eg., by time; when testing metrics fall below threshold, model drift; data drift)

## Notes
In AWS CodeBuild, the pre_build phase is used to execute commands before the build starts. The build phase is used to execute commands that build the source code of your application.
You can write unit tests in either the pre_build or build phase depending on your use case. If you want to run unit tests before building the source code, you can write them in the pre_build phase. If you want to run unit tests after building the source code, you can write them in the build phase.
https://docs.aws.amazon.com/codebuild/latest/userguide/test-report-pytest.html
```
version: 0.2

phases:
  install:
    runtime-versions:
      python: 3.8
    commands:
      - pip install pytest
  pre_build:
    commands:
      
  build:
    commands:
      - python -m pytest --junitxml=<test report directory>/<report filename>
      - python pipeline.py

reports:
  pytest_reports:
    files:
      - <report filename>
    base-directory: <test report directory>
    file-format: JUNITXML
```

## Log
Updates on progress

* 1 Jun 2023 — modified preprocess, xgboost_train, evaluate, and pipeline for SageMaker (haven't test if it works) 
* 27 May 2023 — created preprocessing.py and xgboost_train.py with arguments (working on local)




## AWS Steps
1. [Set up AWS Environment](https://aws.amazon.com/getting-started/guides/setup-environment/)
2. Set up public VPC
3. Set up Amazon SageMaker Studio Domain
4. Set up a SageMaker Studio notebook and parameterize the pipeline


~~1. Go to IAM, under Roles, edit AmazonSageMakerServiceCatalogProductsLaunchRole by adding "sagemaker:DescribeCodeRepository", "sagemaker:AddTags", "sagemaker:CreateCodeRepository" to it.~~

<br>

1. Follow [SageMaker MLOps Project Walkthrough Using Third-party Git Repos](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-projects-walkthrough-3rdgit.html) BUT before creating the project, do the following steps:

    1. Follow [this](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-projects-templates-sm.html#sagemaker-projects-templates-update) by editing AmazonSageMakerServiceCatalogProductsUseRole:

    ```
        {
        "Effect": "Allow",
        "Action": [
            "codestar-connections:UseConnection"
        ],
        "Resource": "arn:aws:codestar-connections:*:*:connection/*",
        "Condition": {
            "StringEqualsIgnoreCase": {
                "aws:ResourceTag/sagemaker": "true"
            }
        }
    },
    {
        "Effect": "Allow",
        "Action": [
            "s3:PutObjectAcl"
        ],
        "Resource": [
            "arn:aws:s3:::sagemaker-*"
        ]
    }
    ```

    2. Go to IAM, under Toles, add permissions to AmazonSageMakerServiceCatalogProductsLaunchRole: AmazonSageMakerServiceCatalogProductsLaunchRole and AmazonSageMakerServiceCatalogProductsLaunchRole

<br>

2. After uploading data to S3, in IAM role find AmazonSageMakerServiceCatalogProductsUseRole, add below policy in this role:
https://github.com/aws/amazon-sagemaker-examples/issues/1923

```
{
            "Effect": "Allow",
            "Action": [
                "s3:AbortMultipartUpload",
                "s3:DeleteObject",
                "s3:GetObject",
                "s3:GetObjectVersion",
                "s3:PutObject"
            ],
            "Resource": [
                "arn:aws:s3:::<BUCKET_NAME>"
            ]
        }
```

## Useful Resources
* https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html#step-type-training
* https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker-pipelines/tabular/custom_callback_pipelines_step/sagemaker-pipelines-callback-step.ipynb
# DSCI6015-Midterm-MalConv-Malware-Classification

## Project Overview
Welcome to the DSCI6015 SP24 Midterm Project - Cloud-based PE Malware Detection API. This project is part of the Artificial Intelligence and Cybersecurity course (DSCI 6015) for the Spring 2024 semester at the University of New Haven, instructed by Professor Vahid Behzadan.

## Project Description
The main objective of this project is to implement and deploy a machine learning model for malware classification using the MalConv architecture. The model is trained to classify Portable Executable (PE) files as either malware or benign. The project consists of three main tasks:

### Task 1 – Building and Training the Model
In this task, a deep neural network based on the MalConv architecture is implemented in Python 3.10 using PyTorch (2.x). The EMBER-2018 v2 dataset is utilized for training. The provided sample implementation in the EMBER repository is a useful resource, but modifications are required to meet the project's specific requirements. The model will be documented in a Jupyter Notebook, and training can be accelerated using cloud platforms such as Google Colab or AWS Sagemaker.

### Task 2 - Deploy the model as a cloud API
After successfully training the model, the next step is deploying it as a cloud API using Amazon Sagemaker. This allows other applications to make real-time use of the model for malware classification. 

### Task 3 – Create a webclient
For the final task, a user-friendly web application is created using Streamlit. Users will be able to upload PE files, and the application will convert them into a feature vector compatible with the MalConv/EMBER model. The feature vector will then be sent to the deployed API, and the results (Malware or Benign) will be displayed to the user.

## Getting Started
To begin, clone this repository to your local machine. Follow the instructions for detailed implementation steps.
The presentation for this project can be located here[https://youtu.be/IyihNc5NlZM]


## Additional Resources
Additional resources is mentioned in the report for your reference!

Feel free to explore and modify the code to enhance the project further.

# Predicting the Number of Hits of ImmobilienScout24 Listings

### General Information 

This research project that should allow us to gain first hand experience with the methods, challenges and possibilities of the machine learning pipeline. In order to achieve this, we will implement a prediction task with a number of different model architectures, hyper-parameters and methods. Correspondingly, this project falls under categories 1 and 5 of the project types described in the document titled *Guidance Notes for Final Projects*: applying an existing ML model to a new task or problem and experimental and/or theoretical analysis of machine learning models. From a theoretical perspective, our project falls in the supervised learning paradigm.

The project's main use case will be predicting the number of clicks of online listings in the German housing market. Specifically, we will use listing features from a large dataset (Campus data RWI-GEO-RED) to predict how often these housing offers are clicked on. The dataset in question includes a good number of interesting listing features which will allow us to explore ancillary questions that should allow us to explore how changing specific parameters in our models produce different performance results. Just to mention an example, we will try using balanced and unbalanced training sets to explore how these different approaches affect different performance metrics.

Since all the project’s contributors are very familiar with the linear regression methodology, we decided to explore the possibilities that machine learning has to offer to solve regression tasks. For this same reason, we decided to use a linear model as our baseline against which all other models would be compared. With this in mind, we set out to look for a dataset that could allow us to predict the values of a continuous variable, the quintessential regression task.

### Data

We will be using a dataset called _Real estate data for Germany (RWI-GEO-RED)_, described in Breidenbach and Schaffner's paper (https://www.degruyter.com/document/doi/10.1515/ger-2019-0126). The dataset contains more than
400,000 samples of 57 variables describing residential commercial real estate listings from ImmobilienScout24 (www.immobilienscout24.de), considered the largest online platform for real estate offers in Germany. These variables describe not only the property being listed (e.g. surface area, number of rooms, number of bathrooms, etc.) but also contains information about the ad itself (e.g. days of availability, number of hits, etc). In order to obtain access to the dataset, we needed to submit a formal request and have been recently granted access

With respect to the project's need for specialized hardware -such as access to a GPU farm or the like- or computational tools -such as cloud computing beyond Google Workspace-, we are at this point in time unaware of any particular requirements. However, we believe it is a good idea to become acquainted with services such as DataBricks or Amazon Web Services, so processing our project with the free versions of one of these services could become part of the learning objectives if time permits.

### Models

We decided to go for, in addition to a baseline linear regression model, four additional regressors: Poisson, regression tree, random forest and multi-layer perceptron. These were chosen because of their ability to solve regression tasks and their availability in the scikitlearn module, which we considered a good place to start before going into more sophisticated options like TensorFlow. Preliminary results from these models are displayed below.

<img width="273" alt="image" src="https://user-images.githubusercontent.com/72525078/163332234-316ce6cd-3d8e-4747-821c-d0640566aa68.png">


### Contributors

- [Carlo Greß](https://github.com/carlo-gress)
- [Wojciech Kuznicki](https://github.com/wkuznicki)
- [Santiago Sordo](https://github.com/odros)


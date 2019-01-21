# 3D-Digit-Recognition
The practical assignment requires us to recognize and classify the given hand-written digits based on the 3-D location time series. For this purpose, we have used Python as our implementation programming language with NumPy, Pandas and SciPy library functions.

## Information on the given data
We were provided with training dataset where the location information of each stroke of the experiment had been stored into separate comma-separated values (CSV) file. The strokes provided were for numbers 0 to 9 with 100 different observations for each number
(1000 observations in total). The dataset was such that each stroke had different counts based on the time taken by the subject.

## Model Selection
While selecting the model for our manual implementation, we have used high level Python libraries to research different approaches. The model in bold has been chosen.

| Solution  | Results summary |
| ------------- | ------------- |
| Mnist-style feature extraction (28x28 matrix of binary values) + Deep Neural Network in Tensorflow + Keras  | High accuracy (~90%), complex implementation  |
| Custom feature extraction (distance & angle between the first and the last point, variance on x and y axes) + Neural Network | Low accuracy (~60%), complex implementation  |
| Custom feature extraction (distance & angle between the first and the last point, variance on x and y axes) + KNN Classifier in Scikitlearn | Low accuracy (~60%), complex implementation  |
| Resampling and unit vectors extraction for movements + KNN Classifier | High accuracy (~94%), simple implementation  |

## Data Pre-processing and Feature Extraction (input_funcs.py)
We start with automation of filename generation for the strokes for proper dataset parsing.

```
Give examples
```

We then process and parse the data. After importing the csv file into a temporary pandas DataFrame, we name the 3 columns into x, y, and z. These columns represent the coordinates of the digits. We first drop the z coordinates from our dataset.

```
Give examples
```

We do this because the z coordinates do not provide any value for the classification of our dataset. We thus decided to work with the x and y coordinates only. We make sure there are no duplicate values in the dataset.

```
Give examples
```

The dataset is then divided into n = 10 datapoints. We resample the data as the number of counts in each stroke for every digit is different. There are some strokes that have 17 counts while some have more than 220 counts in their observations So, we use the SciPy library function to resample the datapoints. We then use Min-max normalization feature scaling on the new data-frame so that the values of the features of the dataset are scaled to a fixed range. In this program we do not use PCA for dimensionality reduction as knowing the max variance between the 2 coordinates does not provide us with a good reading.

```
Give examples
```

Now we try to get the values for the unit vectors for each of the 10 movements. We first get the differences between the points for both x and y coordinates. Then we find the length of the vector points, where A and B are the different points in the database. We divide this factor with the difference values for both x any y to get the unit vector values. 

```
Give examples
```

We then flatten the data we have to a list. Flattening is when we take the rows of data and combine then together in order to get a single 1-D vertical vector. As we are processing all the datasets in cycles, we append the data so that at the end we have 1000 rows of data points, in the main file, after which we convert final list into a 2-D array (named x). We create a vector of labels for the y
values.

Thus, the features that were selected for this program were unit vectors (directions) of movements after resampling. They are represented in x and y coordinates, which are then flattened (the order of the coordinates does not matter since we are going to measure distances between samples). 

## Input call Script (prepare_train.py)


### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

# multiclass-text-classification-RNN
text classification using RNN/LSTMr as part of Kumparan Data Science skill test

You are given a dataset of articles in file data.csv and model interface file model.py.
The data.csv file is contains 3 columns:

● ```article_id: unique id of the article```

● ```article_topic: main topic of the article```

● ```article_content: content of the article```

You will implement the task explained below inside file model.py.

Task Overview

From the dataset, you should able to create a topic extractor model that basically extract a
topic given the content of the article. We expect the topic extractor to behave as the dataset
given. So given the article_content it will output the article_topic.

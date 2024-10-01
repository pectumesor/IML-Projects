Project 4: Sentiment Analysis

Our task in this project was to train a neural network that could best predict how someone was feeling based on product review, they've written. The task is to do a regession
 where 0 is very angry and 10 is very happy.
To solve this task, I proceeded as follows: <br>

1. I first searched the Hugging Face database for a neural network 
that was already trained for a similar task on a large amount of data.
2. I went through Hugging Face Pipeline of how to perform Nature Language Processing: Text
Tokenization, Data Loaders, etc.
3. I trained the model I found on our data and used the classification label = 1, since this is
equivalent to regression.
4. On each epoch I kept track of our most accurate model parameters, and once the training was done, we keep the best one.
5. We then do our predictions and save them to our csv file, keeping overflowing values (< 0 or > 10) in our desired
interval of [0,10]


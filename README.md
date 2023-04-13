# Stock price prediction using the Transformer architecture

Yes, I am aware that this is probably a futile task, but I had to give it a go. So far the results are unsatisfactory, but I'm having fun experimenting with it and trying different things.

## The model

The idea is to exploit the Transformer to find relationships between different tickers which would potentially help predict how a specific ticker would behave in the future. I do this by constructing a feature vector composed of *n* different values representing *n* different tickers for a specific day. These feature vectors are then put in a sequence of size *D* and fed into the transformer, which then outputs a different sequence of size *L*, and of the same feature vector dimension *n*. These outputs are then passed through a fully connected layer to reduce the dimension of features from *n* to *1*. In short, this is what's happening (batch dimension omitted):

	[D, n] -> Transformer -> [L, n] -> FC -> [L, 1]

## The data

The dataset used in this project was downloaded from [here](https://www.kaggle.com/datasets/paultimothymooney/stock-market-data). So far, I only used the NASDAQ dataset. In the original dataset, each ticker consists of 5 features: *Open*, *Volume*, *High*, *Low* and *Closing* (and Adjusted Closing). At first, I tried to use an AutoEncoder to compress these features into a scalar, which I knew probably wouldn't work, but I still tried it, and failed, after which I only decided to use the *High* feature moving forward.
I constructed a new dataset table consisting of rows representing each day and columns representing the High price for each ticker. The resulting table had 13356 rows and 1564 columns. The problem was most of the table was sparse, aka had no values in cells. I had to reduce the table to only a subset of rows and columns where I would still have a relatively high number of columns/tickers and where each ticker would have an unbroken chain of non-nan values. I wrote a function to find such subsets and reduced the table to 499 tickers across 6678 days. Later on, I removed a random ticker from the dataset because the Transformer requires the input feature vector to have a dimension divisible by its number of heads. By default the number of heads is 8, but I reduced it to 6, because 498 is divisible by 6.

I wrote my own function for splitting the dataset into train, validation and test subsets because I couldn't find an equivalent in pytorch. I couldn't just take random indices for each subset because that would almost certainly create overlaps and then the model couldn't be properly tested. My function makes sure there aren't any overlaps in sequences, in other words, makes sure the model is tested on completely unseen sequences of data.

Furthermore, when sampling the dataset, the input sequence is left as is, but the output is transformed into returns, meaning the Transformer takes in actual, unscaled features in a sequence and is expected to output a sequence of *returns* for each day in an output sequence.

I chose the input sequence to be of length *D* = 90 days, and the output sequence to be of length *L* = 7 days.

	[90, 498] -> Transformer -> [7, 498] -> FC -> [7, 1]

I used the mean squared error for the loss function, but to finally conclude how accurate my model turned out to be, I constructed my own "accuracy" metric, by simply checking wether the model correctly predicted if returns would be positive or negative. So far, my pseudo-accuracy hovers around 0.5, making this whole thing a failure, but I am learning a lot and not giving up yet because there's a lot more things to try out.


## Things to try out
	
	Represent all data as returns (and properly scale it)
	Find a way to use all available ticker features
	include some sort of time encoding
	Look into moving average smoothing and try to apply it
	Actually do research what other people have done
	...

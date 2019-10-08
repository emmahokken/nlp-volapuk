# nlp-volapuk

But it's actually Volapük. 


## How to run 

The code for training the network is located in `main.py`. The program takes on a variety of input arguments, but none of them are required and the default settings were used in the report. The following arguments are can be added:

`--input_dim`: Dimensionality of input sequence. Default: 1
<br/>
`--num_hidden`: Number of hidden units in the model. Default: 64
<br/>
`--num_layers`: Number of layers in the model. Default: 2
<br/>
`--batch_size`: Number of examples to process in a batch. Default: 128
<br/>
`--training_steps`: Number of training steps. Default: 20000
<br/>
`--learning_rate`: Learning rate. Default: 0.01 
<br/>
`--momentum`: Momentum. Default: 0.95
<br/>
`--device`: Training device 'cpu' or 'cuda:. Default: cuda:0 
<br/>
`--load_PATH`: Load model from certain path. Default: None. Please either include 'models/model\_lambda0.1' or 'models/model\_LSTM'
<br/>
`--seed`: Set seed to guarantee validity of results. Default: 42
<br/>
`--evaluate_steps`: Evaluate model every so many steps. Default: 25
<br/>
`--eval`: Boolean stating whether to train or evaluate the model. Default: False 
<br/>
`--embedding_size`: Size of the character embeddings used. Default: 256
<br/>
`--importance_sampler`: Boolean representing whether or not to use the importance sampler. Default: False


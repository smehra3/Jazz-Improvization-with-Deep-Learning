# Jazz-Improvization-with-Deep-Learning

Implemented a model that uses deep learning ([LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory)) to generate music.

Trained the network to generate novel jazz solos in a style representative of a body of performed work.

### Overview of the model

Here is the architecture of the model, implemented with **Keras**.

<img src="images/music_generation.png" style="width:600;height:400px;">


* $X = (x^{\langle 1 \rangle}, x^{\langle 2 \rangle}, \cdots, x^{\langle T_x \rangle})$ is a window of size $T_x$ scanned over the musical corpus. 
* Each $x^{\langle t \rangle}$ is an index corresponding to a value.
* $\hat{y}^{t}$ is the prediction for the next value.
* The model will be trained on random snippets of 30 values taken from a much longer piece of music. 
    - Each of the snippets to have the same length $T_x = 30$ to make vectorization easier.

* Model will predict the next note in a style that is similar to the jazz music that it's trained on.  The training is contained in the weights and biases of the model. 
* Then those weights and biases are used in a new model which predicts a series of notes, using the previous note to predict the next note. 
* The weights and biases are transferred to the new model using 'global shared layers' described below"


* The model takes input X of shape $(m, T_x, 78)$ and labels Y of shape $(T_y, m, 78)$. 
* LSTM is used with hidden states that have $n_{a} = 64$ dimensions.

## Generating music

After training the model and having it learn the patterns of the jazz soloist, it will be used to synthesize new music. 

**Predicting & Sampling**

<img src="images/music_gen.png" style="width:600;height:400px;">

At each step of sampling:
* Took as input the activation '`a`' and cell state '`c`' from the previous state of the LSTM.
* Forward propagated by one step.
* Got a new output activation as well as cell state. 
* The new activation '`a`' can was then used to generate the output using the fully connected layer, `densor`. 

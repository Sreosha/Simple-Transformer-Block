# Simple-Transformer-Block
A Simple Transformer Block Built using Pytorch. There're few exceptions in the Transformer Block from the paper "Attention is All you need" because I tried to simplify it in my own way.
# Modules of the Transformer Block and the flow as follows - 
<br>*** Used different variable names as compared to the code for the sake of simplification </br>
# Self-Attention
<br>Here I have tried to cover the Multi-Head Attention (MHA) which follows the concept of Seperation of Concerns, i.e Query , Key & Value tensors we are getting after applying Linear Transformation of the top of input tensor.</br>
<br>In the Forward part the steps were followed as below-</br>
<br>Step 1: Query * Key</br>
<br>Step 2: qkd = Query * Key//√dimension of Key</br>
<br>Step 3: sqkd = Softmax(qkd)</br>
<br>Step 4: vec = Value * sqkd</br>
<br>Step 5: ReLU(vec)</br>
# TransformerBlock 
<br>Step 1: Attention(Query, Key, Value)</br>
<br>Step 2: n1 = Dropout(Layer Normalization(Attention(Query, Key, Value) + Query))</br>
<br>Step 3: fc = Fully Connected with RELU Layer(Dropout(Layer Normalization(Attention(Query, Key, Value))))</br>
<br>Step 4: n2 = Layer Normalization(fc + n1)</br>
# Encoder Block
<br>*** Not using Sinusoidal Positional Encoding, using a normal indexing</br>
<br>Step 1: w = Embedding(input sequence)</br>
<br>Step 2: pos = Embedding(positional index of the sequence)</br>
<br>Step 3: Dropout(w + pos)</br>
# DecoderBlock (TransformerBlock for Decoder)
<br>*** During training the input is the output of the Encoder input. Key & Value are coming from Encoder side and Query is from Decoder side due to "Cross-Attention".</br>
<br>Step 1: x = Attention(Input, Input, Input)</br>
<br>Step 2: intermediate = x + Input</br>
<br>Step 3: query_decoder = Dropout(Layer Normalization(x + intermediate))</br>
<br>Step 4: fc = TransformerBlock(query_decoder, value, key)</br>

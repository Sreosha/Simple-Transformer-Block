# Simple-Transformer-Block
A Simple Transformer Block Built using Pytorch. There're few exceptions in the Transformer Block from the paper "Attention is All you need" because I tried to simplify it in my own way.
# Modules of the Transformer Block and the flow as follows - 
*** Used different variable names as compared to the code for the sake of simplification
# Self-Attention
Here I have tried to cover the Multi-Head Attention (MHA) which follows the concept of Seperation of Concerns, i.e Query , Key & Value tensors we are getting after aapplying Linear Transformation of the top of input tensor.
In the Forward part the steps were followed as below-
Step 1: Query * Key
Step 2: qkd = Query * Key//√dimension of Key
Step 3: sqkd = Softmax(qkd)
Step 4: vec = Value * sqkd
Step 5: ReLU(vec)
# TransformerBlock 
Step 1: Attention(Query, Key, Value)
Step 2: n1 = Dropout(Layer Normalization(Attention(Query, Key, Value) + Query))
Step 3: fc = Fully Connected with RELU Layer(Dropout(Layer Normalization(Attention(Query, Key, Value))))
Step 4: n2 = Layer Normalization(fc + n1)
# Encoder Block
*** Not using Sinusoidal Positional Encoding, using a normal indexing
Step 1: w = Embedding(input sequence)
Step 2: pos = Embedding(positional index of the sequence)
Step 3: Dropout(w + pos)
# DecoderBlock (TransformerBlock for Decoder)
*** During training the input is the output of the Encoder input. Key & Value are coming from Encoder side and Query is from Decoder side due to "Cross-Attention".
Step 1: x = Attention(Input, Input, Input)
Step 2: intermediate = x + Input
Step 3: query_decoder = Dropout(Layer Normalization(x + intermediate))
Step 4: fc = TransformerBlock(query_decoder, value, key)

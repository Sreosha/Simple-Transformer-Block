import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"
        """Applying Linear Transformation 
        input feature dim == output feature dim
        because key, value and query should be of same size"""
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        """When we have MHA or Multi-head Attention, we have the 
        no. of heads == no. of queries == no. of values == no. of keys 
        for a particular item"""

        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # dot product of query and key
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries,keys])
        # queries shape: (N, query_len, heads, head_dim )
        # keys shape: (N, key_len, heads, head_dim )
        # value shape: (N, value_len, heads, head_dim )
        # energy shape: (N, heads, query_len, key_len)


        if mask is None:
            energy = energy.masked_fill(mask = 0, value=float("-1e20"))
        # Normalization by √dk so that we can accomodate with the value tensor without variance
        attention = torch.softmax(input = energy/(self.embed_size **(1/2)), dim = 3)
        # out = softmax((query*key/√dk)*values)
        out = torch.einsum("nhql, nlhd->nqhd", [attention, values]).reshape(N, query_len,
                                                                            self.heads*self.head_dim)
        
        #attention shape: (N, heads, query_len, key_len)
        #value shape: (N, value_len, heads, head_dim )
        #after einsum: (N, query_len, heads, head_dim ) # then flatten last two dimension
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        # Applying layer normalization after self attention
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):

        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        # Attention +  ReLU layer
        forward = self.feed_forward(x)
        # Layer Normalization on the top of ( Attention +  ReLU layer)
        out = self.dropout(self.norm2(forward + x))
        return out
    
class Encoder(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 embed_size,
                 num_layers,
                 heads,
                 device,
                 forward_expansion,
                 dropout,
                 max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        # Generating Word Embeddings using Embedding Layer
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        # For the Positional Embeddings also using Embedding Layer
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.Module(
            [
                TransformerBlock(embed_size,
                                 heads,
                                 dropout,
                                 forward_expansion)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        # Arranging the position based on index number not Sinusoidal Positional Encoding
        positions = torch.arrange(0,seq_length).expand(N,seq_length).to(self.device)
        # Word Embedding of the input using embedding layer + Embedding of index position based on the length of input
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out
    
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm.transformer_block = TransformerBlock(embed_size,
                                                       heads,
                                                       dropout,
                                                       forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        # During Training, generating Masked Self attention of inputs (output of the input in Encoder Block)
        attention = self.attention(x,x,x, trg_mask)
        # The attention from above step is further considered as query
        query = self.dropout(self.norm(attention + x))
        # Taking key and value from the Encoder Block for Cross Attention Purpose
        out = self.norm.transformer_block(value, key, query, src_mask)
        return out
    
class Decoder(nn.Module):
    def __init__(self, trg_vocab_size,
                 embed_size,
                 num_layers,
                 heads,
                 forward_expansion,
                 dropout,
                 device,
                 max_length):
        super(Decoder, self).__init__()
        self.device = device
        # Same as encoder block
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList([
            DecoderBlock(embed_size, heads, forward_expansion, dropout, device) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N,seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            #Cross Attention
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        out = self.fc_out(x)

class Transformer(nn.Module):
    def __init__(self, 
                 src_vocab_size,
                 trg_vocab_size,
                 src_pad_idx,
                 trg_pad_idx,
                 embed_size = 256,
                 num_layers = 6,
                 forward_expansion = 4,
                 heads = 8,
                 dropout = 0,
                 device = "cuda",
                 max_length = 100):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size,
                               embed_size,
                               num_layers,
                               heads,
                               device,
                               forward_expansion,
                               dropout,
                               max_length)
        self.decoder = Decoder(trg_vocab_size,
                               embed_size,
                               num_layers,
                               heads,
                               forward_expansion,
                               dropout,
                               device,
                               max_length)

    def make_src_mask(self, src):
        # Increasing dimension of padding by 1 dim, then 2 dim further
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        #(N, 1, 1, src_len)
        return src_mask.to(self.device)
    
    def make_trg_mask(self, trg):
        n, trg_len = trg.shape
        # Getting sparse matrics of 1 and expanding the dimention further
        trg_mask = torch.tril((torch.ones((trg_len, trg_len)))).expand(N, 1,trg_len, trg_len)
        return trg_mask.to(self.device)
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out

# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     x = torch.tensor([[1,5,6,4,3,9,5,2,0], [1,8,7,3,4,5,6,7,2]]).to(device)

#     trg = torch.tensor([[1,7,4,3,5,9,2,0], [1,5,6,2,4,7,6,2]]).to(device)

#     src_pad_idx = 0
#     trg_pad_idx = 0
#     src_vocab_size = 10
#     trg_vocab_size = 10
#     model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)
#     out = model(x, trg[::-1])
#     print(out.shape)






import torch
import torch.nn as nn

class CodeSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_encoder_layers=3, num_decoder_layers=3):
        super(CodeSeq2Seq, self).__init__()
        # Embedding layer to turn token IDs into vectors
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Transformer: Handles the Encoder (reading code) and Decoder (generating summary)
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers, 
            batch_first=True
        )
        
        # Final layer to turn vectors back into token probabilities
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        # Create a mask so the decoder cannot see future words (causal masking)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
        
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)
        
        # Pass through Transformer
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        return self.fc_out(out)
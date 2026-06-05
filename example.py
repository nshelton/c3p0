import interp

m = interp.load("gpt2-small")
cap = interp.run(m, "The cat sat on the")

print("prompt:", cap.prompt)
print("tokens:", cap.tokens)
print("attn:  ", cap.attn.shape, "[n_layers, n_heads, seq, seq]")
print("logits:", cap.logits.shape, "[seq, vocab]")

# a single head's [seq, seq] attention matrix, ready to imshow
print("attn[0, 0]:\n", cap.attn[0, 0])

# any other activation as numpy
print("resid_post layer 6:", interp.act(cap, "resid_post", 6).shape)
print("mlp_out layer 3:   ", interp.act(cap, "mlp_out", 3).shape)

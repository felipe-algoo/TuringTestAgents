### `Overview`

- It's important to mention that this was mostly for my own curiosity and as part of a college project. I plan to keep working on it and addressing its current limitations.
- Two deterministic probabilistic agents exchange messages. 
- A statistical analyzer examines the resulting dialogue for evidence that the agents follow different generative models. 
- The system uses no language models. 
- All responses are produced from fixed templates modulated by internal state variables and seeded random choice.

https://github.com/user-attachments/assets/9f713e05-16e5-439a-a414-359f8795c60d


- It simultaneously measures multiple distributive and sequential signals and maintains an episodic memory of previous analyses, allowing the decision threshold to adapt to previously observed confidence patterns.


### `How distinction is computed`

- Before each analysis the current state (message counts, formality values, total length, window size) is hashed. 
- The hash is deterministically embedded into a fixed-dimensional vector. The nearest past episodes are retrieved with cosine similarity weighted by an exponential temporal decay. 
- Their labels and confidences are summarized into a short natural-language context string that is later stored with the new result.
- The analyzer then computes seven signals on the (optionally windowed) histories:

- Stylometric divergence of six surface features (word count, average word length, punctuation density, capitalization ratio, presence of question marks and exclamation marks).
      
      - absolute difference in lexical entropy
      - Kolmogorov Smirnov scores on the same feature distributions
      - Jensen-Shannon divergence of token distributions
      - 2 gram overlap (reported as 1 Jaccard)
      - turn taking length regularity
      - A small prior boost derived from the retrieved episodes

- These are combined with fixed weights into a composite score. 
- An adaptive threshold is calculated from the confidence distribution of previously stored indistinguishable episodes (or a simple shift of the global mean if too few such episodes exist). 
- The final label is distinct when the composite meets or exceeds the threshold; confidence is the normalized distance from the threshold.
- All embeddings and random draws are fully deterministic given the seeds and the hash of the analysis state.

## abb

Embedding : Number of feature per token, example 32
WTE : word token Embedding
WPE : Word positional Embedding


## Step 1:
Reading the training data 
Use the custom class Dataset, and load the contain of files
to the List<String> lines, Dataset class has the custom methode
to Shuffle the lines : 

## Why Shuffle at All?
Prevents Bias: Neural networks can overfit if similar documents are grouped together
Better Generalization: Random order helps the model learn general patterns
Training Stability: Improves convergence properties

## Step 2

Then its use the Tokenizer class to tokenize the lines
It does the character-level tokenization, convert character to some number
it's usage 0 to 225 Byte to support all the unicode char range
Its check which all char are there in lines and build an array and just assign 0 to n for all n chars
* to Improve the word level tokenization and o(1) search for char to int, int to char

|Character-Level vs Word-Level Tokenization
Aspect          Character-Level            Word-Level
Vocab Size  ~256 max (limited)          Tens of thousands 
Context Window  chars ≈ 1-2 words       8 tokens ≈ 6-8 words
                BLOCK_SIZE=8 
Unknown Words   Never                   Frequent (OOV problem)
                (all chars exist) 
Memory          Minimal                 Significant

Training        Speed Slower             Faster
               (longer sequences)        (shorter sequences)


rade-offs
Character-Level (What we have) ✅
✓ Simple, no preprocessing needed
✓ Can generate any character combination
✓ Works with typos and rare patterns
✗ Long sequences = slower training
✗ Model must learn word patterns from scratch
✗ With BLOCK_SIZE=8, only 1-2 words of context
Word-Level ✅
✓ Fewer tokens = faster training
✓ Richer context (8 tokens = 6-8 words)
✓ Model learns semantic word relationships
✗ Need dictionary/tokenizer for vocabulary
✗ Unknown words are a problem (OOV - Out Of Vocabulary)
✗ Can't generate new words or typos well

Most modern LLMs use Byte Pair Encoding (BPE) which combines advantages:
* Vocabulary: 10,000-50,000 tokens (good compression)
* Never has OOV problems (can break into bytes)
* Captures word meaning efficiently
to do use BPE

## Step 3
* after the tokonization we will come to know our Vocab size, based on this we need to set other parameter
 embSize = vocabSize * GptConfig.N_EMBD, N_EMBD is 32
* Each token in the vocabulary gets one embedding vector of length N_EMBD
* So the matrix size is vocabSize * N_EMBD (ROWS X COLS)
* This is just allocating a flattened 1D array large enough to hold a conceptual matrix [vocabSize x N_EMBD].
* Here N_EMBD, represent the feature per token which is 32.
* N_EMBD- Embedding Dimension or Hidden Dimension
* This is the size of the vector representation for each token and the hidden layer size throughout the entire transformer model.
* we can have an 1D flat array for word token embedding, size is num_tokens* Embed_size
* The same way we need to calculate the word position embedding
* The Position Embedding size needs to calulate like
Block size = number positions
Embedding width (feature) per position
Final size = Block size * Embedding width


"""
Genetic_algorithm_processes/S3_mutation/methods/synonym_mutation.py
"""

import random
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
    
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')


class SynonymMutation:
    def __init__(self,
        mutation_rate: float = 0.3,
        pos_tags_to_mutate: list[str] = [
            'NN', 'NNS', 'NNP', 'NNPS',  # Nouns
            'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # Verbs
            'JJ', 'JJR', 'JJS',  # Adjectives
            'RB', 'RBR', 'RBS'   # Adverbs
        ]
    ):
        """
        Initialize the synonym mutation operator.
        
        Parameters:
        - mutation_rate: Probability of mutating each eligible word (0.0 - 1.0)
        - pos_tags_to_mutate: List of POS tags to consider for mutation
                              Default: ['NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']
                              (nouns, verbs, adjectives, adverbs)
        """
        self.mutation_rate = mutation_rate
        self.pos_tags_to_mutate = pos_tags_to_mutate
    
    def get_wordnet_pos(self, treebank_tag):
        """Convert treebank POS tag to WordNet POS tag."""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None
    
    def get_synonyms(self, word, pos_tag):
        """Get synonyms for a word given its POS tag."""
        wordnet_pos = self.get_wordnet_pos(pos_tag)
        if wordnet_pos is None:
            return []
        
        synonyms = set()
        for synset in wordnet.synsets(word.lower(), pos=wordnet_pos):
            for lemma in synset.lemmas():
                synonym = lemma.name().replace('_', ' ')
                # Only add if it's different from the original word
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)
        
        return list(synonyms)
    
    def mutate_text(self, text):
        """
        Apply synonym mutation to the input text.
        
        Parameters:
        - text: Input sentence or paragraph
        
        Returns:
        - mutated_text: Text with some words replaced by synonyms
        """
        # Tokenize and POS tag
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        
        mutated_tokens = []
        
        for word, pos in pos_tags:
            # Check if this word should be considered for mutation
            if pos in self.pos_tags_to_mutate and random.random() < self.mutation_rate:
                synonyms = self.get_synonyms(word, pos)
                
                if synonyms:
                    # Choose a random synonym
                    synonym = random.choice(synonyms)
                    
                    # Preserve capitalization
                    if word[0].isupper():
                        synonym = synonym.capitalize()
                    if word.isupper():
                        synonym = synonym.upper()
                    
                    mutated_tokens.append(synonym)
                else:
                    # No synonyms found, keep original
                    mutated_tokens.append(word)
            else:
                # Don't mutate this word
                mutated_tokens.append(word)
        
        # Reconstruct the sentence
        mutated_text = self._reconstruct_text(tokens, mutated_tokens)
        
        return mutated_text
    
    def _reconstruct_text(self, original_tokens, mutated_tokens):
        """Reconstruct text with proper spacing and punctuation."""
        result = []
        for i, token in enumerate(mutated_tokens):
            if i == 0:
                result.append(token)
            elif token in '.,!?;:\'")-]}>':
                # No space before punctuation
                result.append(token)
            elif original_tokens[i-1] in '([{<"\'':
                # No space after opening brackets/quotes
                result.append(token)
            else:
                result.append(' ' + token)
        
        return ''.join(result)
    
    def mutate(self, prompt_chain):
        """
        Apply synonym mutation to a list of prompt chains.
        
        Parameters:
        - prompt_chain: List of prompts in a chain
        
        Returns:
        - mutated_prompt_chain: List of mutated prompts in the chain
        """
        mutated_prompt_chain = []
        for prompt in prompt_chain:
            model, *prompt_parts = prompt
            full_prompt = ' '.join(prompt_parts)
            mutated_prompt = self.mutate_text(full_prompt)
            mutated_prompt_chain.append((model, mutated_prompt))
        return mutated_prompt_chain


if __name__ == "__main__":
    # Test examples
    prompt_chain = [
        ("gpt-3.5-turbo", "The quick brown fox jumps over the lazy dog.", "A classic example sentence."),
        ("gpt-4", "Machine learning algorithms can process large amounts of data efficiently.", "An advanced technology topic."),
        ("gpt-3.5-turbo", "The beautiful sunset painted the sky with vibrant colors.", "A descriptive nature scene."),
        ("gpt-4", "Scientists discovered a new species of butterfly in the rainforest.", "A scientific discovery.")
    ]
    
    print("=== Synonym Mutation with mutation_rate=0.3 ===\n")
    mutator = SynonymMutation(mutation_rate=0.3)
    aggressive_mutator = SynonymMutation(mutation_rate=0.5)
    selective_mutator = SynonymMutation(
        mutation_rate=0.5,
        pos_tags_to_mutate=['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS']
    )
    
    mutated = mutator.mutate(prompt_chain)
    print(f"Mutated: {mutated}")

import nltk;
import os;
from nltk.corpus import wordnet;
from nltk.tokenize import word_tokenize;
from nltk.corpus import conll2000;
os.environ["MEGAM"]='/mnt/c/Users/jsiekmann/AI/megam-64';

partOfSpeech = {
    "CC": "Coordinating Conjunction",
    "CD": "Cardinal Number",
    "DT": "Determiner",
    "EX": "Existential There",
    "FW": "Foreign Word",
    "IN": "Preposition or Subordinating Conjunction",
    "JJ": "Adjective",
    "JJR": "Adjective (Comparative)",
    "JJS": "Adjective (Superlative)",
    "LS": "List Item Marker",
    "MD": "Modal",
    "NN": "Noun (Singular or Mass)",
    "NNS": "Noun (plural)",
    "NNP": "Proper Noun (Singular)",
    "NNPS": "Proper Noun (Plural)",
    "PDT": "Predeterminer",
    "POS": "Possessive Ending",
    "PRP": "Personal Pronoun",
    "PRP$": "Possessive Pronoun",
    "RB": "Adverb",
    "RBR": "Adverb (Comparative)",
    "RBS": "Adverb (Superlative)",
    "RP": "Particle",
    "SYM": "Symbol",
    "TO": "To",
    "UH": "Interjection",
    "VB": "Verb (Base Form)",
    "VBD": "Verb (Past Tense)",
    "VBG": "Verb (Gerund or Present Participle)",
    "VBN": "Verb (Past Participle)",
    "VBP": "Verb (Non-3rd person Singular Present)",
    "VBZ": "Verb, 3rd Person Singular Present)",
    "WDT": "Wh-determiner",
    "WP": "Wh-pronoun",
    "WP$": "Possessive wh-pronoun",
    "WRB": "Wh-adverb",
    ",": "Comma",
    ".": "Period",
    ":": "Colon",
    "?": "Question Mark",
    "!": "Exclamation Mark",
    "NP": "Noun Phrase",
    "PP": "Prepositional Phrase",
    "VB": "Verb Phrase",
    
}

class BigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        self.tagger = nltk.BigramTagger(train_data)

    def parse(self, sentence):
        pos_tags = [pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word,pos),chunktag)
                     in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)
    

def process_input(inpt):
    words = nltk.pos_tag(word_tokenize(inpt));
    chunks = [];
    
    grammar = """NP: {<DT|PRP\$>?<JJ.*>*<NN.*>}
                     {<NNP>+} 
    """;
    cp = nltk.RegexpParser(grammar);
    print(cp.parse(words));   
                
def determine_synonym(inpt):
    return wordnet.synsets(inpt);

def print_synset(inpt):
    print("Possible definitions for this input:");
    for i in wordnet.synsets(inpt):
        print("Definition: " + i.definition());
        print("Word(s) that describe this concept: ");
        for j in i.lemma_names():
            print(j);
        print(" **** ");

        
        
print("Training chunker...");
train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP']);
test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP']);  
chunker = BigramChunker(train_sents)
inpt = input("Enter an English sentence: ");

while(inpt != "quit"):
    
    grammar1 = """NP: {<DT|PRP\$>?<JJ.*>*<NN.*>}
                      {<NNP>+} 
    """;
    
    words = nltk.pos_tag(word_tokenize(inpt));
    chunks = chunker.parse(words);



    inpt = input("Enter an English sentence: ");




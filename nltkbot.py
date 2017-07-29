import nltk;
import os;
from nltk.corpus import wordnet;
from nltk.corpus import conll2000;
from nltk.corpus import sentiwordnet;
from nltk.tree import *;
from nltk.tokenize import word_tokenize;
from nltk.wsd import lesk

os.environ["MEGAM"]='/mnt/c/Users/jsiekmann/AI/megam-64';


#Input modification
REMOVE_ADJ_ADV = True;

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
    'v': "Verb",
    'n': "Noun",
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
    

            
def print_synset(inpt):
    print("Possible definitions for this input:");
    for i in wordnet.synsets(inpt):
        print("Definition: " + i.definition());
        print("Word(s) that describe this concept: ");
        for j in i.lemma_names():
            print(j);
        print(" ****  " + str(wordnet.synsets(inpt)));

        
        
def print_chunks(chunks):
    for i in chunks:
        if type(i) == nltk.tree.Tree:
            if i.label() == "NP":
                print("Noun Phrase:")
                for j in i:
                    print("       " + str(j));

            elif i.label() == "VP":
                print("Verb Phrase:");
                for j in i:
                    print("       " + str(j));

            elif i.label() == "PP":
                print("Preposition Phrase:");
                for j in i:
                    print("       " + str(j));
        else:
            print("RADICAL!" + str(i));
        

                
def disambiguate(chunks, sentence):
    for i in chunks:
        if type(i) == nltk.tree.Tree:
            for j in i:
                if(j[:2].matches("NN|JJ|VB|RB")):
                    print("sup");
            
    disambiguated_words = list();
    for i in chunks:
        print("");
        
        
        
def convert_POS_to_wn(POS):
    
    if POS[:2] == ("NN"):
        return 'v';
    elif POS[:2] == ("JJ"):
        return 's'
    elif POS[:2] == ("VB"):
        return 'v'
    elif POS[:2] == ("RB"):
        return 'r'

    
def chunk_sentiment(chunks, sentence):
    for i in chunks:
        if type(i) == nltk.tree.Tree:
            for j in i:
                if j[1][:2] == "NN" or j[1][:2] == "JJ" or j[1][:2] == "VB" or j[1][:2] == "RB":
                    print(j);
                    d = lesk(sentence, j[0], convert_POS_to_wn(j[1]))
                    print(str(d) + ", " + d.definition());
                
        
def train_ai():        
    print("Training...");
    train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP', 'VP', 'PP']);
    test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP', 'VP', 'PP']);  
    return BigramChunker(train_sents);






parser = train_ai();
inpt = input("Enter an English sentence: ");

while(inpt != "quit"):
    
    
    #If NP is followed by VB, it's probably the subject.
    #If NP is preceded by VB, it's probably the object
    #Question, answer, elaboration
    
    sentence = word_tokenize(inpt);
    #print_synset(sentence[0]);
    
    words = nltk.pos_tag(sentence);
    chunks = parser.parse(words);
    
    
    
    #disambiguate(chunks, sentence);
    chunk_sentiment(chunks, sentence);
    print_chunks(chunks);

    inpt = input();




from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import nltk
import csv
import re

with open('quora_corpus.txt') as f:
    words_quora = [word for line in f for word in line.split()]

with open('wiki_corpus.txt') as f:
    words_wiki = [re.sub(r'[^\w\s]','',word) for line in f for word in line.split()]

wiki_sentence = []
for w in words_wiki:
    if w.lower() not in stop_words:
        wiki_sentence.append(w.lower())

quora_sentence = []
for w in words_quora:
    if w.lower() not in stop_words:
        quora_sentence.append(w.lower())

finder = TrigramCollocationFinder.from_words(wiki_sentence)
trigram_wiki = finder.nbest(TrigramAssocMeasures.likelihood_ratio, 10)
finder = TrigramCollocationFinder.from_words(quora_sentence)
trigram_quora = finder.nbest(TrigramAssocMeasures.likelihood_ratio, 10)

finder = BigramCollocationFinder.from_words(wiki_sentence)
bigram_wiki = finder.nbest(BigramAssocMeasures.likelihood_ratio, 10)
finder = BigramCollocationFinder.from_words(quora_sentence)
bigram_quora = finder.nbest(BigramAssocMeasures.likelihood_ratio, 10)

universal = trigram_wiki + bigram_wiki + trigram_quora + bigram_quora

# with open("Trigram_Quora.csv", "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(trigram_quora)

# with open("Trigram_Wiki.csv", "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(trigram_wiki)

# with open("Bigram_Quora.csv", "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(bigram_quora)

# with open("Bigram_Wiki.csv", "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(bigram_wiki)
import spacy
nlp = spacy.load('en_core_web_lg')
import math
from subprocess import Popen, PIPE

similarity_mat = [[ nlp(' '.join(k)).similarity(nlp(' '.join(i))) for k in universal] for i in universal]
distance_mat = [[ int(math.ceil(((1/k)-1))) for k in i] for i in similarity_mat]

with open("Universal_Pool.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(universal)

with open('Distance_Matrix.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(distance_mat)

process = Popen(['Rscript', 'ddcrp/ddcrp.R'], stdout=PIPE, stderr=PIPE)
stdout, stderr = process.communicate()
print(stderr)

try:
    with open("cluster.txt", "r") as f:
        ddcrp_clust = f.readlines()

    ddcrp_clust = ddcrp_clust[1:]
    op = {}
    for i in ddcrp_clust:
        line = i.split(' ')
        if line[1] not in op:
            op[line[1]] = []
        op[line[1]].append(universal[int(line[2])-1])

    for key, value in op.items():
        val = list(set(value))
        op[key] = val

    opfile = open("output.txt", "a")
    for key, value in op.items():
        quora_score = 0
        wiki_score = 0
        val_list = []
        for i in value:
            val_list.extend(i)
            if i in bigram_quora or i in trigram_quora:
                quora_score += 1
            if i in bigram_wiki or i in trigram_wiki:
                wiki_score += 1
        quora_prob = quora_score/(quora_score + wiki_score)
        wiki_prob = wiki_score/(quora_score + wiki_score)
        ent_quora = quora_prob*(math.log(quora_prob, 2)) if quora_prob != 0 else 0
        ent_wiki = wiki_prob*(math.log(wiki_prob, 2)) if wiki_prob != 0 else 0
        score = ent_quora + ent_wiki
        if quora_prob == wiki_prob:
            lead = "Wiki + Quora"
        elif quora_prob > wiki_prob:
            lead = "Quora"
        elif quora_prob < wiki_prob:
            lead = "Wiki"

        opfile.write("Cluster " + str(list(set(val_list))) + ". Score = " + "{0:.2f} ".format(-1*score) + lead + "\n")

except Exception as e:
    print(e)
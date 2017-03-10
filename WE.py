class WordEmbeddings:
    def __init__(self):
        self.vocab = None
        self.embeddings = dict()

    def set_corpus_vocab(self, corpus):
        self.vocab = dict()
        for key, doc in corpus.iteritems():
            for sent in doc:
                for word in sent:
                    self.vocab[word] = ""

    def load_embedding_vectors(self, path):
        with open(path) as f:
            for line in f.readlines():
                line = line.split()
                word = line.pop(0)
                line = [float(x) for x in line]

                self.embeddings[word] = numpy.array(line)

        print('loaded embeddings with dim: '+str(len(embeddings[embeddings.keys()[0]])))
        print('vocab size: '+ str(len(embeddings)))

    def load_embedding_vectors_with_vocab(self, path,vocab):
        if self.vocab is None:
            print("run 'set_corpus_vocab' before loading embeddings ")
            return

        with open(path) as f:
            for line in f:
                line = line.split()
                word = line.pop(0)
                if word not in vocab:
                    continue

                line = [float(x) for x in line]

                embeddings[word] = numpy.array(line)

        print('loaded embeddings with dim: '+str(len(embeddings[embeddings.keys()[0]])))
        print('vocab size: '+ str(len(embeddings)))

    def get_doc_vec_centroid(self, document, embeddings):
        dim = self.embeddings[self.embeddings.keys()[0]].shape[0]
        doc_centroid = numpy.zeros(dim)
        word_count = 0
        for word in document:
            if word in self.embeddings:
                word_vec = self.embeddings[word]
                doc_centroid = numpy.add(doc_centroid,word_vec)
                word_count += 1

        doc_centroid /= word_count

        return doc_centroid

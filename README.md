# docsearch

To test this, initialize a Library object `lib = Library()`

Create a few "documents" `doc1 = "Once upon a time"
doc2 = "Let's move to Paris"
docs = [doc1, doc2]`

Add the documents to the library object `lib.addDocuments(docs)`

Search the library `lib.search("Paris")`

Take a look at what's stored in the library `lib.collections`

Take a look at the vector representation of the documents in the library `lib.word_vectors`

Take a look at the "Bag of Words" that resembles the library `lib.bow`

import pymongo

class db_mongo:
    def __init__(self):
        print("TESTE")

    def conecta(self):
        print("TESTE DE CONEXÃO")
        client = pymongo.MongoClient("mongodb+srv://admin:tcc123@cluster0-c8iqj.mongodb.net/test?retryWrites=true&w=majority")
        db = client.teste
        print(db)
        collection = db.tab1
        print(collection)
        nome = "Alleff"
        idade = 23
        test = {
            "name" : nome,
            "age" : idade
        }
        collection.insert_one(test).inserted_id
        #tst = db.reviews.insert_one(test)
        #print(db)
        #pprint(tst)
        client.close()
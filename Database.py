import pymongo

PATH = "C://Users//Dymytry//Desktop//TCC Alleff//git-TCC//"

class db_mongo:
    def __init__(self):
        print("TESTE")

    def conecta(self):
        # global client
        open_arq = PATH + "key.txt"
        filea = open(open_arq,"r")
        client = pymongo.MongoClient(str(filea.readline()))
        return client

    def kill_connection(self,client):
        print("CLOSING THE CONNECTION WITH DB.")
        client.close()

    def gamma2db(self,client,img_obj):
        db = client.tcc
        collection = db.images
        inserted = collection.insert_one(img_obj).inserted_id


    def gamma2db_mod(self,client,gamma):
        # client = db_mongo.conecta(self)
        # print(client)
        db = client.tcc
        collection = db.fotos3
        inserted = collection.insert_one(gamma).inserted_id
        print(inserted)

    def db2gamma(self):
        client = db_mongo.conecta(self)
        db = client.tcc
        collection = db.fotos2
        full_gamma = collection.find()
        return full_gamma

    def load_data(self):
        client = db_mongo.conecta(self)
        db = client.tcc
        collection = db.fotos
        for data in collection.find():
            print(data)

    def insert_data(self):
        db_mongo.conecta(self)
        db = client.tcc
        print(db)
        collection = db.fotos
        print(collection)
        nome = "Alleff"
        idade = 24
        test = {
            "name" : nome,
            "age" : idade
        }
        collection.insert_one(test).inserted_id
        client.close()

# mongo = db_mongo()
# teste = mongo.conecta()
# print(teste)
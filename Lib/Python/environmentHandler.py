import os
import shutil
import sqlite3
from flask import request
from passlib.hash import pbkdf2_sha256

from Lib.Python.Users import addUser
from werkzeug.utils import secure_filename

def getImportDirectory():
    try:
        with open("./db/globalSettings",'r') as file:
            for x in file.readlines():
                if "importDirectory" in x:
                    importDirectory= x.split("=")[1]
                    importDirectory = importDirectory.strip()
        return importDirectory
    except:
        return "NONE"

# Creates an empty database with a default 'admin' user
def createDB():
    with open("./db/imageSQL.db",'w') as file:
        file.write("")
    conn = sqlite3.connect('db/imageSQL.db')
    with open('Lib/sql/schema.sql') as f:
        conn.executescript(f.read())
    
    conn.commit()
    conn.close()
    addUser("admin", pbkdf2_sha256.hash("admin"), isAdmin=True)


def initialiseSettings():
    settings=["importDirectory=NONE"]

    with open("./db/globalSettings",'w') as file:
        file.writelines(settings)

def updateEnvironment(form):
    settings = []
    for x in form:
        if x.name == 'csrf_token':
            continue

        settings.append(x.name + "=" + x.data+'\n')

    with open("./db/globalSettings",'w') as file:
        file.writelines(settings)

# This returns a list of all files in the imports folder
def getImportImages():
    importDirectory = getImportDirectory()
    if importDirectory!= "NONE":
        print(os.listdir(importDirectory))
        return os.listdir(importDirectory)
    print("NONE")
        

def importImage(form, userID):

    # TODO: Remove this once the classification is in place
    def classifyImage():
        return "Foo"

    fileName= secure_filename(form.importFile.data.filename)



    conn = sqlite3.connect('./db/imageSQL.db')
    
    # Check if a file with the same name already exists in database.
    # If an existing file is found an index is added to the end e.g. someName(1).jpg
    # This will iterate until it produces a unique index
    tempfileName = fileName
    index = 1
    while True:
        
        if (conn.execute('SELECT filePath from Images WHERE filepath = ? ',(tempfileName, )).fetchone()) != None:
            
            tempfileName = fileName.split(".")[0] + "("+str(index)+")" + fileName.split(".")[1]
            index+=1
        else:
            fileName = tempfileName
            break
    
    
    # # will need to encrypt this before we save it
    form.importFile.data.save('./Images/'+ fileName)

    # This is a placeholder function until the ML is working
    imageClass = classifyImage()

    conn.execute('Insert into Images (filePath, Tag) VALUES (?, ?);',
                 (fileName,imageClass ))
    
    
    imageID = conn.execute("SELECT ImageID from Images WHERE filepath = ? ",
                           (fileName, )).fetchone()[0]
    


    # add the image to the user's catalog
    conn.execute('Insert into Catalog (UserID, ImageID) VALUES (?, ?)',
                 (userID, imageID))

    conn.commit()
    conn.close()

    

    return
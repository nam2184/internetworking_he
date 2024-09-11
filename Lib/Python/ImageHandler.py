import sqlite3

# returns an object with the image's data
def getImage(id):
    conn = sqlite3.connect('./db/imageSQL.db')
    info = conn.execute("select * from Images where ImageID = ?",
                        (id,)).fetchone()
    
    class Image():
        def __init__(self, info) -> None:
            self.id = info[0]
            self.filePath = info[1]
            self.tag = info[2]

    return Image(info)

def updateImage(form, userID):
    conn = sqlite3.connect('./db/imageSQL.db')
    
    conn.execute("UPDATE Images Set Name = ?, Artist = ?, MuseScoreLink = ?, YoutubeLink = ? WHERE ImageID = ?",
                 (form.name.data,
                  form.artist.data,
                  form.msLink.data if form.msLink.data!="" else None,
                  form.ytLink.data if form.ytLink.data!="" else None,
                  form.id.data,))
    
    conn.execute("UPDATE Catalog Set Tag = ? WHERE ImageID = ? AND UserID = ?",
                (form.tag.data if form.tag.data!="" else None,
                form.id.data,
                userID))
    conn.commit()
    conn.close()

# Returns all images names in the database. Primarily used for manual imports
def getAllImages():

    query = """
        SELECT ImageID, Name from Images
    """
    conn = sqlite3.connect('./db/imageSQL.db')
    images = conn.execute(query).fetchall()

    return images


def getImageID(name):
    conn = sqlite3.connect('./db/imageSQL.db')
    info = conn.execute("select ImageID from Images where Name = ?",
                        (name,)).fetchone()
    try:
        return info[0]
    # if no image is found, return a null values (it will be unsubscriptable so it'll fail the try)
    except:
        return None

def getArtistImages(artist, userID):
    query = '''
    SELECT  * from Images
        where ImageID IN 
            (Select ImageID from Catalog
            where UserID = ?
            
            )
        AND Artist = ?
'''

    conn = sqlite3.connect('./db/imageSQL.db')
    images = conn.execute(query, (userID, artist)).fetchall()

    try: 
        images[0]
        return images
    
    except:
        return None

def getTagImages(tag, userID):
    query = '''
    SELECT * from Images
        where ImageID IN 
            (Select ImageID from Catalog
            where UserID = ? AND Tag = ?
            
            )
'''

    conn = sqlite3.connect('./db/imageSQL.db')
    images = conn.execute(query, (userID, tag)).fetchall()

    try: 
        images[0]
        return images
    
    except:
        return None
    
def getImageTag(ImageID, userID):
    conn = sqlite3.connect('./db/imageSQL.db')
    images = conn.execute("SELECT Tag from Catalog where UserID = ? AND ImageID = ?", (userID, ImageID)).fetchone()
    
    try:
        return images[0]
    except:
        return None
    
# Used to delete images from the db
def removeImage(ImageID):
    conn = sqlite3.connect('./db/imageSQL.db')
    conn.execute("DELETE FROM Images WHERE ImageID=?",(ImageID,))
    conn.execute("DELETE FROM Catalog WHERE ImageID=?",(ImageID,))
    conn.commit()
    conn.close()


CREATE TABLE IF NOT EXISTS Users (
    UserId INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    userName TEXT UNIQUE NOT NULL,
    isAdmin TEXT not NULL
);
-- This will need to be updated for images
CREATE TABLE IF NOT EXISTS Images (
    ImageID INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE NOT NULL,
    filePath TEXT,
    Tag TEXT
);
-- This maps the image to the specific user
CREATE TABLE IF NOT EXISTS Catalog (
    UserID TEXT NOT NULL,
    ImageID TEXT NOT NULL,
    
    FOREIGN KEY (UserID) References Users (UserID),
    FOREIGN KEY (ImageID) References Images (ImageID) PRIMARY KEY (UserID, ImageID)
);

CREATE TABLE IF NOT EXISTS UserSignup (
    UserId INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    userName TEXT UNIQUE NOT NULL
);